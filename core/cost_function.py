"""
core/cost_function.py
=====================
THE shared violation counter for the AI Timetable Scheduler.

This is the single most important file in the project.
EVERY algorithm (Backtracking, Min-Conflicts, Simulated Annealing, Hybrid)
and the manual-edit UI all call the SAME functions defined here.

Two main public functions:
  1. count_violations(timetable, sessions, dataset)
        -> Scores a COMPLETE timetable. Returns a ViolationReport with:
           - hard_violations : total hard constraint violations (H1-H9)
           - soft_penalty    : total soft constraint penalty (S1-S5)
           - total_cost      : 1000 * hard_violations + soft_penalty
           - details         : dict breaking down each constraint
           - messages        : human-readable descriptions of violations

  2. is_consistent(session, assignment, partial_timetable, sessions, dataset)
        -> FAST incremental check used by Backtracking.
           Returns True only if adding this (session, assignment) pair to
           the partial timetable does NOT violate any hard constraint.
           Does NOT check soft constraints (irrelevant during search).

Cost formula (from spec):
    Cost = 1000 * hard_violations + soft_penalty

The factor 1000 ensures any hard violation is always worse than any number
of soft violations. This is called a WEIGHTED SUM objective function.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.csp_model import Assignment, Class, Dataset, Room, Session, Teacher, Timetable


# ---------------------------------------------------------------------------
# REPORT OBJECT  (what count_violations returns)
# ---------------------------------------------------------------------------

@dataclass
class ViolationReport:
    """
    Detailed breakdown of all constraint violations in a timetable.

    hard_violations : total count of hard constraint violations (H1-H9)
    soft_penalty    : total soft penalty score (S1-S5)
    total_cost      : 1000 * hard_violations + soft_penalty
    details         : dict mapping constraint id to count/penalty value
                      e.g. {"H1": 2, "H3": 0, "S1": 15, ...}
    messages        : human-readable list of violation descriptions
                      (shown in the UI manual-edit conflict panel)
    """
    hard_violations: int = 0
    soft_penalty: int = 0
    details: Dict[str, int] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

    @property
    def total_cost(self) -> int:
        """Cost = 1000 * hard + soft  (hard violations dominate)."""
        return 1000 * self.hard_violations + self.soft_penalty

    def add_hard(self, constraint_id: str, count: int, msg: str = "") -> None:
        """Record a hard constraint violation."""
        self.hard_violations += count
        self.details[constraint_id] = self.details.get(constraint_id, 0) + count
        if msg and count > 0:
            self.messages.append(f"[{constraint_id}] {msg}")

    def add_soft(self, constraint_id: str, penalty: int, msg: str = "") -> None:
        """Record a soft constraint penalty."""
        self.soft_penalty += penalty
        self.details[constraint_id] = self.details.get(constraint_id, 0) + penalty
        if msg and penalty > 0:
            self.messages.append(f"[{constraint_id}] {msg}")


# ---------------------------------------------------------------------------
# INTERNAL HELPER: build lookup tables
# ---------------------------------------------------------------------------

def _build_lookups(
    timetable: Timetable,
    sessions: List[Session],
    dataset: Dataset,
) -> Tuple[
    Dict[str, Session],                       # session_id -> Session
    Dict[Tuple[int, int, str], List[str]],    # (day,period,teacher_id) -> [session_ids]
    Dict[Tuple[int, int, str], List[str]],    # (day,period,class_id)   -> [session_ids]
    Dict[Tuple[int, int, str], List[str]],    # (day,period,room_id)    -> [session_ids]
    Dict[str, List[Tuple[int, int]]],         # session_id -> [(day,period),...]
]:
    """
    Build index dicts for O(1) conflict lookups.

    Rather than doing O(n^2) nested loops to find clashes, we index every
    assigned session by (day, period, teacher/class/room). Then checking for
    a clash is just: len(index[key]) > 1.

    Returns 5 dicts used by the hard-constraint checkers.
    """
    session_map: Dict[str, Session] = {s.session_id: s for s in sessions}

    # Maps: (day, period, entity_id) -> list of session_ids occupying that slot
    teacher_index: Dict[Tuple[int, int, str], List[str]] = defaultdict(list)
    class_index:   Dict[Tuple[int, int, str], List[str]] = defaultdict(list)
    room_index:    Dict[Tuple[int, int, str], List[str]] = defaultdict(list)

    # Map: session_id -> list of (day, period) it occupies (>1 for labs)
    session_slots: Dict[str, List[Tuple[int, int]]] = {}

    for sid, asgn in timetable.assignments.items():
        sess = session_map.get(sid)
        if sess is None:
            continue  # unknown session - skip

        occupied = asgn.occupied_slots(sess.duration)
        session_slots[sid] = occupied

        for (day, period) in occupied:
            teacher_index[(day, period, sess.teacher_id)].append(sid)
            class_index  [(day, period, sess.class_id)  ].append(sid)
            room_index   [(day, period, asgn.room_id)   ].append(sid)

    return session_map, teacher_index, class_index, room_index, session_slots


# ===========================================================================
# MAIN PUBLIC FUNCTION 1: full timetable scorer
# ===========================================================================

def count_violations(
    timetable: Timetable,
    sessions: List[Session],
    dataset: Dataset,
) -> ViolationReport:
    """
    Score a (complete or partial) timetable against all hard + soft constraints.

    Parameters
    ----------
    timetable : Timetable
        The assignment to evaluate. May be partial (some sessions unassigned).
    sessions  : List[Session]
        All CSP variables for this problem (from dataset.get_sessions()).
    dataset   : Dataset
        The problem input (subjects, teachers, rooms, classes, time_structure).

    Returns
    -------
    ViolationReport
        Contains hard_violations, soft_penalty, total_cost, details, messages.
    """
    report = ViolationReport()

    session_map, teacher_idx, class_idx, room_idx, session_slots = (
        _build_lookups(timetable, sessions, dataset)
    )

    # -----------------------------------------------------------------------
    # HARD CONSTRAINT H1: Teacher can't teach two sessions at once
    # -----------------------------------------------------------------------
    h1_count = 0
    for (day, period, teacher_id), sids in teacher_idx.items():
        if len(sids) > 1:
            h1_count += len(sids) - 1
    report.add_hard("H1", h1_count,
        f"{h1_count} teacher double-booking(s) found" if h1_count else "")

    # -----------------------------------------------------------------------
    # HARD CONSTRAINT H2: Class can't have two sessions at once
    # -----------------------------------------------------------------------
    h2_count = 0
    for (day, period, class_id), sids in class_idx.items():
        if len(sids) > 1:
            h2_count += len(sids) - 1
    report.add_hard("H2", h2_count,
        f"{h2_count} class clash(es) found" if h2_count else "")

    # -----------------------------------------------------------------------
    # HARD CONSTRAINT H3: Room can't host two sessions at once
    # -----------------------------------------------------------------------
    h3_count = 0
    for (day, period, room_id), sids in room_idx.items():
        if len(sids) > 1:
            h3_count += len(sids) - 1
    report.add_hard("H3", h3_count,
        f"{h3_count} room double-booking(s) found" if h3_count else "")

    # -----------------------------------------------------------------------
    # HARD CONSTRAINTS H4, H5, H6, H8, H9: per-session checks
    # -----------------------------------------------------------------------
    h4_count = h5_count = h6_count = h8_count = h9_count = 0

    # Track teacher period counts for H8
    teacher_period_count: Dict[str, int] = defaultdict(int)

    for sid, asgn in timetable.assignments.items():
        sess = session_map.get(sid)
        if sess is None:
            continue

        room    = dataset.rooms.get(asgn.room_id)
        cls     = dataset.classes.get(sess.class_id)
        teacher = dataset.teachers.get(sess.teacher_id)
        subject = dataset.subjects.get(sess.subject_id)
        occupied = session_slots.get(sid, [(asgn.day, asgn.start_period)])

        # H4: Room capacity >= class strength
        if room and cls and room.capacity < cls.strength:
            h4_count += 1

        # H5: Labs in lab rooms, theory in classrooms
        if room and subject:
            if subject.subject_type == "lab" and room.room_type != "lab":
                h5_count += 1
            elif subject.subject_type == "theory" and room.room_type != "classroom":
                h5_count += 1

        # H6: Teacher available in all occupied slots
        if teacher:
            for (day, period) in occupied:
                if (day, period) in teacher.unavailable_slots:
                    h6_count += 1
                    break

        # H8: Teacher period count
        teacher_period_count[sess.teacher_id] += sess.duration

        # H9: Class lunch break must be empty
        if cls:
            lunch = cls.lunch_break_period
            for (day, period) in occupied:
                if period == lunch:
                    h9_count += 1
                    break

    # H8 final check: is any teacher over their max load?
    for teacher_id, total_periods in teacher_period_count.items():
        teacher = dataset.teachers.get(teacher_id)
        if teacher and total_periods > teacher.max_weekly_load:
            h8_count += 1

    report.add_hard("H4", h4_count, f"{h4_count} capacity violation(s)" if h4_count else "")
    report.add_hard("H5", h5_count, f"{h5_count} room-type mismatch(es)" if h5_count else "")
    report.add_hard("H6", h6_count, f"{h6_count} teacher unavailability violation(s)" if h6_count else "")
    report.add_hard("H8", h8_count, f"{h8_count} teacher(s) exceed max weekly load" if h8_count else "")
    report.add_hard("H9", h9_count, f"{h9_count} lunch-break slot(s) occupied" if h9_count else "")

    # -----------------------------------------------------------------------
    # HARD CONSTRAINT H7: Every subject gets exactly its required weekly hours
    # Unassigned sessions are H7 violations (timetable is incomplete).
    # -----------------------------------------------------------------------
    h7_count = sum(1 for s in sessions if s.session_id not in timetable.assignments)
    report.add_hard("H7", h7_count,
        f"{h7_count} session(s) not yet scheduled" if h7_count else "")

    # =======================================================================
    # SOFT CONSTRAINTS (S1-S5) — penalty-based, not binary
    # =======================================================================

    # S1: Minimize gaps in a class's daily schedule
    s1_penalty = _check_s1_gaps(timetable, sessions, session_map, dataset)
    report.add_soft("S1", s1_penalty,
        f"Gap penalty: {s1_penalty}" if s1_penalty else "")

    # S2: Spread subject lectures across different days
    s2_penalty = _check_s2_spread(timetable, sessions, session_map, dataset)
    report.add_soft("S2", s2_penalty,
        f"Same-day repeat penalty: {s2_penalty}" if s2_penalty else "")

    # S4: Respect teacher preferences
    s4_penalty = _check_s4_preferences(timetable, sessions, session_map, dataset)
    report.add_soft("S4", s4_penalty,
        f"Teacher preference penalty: {s4_penalty}" if s4_penalty else "")

    # S5: Avoid more than 2 consecutive lectures for a teacher
    s5_penalty = _check_s5_consecutive(timetable, sessions, session_map, dataset)
    report.add_soft("S5", s5_penalty,
        f"Consecutive-lecture penalty: {s5_penalty}" if s5_penalty else "")

    return report


# ---------------------------------------------------------------------------
# SOFT CONSTRAINT HELPERS
# ---------------------------------------------------------------------------

def _check_s1_gaps(
    timetable: Timetable,
    sessions: List[Session],
    session_map: Dict[str, Session],
    dataset: Dataset,
) -> int:
    """
    S1: Minimize free periods (gaps) in a class's day.
    Penalty: +1 per gap period between first and last class on that day.
    """
    class_day_periods: Dict[str, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))

    for sid, asgn in timetable.assignments.items():
        sess = session_map.get(sid)
        if not sess:
            continue
        for (day, period) in asgn.occupied_slots(sess.duration):
            class_day_periods[sess.class_id][day].add(period)

    penalty = 0
    for class_id, day_map in class_day_periods.items():
        cls = dataset.classes.get(class_id)
        lunch = cls.lunch_break_period if cls else -1
        for day, periods in day_map.items():
            if len(periods) < 2:
                continue
            earliest = min(periods)
            latest   = max(periods)
            # Gap periods excluding the designated lunch period
            total_span = set(range(earliest, latest + 1))
            gaps = total_span - periods
            if lunch in gaps:
                gaps.remove(lunch)  # Lunch doesn't count as an undesirable gap
            penalty += len(gaps)
    return penalty


def _check_s2_spread(
    timetable: Timetable,
    sessions: List[Session],
    session_map: Dict[str, Session],
    dataset: Dataset,
) -> int:
    """
    S2: A subject's lectures should not appear twice on the same day (unless lab).
    Penalty: +5 per extra theory session beyond the first on the same day.
    """
    count_map: Dict[Tuple[str, str, int], int] = defaultdict(int)

    for sid, asgn in timetable.assignments.items():
        sess = session_map.get(sid)
        if not sess:
            continue
        subj = dataset.subjects.get(sess.subject_id)
        if subj and subj.subject_type == "lab":
            continue  # lab is exempt from same-day spread penalty
        key = (sess.class_id, sess.subject_id, asgn.day)
        count_map[key] += 1

    penalty = 0
    for count in count_map.values():
        if count > 1:
            penalty += (count - 1) * 5
    return penalty


def _check_s4_preferences(
    timetable: Timetable,
    sessions: List[Session],
    session_map: Dict[str, Session],
    dataset: Dataset,
) -> int:
    """
    S4: Respect teacher time preferences.
    """
    penalty = 0
    for sid, asgn in timetable.assignments.items():
        sess = session_map.get(sid)
        if not sess:
            continue
        teacher = dataset.teachers.get(sess.teacher_id)
        if not teacher or not teacher.preferences:
            continue
        for (day, period) in asgn.occupied_slots(sess.duration):
            penalty += teacher.preferences.get((day, period), 0)
    return penalty


def _check_s5_consecutive(
    timetable: Timetable,
    sessions: List[Session],
    session_map: Dict[str, Session],
    dataset: Dataset,
) -> int:
    """
    S5: Avoid more than 2 consecutive lectures for a teacher on any day.
    Penalty: +3 per run exceeding 2 consecutive periods.
    """
    teacher_day: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

    for sid, asgn in timetable.assignments.items():
        sess = session_map.get(sid)
        if not sess:
            continue
        for (day, period) in asgn.occupied_slots(sess.duration):
            teacher_day[sess.teacher_id][day].append(period)

    penalty = 0
    for teacher_id, day_map in teacher_day.items():
        for day, periods in day_map.items():
            sorted_periods = sorted(set(periods))
            run_length = 1
            for i in range(1, len(sorted_periods)):
                if sorted_periods[i] == sorted_periods[i - 1] + 1:
                    run_length += 1
                    if run_length > 2:
                        penalty += 3
                else:
                    run_length = 1
    return penalty


# ===========================================================================
# MAIN PUBLIC FUNCTION 2: fast incremental consistency check for Backtracking
# ===========================================================================

def is_consistent(
    session: Session,
    assignment: Assignment,
    partial_timetable: Timetable,
    sessions: List[Session],
    dataset: Dataset,
) -> bool:
    """
    Fast check: would adding (session -> assignment) to partial_timetable
    violate any HARD constraint with an already-assigned session?

    Used by Backtracking BEFORE committing an assignment.
    Much faster than calling count_violations() on the full timetable
    because we only check the NEW session against existing ones.
    """
    sess = session
    asgn = assignment
    new_slots = asgn.occupied_slots(sess.duration)

    # Check boundaries of the schedule (cannot go past last period of day)
    for (day, period) in new_slots:
        if period >= dataset.time_structure.periods_per_day:
            return False
        if period in dataset.time_structure.break_periods:
            return False

    session_map = {s.session_id: s for s in sessions}

    # H9: Is any slot the lunch break for this class?
    cls = dataset.classes.get(sess.class_id)
    if cls:
        for (day, period) in new_slots:
            if period == cls.lunch_break_period:
                return False

    # H4: Room capacity >= class strength (enforced only if an adequate room exists)
    room = dataset.rooms.get(asgn.room_id)
    subject = dataset.subjects.get(sess.subject_id)
    if room and cls and room.capacity < cls.strength:
        stype = subject.subject_type if subject else "theory"
        has_adequate = any(
            r.capacity >= cls.strength and (
                (stype == "lab" and r.room_type == "lab") or
                (stype == "theory" and r.room_type == "classroom") or
                (not subject)
            )
            for r in dataset.rooms.values()
        )
        if has_adequate:
            return False

    # H5: Room type must match subject type
    subject = dataset.subjects.get(sess.subject_id)
    if room and subject:
        if subject.subject_type == "lab" and room.room_type != "lab":
            return False
        if subject.subject_type == "theory" and room.room_type != "classroom":
            return False

    # H6: Teacher available in all slots this session occupies
    teacher = dataset.teachers.get(sess.teacher_id)
    if teacher:
        for (day, period) in new_slots:
            if (day, period) in teacher.unavailable_slots:
                return False

    # Check against every already-assigned session for clashes (H1, H2, H3)
    new_slot_set = set(new_slots)
    for existing_sid, existing_asgn in partial_timetable.assignments.items():
        existing_sess = session_map.get(existing_sid)
        if existing_sess is None:
            continue
        existing_slots = set(existing_asgn.occupied_slots(existing_sess.duration))
        common_slots = new_slot_set.intersection(existing_slots)

        if common_slots:
            # H1: Same teacher?
            if existing_sess.teacher_id == sess.teacher_id and sess.teacher_id:
                return False
            # H2: Same class?
            if existing_sess.class_id == sess.class_id:
                return False
            # H3: Same room?
            if existing_asgn.room_id == asgn.room_id:
                return False

    # H8: Teacher load check
    if teacher:
        current_load = sum(
            session_map[sid].duration
            for sid in partial_timetable.assignments
            if session_map.get(sid) and session_map[sid].teacher_id == sess.teacher_id
        )
        if current_load + sess.duration > teacher.max_weekly_load:
            return False

    return True
