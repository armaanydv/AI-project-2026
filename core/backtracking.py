"""
core/backtracking.py
====================
Backtracking CSP Solver for Timetable Scheduling.

Implements:
1. Baseline Plain Backtracking (chronological variable ordering, standard DFS)
2. Variable Ordering Heuristics:
   - MRV (Minimum Remaining Values / "Fail-First")
   - Degree Heuristic (most constraints on remaining unassigned variables, used as tie-breaker)
3. Inference:
   - Forward Checking (prunes conflicting values from remaining domains after each assignment)

Viva Concepts:
- Variable: Each required lecture/lab session
- Domain: All valid (day, start_period, room_id) triples
- Unary constraints (room type, capacity, lunch slot, teacher availability) are pruned upfront
- Binary constraints (no teacher/class/room overlap) are checked during search
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import time
from typing import Dict, List, Optional, Set, Tuple

from core.csp_model import Assignment, Dataset, Session, Timetable
from core.cost_function import count_violations, is_consistent, ViolationReport


@dataclass
class BacktrackingResult:
    """Statistics and final timetable produced by Backtracking."""
    timetable: Optional[Timetable]
    success: bool
    nodes_expanded: int
    time_taken: float
    report: Optional[ViolationReport] = None


class BacktrackingSolver:
    """
    Configurable CSP Backtracking Solver.

    Parameters
    ----------
    dataset : Dataset
        The problem instance.
    use_mrv : bool
        If True, selects unassigned variable with smallest remaining domain.
    use_degree : bool
        If True, uses degree heuristic as tie-breaker for MRV (or primary variable selector).
    use_forward_checking : bool
        If True, maintains remaining domains and backtracks early on wipeout.
    timeout_seconds : float
        Maximum seconds allowed before search aborts.
    max_nodes : int
        Maximum nodes (recursive calls) allowed before search aborts.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        use_mrv: bool = False,
        use_degree: bool = False,
        use_forward_checking: bool = False,
        timeout_seconds: float = 60.0,
        max_nodes: int = 100000,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.use_mrv = use_mrv
        self.use_degree = use_degree
        self.use_forward_checking = use_forward_checking
        self.timeout_seconds = timeout_seconds
        self.max_nodes = max_nodes
        self.seed = seed

        # Prioritize multi-period sessions (labs) first to fail-early on tight blocks
        self.sessions: List[Session] = sorted(self.dataset.get_sessions(), key=lambda s: s.duration, reverse=True)
        self.nodes_expanded: int = 0
        self.start_time: float = 0.0
        self.best_timetable: Timetable = Timetable()

        # Build initial domains respecting static unary constraints
        self.initial_domains: Dict[str, List[Assignment]] = self._build_initial_domains()

    def _build_initial_domains(self) -> Dict[str, List[Assignment]]:
        """
        Build domain for each session, enforcing static unary constraints:
        - Room type matches subject type (H5)
        - Room capacity >= class strength (H4)
        - Slot + duration fits within periods_per_day
        - Slot not on global break periods
        - Slot not on class lunch break (H9)
        - Slot not on teacher unavailable slots (H6)
        """
        domains: Dict[str, List[Assignment]] = {}
        ts = self.dataset.time_structure

        for session in self.sessions:
            valid_assignments: List[Assignment] = []
            cls = self.dataset.classes.get(session.class_id)
            subject = self.dataset.subjects.get(session.subject_id)
            teacher = self.dataset.teachers.get(session.teacher_id)

            if not cls or not subject:
                domains[session.session_id] = []
                continue

            for day in range(ts.num_days):
                for period in range(ts.periods_per_day):
                    # Check if session fits within the day
                    if period + session.duration > ts.periods_per_day:
                        continue

                    # Occupied slots for this candidate
                    occupied_slots = [(day, period + offset) for offset in range(session.duration)]

                    # Check global breaks
                    if any(p in ts.break_periods for (_, p) in occupied_slots):
                        continue

                    # H9: Lunch break
                    if any(p == cls.lunch_break_period for (_, p) in occupied_slots):
                        continue

                    # H6: Teacher unavailable slots
                    if teacher and any(s in teacher.unavailable_slots for s in occupied_slots):
                        continue

            # Room filtering: H5 (type) first, then H4 (capacity) with graceful fallback
            matching_rooms = [
                r for r in self.dataset.rooms.values()
                if (subject.subject_type == "lab" and r.room_type == "lab")
                or (subject.subject_type == "theory" and r.room_type == "classroom")
            ]
            if not matching_rooms:
                matching_rooms = list(self.dataset.rooms.values())

            capacity_rooms = [r for r in matching_rooms if r.capacity >= cls.strength]
            allowed_rooms = capacity_rooms if capacity_rooms else sorted(matching_rooms, key=lambda r: r.capacity, reverse=True)

            for day in range(ts.num_days):
                for period in range(ts.periods_per_day):
                    # Check if session fits within the day
                    if period + session.duration > ts.periods_per_day:
                        continue

                    # Occupied slots for this candidate
                    occupied_slots = [(day, period + offset) for offset in range(session.duration)]

                    # Check global breaks
                    if any(p in ts.break_periods for (_, p) in occupied_slots):
                        continue

                    # H9: Lunch break
                    if any(p == cls.lunch_break_period for (_, p) in occupied_slots):
                        continue

                    # H6: Teacher unavailable slots
                    if teacher and any(s in teacher.unavailable_slots for s in occupied_slots):
                        continue

                    for room in allowed_rooms:
                        valid_assignments.append(Assignment(day=day, start_period=period, room_id=room.room_id))

            # Graceful slot fallback: if teacher unavailability blocked all slots, allow non-lunch slots
            if not valid_assignments:
                for day in range(ts.num_days):
                    for period in range(ts.periods_per_day):
                        if period + session.duration > ts.periods_per_day:
                            continue
                        occupied_slots = [(day, period + offset) for offset in range(session.duration)]
                        if any(p in ts.break_periods for (_, p) in occupied_slots):
                            continue
                        if any(p == cls.lunch_break_period for (_, p) in occupied_slots):
                            continue
                        for room in allowed_rooms:
                            valid_assignments.append(Assignment(day=day, start_period=period, room_id=room.room_id))

            domains[session.session_id] = valid_assignments

        return domains

    def _get_degree(self, session: Session, unassigned: Set[str]) -> int:
        """
        Degree Heuristic: Number of constraints between this variable
        and other currently unassigned variables.
        Sessions sharing the same teacher or class constrain each other.
        """
        degree = 0
        for sid in unassigned:
            if sid == session.session_id:
                continue
            other = next(s for s in self.sessions if s.session_id == sid)
            if (other.teacher_id == session.teacher_id and session.teacher_id) or (other.class_id == session.class_id):
                degree += 1
        return degree

    def _select_variable(
        self,
        unassigned: List[Session],
        domains: Dict[str, List[Assignment]],
    ) -> Session:
        """
        Select next variable to assign.
        Baseline: first in list.
        With heuristics: MRV, tie-broken by Degree Heuristic.
        """
        if not self.use_mrv and not self.use_degree:
            return unassigned[0]

        unassigned_ids = {s.session_id for s in unassigned}

        def key_func(s: Session) -> Tuple[int, int]:
            # Primary: MRV (fewest remaining values) if enabled, else 0
            mrv_val = len(domains[s.session_id]) if self.use_mrv else 0
            # Tie-breaker: Degree heuristic (highest degree first -> negative degree)
            degree_val = -self._get_degree(s, unassigned_ids) if self.use_degree else 0
            return (mrv_val, degree_val)

        return min(unassigned, key=key_func)

    def _forward_check(
        self,
        assigned_session: Session,
        assignment: Assignment,
        unassigned: List[Session],
        domains: Dict[str, List[Assignment]],
    ) -> Optional[Dict[str, List[Assignment]]]:
        """
        Forward Checking: Prune assignments from unassigned variables' domains
        that clash with the newly assigned session.
        Returns new pruned domains dict, or None if domain wipeout occurred.
        """
        new_domains = {sid: list(vals) for sid, vals in domains.items()}
        occupied_slots = set(assignment.occupied_slots(assigned_session.duration))

        for other_sess in unassigned:
            if other_sess.session_id == assigned_session.session_id:
                continue

            filtered_vals: List[Assignment] = []
            for val in new_domains[other_sess.session_id]:
                other_slots = set(val.occupied_slots(other_sess.duration))
                common = occupied_slots.intersection(other_slots)

                if common:
                    # H1 clash: same teacher
                    if other_sess.teacher_id == assigned_session.teacher_id and assigned_session.teacher_id:
                        continue
                    # H2 clash: same class
                    if other_sess.class_id == assigned_session.class_id:
                        continue
                    # H3 clash: same room
                    if val.room_id == assignment.room_id:
                        continue

                filtered_vals.append(val)

            # Domain wipeout -> failure, trigger backtrack
            if not filtered_vals:
                return None

            new_domains[other_sess.session_id] = filtered_vals

        return new_domains

    def _complete_partial(self, partial_tt: Timetable) -> Timetable:
        """
        Complete any unassigned sessions in partial_tt by selecting
        the candidate value from the variable's domain that minimizes clashes.
        Guarantees that a complete, rendered timetable is returned even if
        exact exhaustive search reached timeout on large datasets.
        """
        completed = partial_tt.copy()
        session_map = {s.session_id: s for s in self.sessions}
        unassigned = [s for s in self.sessions if s.session_id not in completed.assignments]

        for s in unassigned:
            dom = self.initial_domains.get(s.session_id, [])
            if not dom:
                # Emergency fallback if domain is somehow empty
                first_room = list(self.dataset.rooms.keys())[0] if self.dataset.rooms else "R01"
                completed.assign(s.session_id, Assignment(day=0, start_period=0, room_id=first_room))
                continue

            best_val = dom[0]
            best_clashes = 999999
            # Sample up to 20 candidate values for speed
            sample_candidates = dom[:20] if len(dom) > 20 else dom
            for val in sample_candidates:
                clashes = 0
                occupied = set(val.occupied_slots(s.duration))
                for exist_sid, exist_asgn in completed.assignments.items():
                    exist_sess = session_map.get(exist_sid)
                    if not exist_sess:
                        continue
                    if occupied.intersection(exist_asgn.occupied_slots(exist_sess.duration)):
                        if (exist_sess.teacher_id == s.teacher_id and s.teacher_id) or (exist_sess.class_id == s.class_id) or (exist_asgn.room_id == val.room_id):
                            clashes += 1
                if clashes < best_clashes:
                    best_clashes = clashes
                    best_val = val
                if clashes == 0:
                    break

            completed.assign(s.session_id, best_val)

        return completed

    def solve(self) -> BacktrackingResult:
        """Execute the backtracking search."""
        random.seed(self.seed)
        self.nodes_expanded = 0
        self.start_time = time.time()
        self.best_timetable = Timetable()

        timetable = Timetable()
        domains = {sid: list(vals) for sid, vals in self.initial_domains.items()}

        success = self._backtrack(timetable, domains)
        elapsed = time.time() - self.start_time

        if success and timetable.is_complete(self.sessions):
            report = count_violations(timetable, self.sessions, self.dataset)
            return BacktrackingResult(
                timetable=timetable,
                success=True,
                nodes_expanded=self.nodes_expanded,
                time_taken=elapsed,
                report=report,
            )
        else:
            # Complete the best partial timetable found during search
            final_tt = self._complete_partial(self.best_timetable)
            report = count_violations(final_tt, self.sessions, self.dataset)
            return BacktrackingResult(
                timetable=final_tt,
                success=(report.hard_violations == 0),
                nodes_expanded=self.nodes_expanded,
                time_taken=elapsed,
                report=report,
            )

    def _backtrack(
        self,
        timetable: Timetable,
        domains: Dict[str, List[Assignment]],
    ) -> bool:
        """Recursive backtracking engine."""
        self.nodes_expanded += 1

        # Keep track of deepest/best partial timetable found
        if len(timetable.assignments) > len(self.best_timetable.assignments):
            self.best_timetable = timetable.copy()

        # Check bounds
        if time.time() - self.start_time > self.timeout_seconds:
            return False
        if self.nodes_expanded > self.max_nodes:
            return False

        # Termination: all sessions assigned
        if timetable.is_complete(self.sessions):
            return True

        # Unassigned sessions
        unassigned = [s for s in self.sessions if s.session_id not in timetable.assignments]
        var = self._select_variable(unassigned, domains)

        # Values to try for this variable
        candidate_values = domains[var.session_id]

        for val in candidate_values:
            if is_consistent(var, val, timetable, self.sessions, self.dataset):
                timetable.assign(var.session_id, val)

                if self.use_forward_checking:
                    remaining_unassigned = [s for s in unassigned if s.session_id != var.session_id]
                    pruned_domains = self._forward_check(var, val, remaining_unassigned, domains)
                    if pruned_domains is not None:
                        if self._backtrack(timetable, pruned_domains):
                            return True
                else:
                    if self._backtrack(timetable, domains):
                        return True

                # Undo assignment (backtrack)
                timetable.unassign(var.session_id)

                if time.time() - self.start_time > self.timeout_seconds:
                    return False

        return False
