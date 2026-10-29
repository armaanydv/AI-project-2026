"""
core/min_conflicts.py
=====================
Min-Conflicts Local Search Algorithm for Timetable Scheduling.

Algorithm Overview (Russell & Norvig, AIMA):
1. Initial Assignment:
   Start with a COMPLETE assignment where every session is assigned a slot
   from its unary-valid domain (chosen randomly or greedily). Clashes may exist.
2. Main Loop (up to max_iterations):
   - If current assignment has 0 hard violations, return SUCCESS.
   - Identify all CONFLICTED variables (sessions involved in a clash).
   - Randomly pick one conflicted variable.
   - Evaluate all candidate values in its domain; pick the value that MINIMIZES
     conflicts (breaking ties randomly).
   - Plateau Escape: With probability `random_walk_prob`, pick a random value
     from the domain instead of the min-conflicts value to escape local minima.
3. Cost History:
   Tracks cost at each iteration for analysis and plotting.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import random
import time
from typing import Dict, List, Optional, Set, Tuple

from core.csp_model import Assignment, Dataset, Session, Timetable
from core.cost_function import count_violations, ViolationReport


@dataclass
class MinConflictsResult:
    """Statistics, cost trajectory, and final timetable produced by Min-Conflicts."""
    timetable: Timetable
    success: bool
    iterations: int
    time_taken: float
    cost_history: List[int] = field(default_factory=list)
    report: Optional[ViolationReport] = None


class MinConflictsSolver:
    """
    Min-Conflicts Local Search Solver.

    Parameters
    ----------
    dataset : Dataset
        The problem instance.
    max_iterations : int
        Maximum number of local search steps before termination.
    random_walk_prob : float
        Probability (e.g., 0.05) of choosing a random value to escape plateaus.
    seed : int
        Random seed for reproducible experiments.
    """

    def __init__(
        self,
        dataset: Dataset,
        max_iterations: int = 2000,
        random_walk_prob: float = 0.06,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.max_iterations = max_iterations
        self.random_walk_prob = random_walk_prob
        self.seed = seed

        self.sessions: List[Session] = self.dataset.get_sessions()
        self.domains: Dict[str, List[Assignment]] = self._build_domains()

    def _build_domains(self) -> Dict[str, List[Assignment]]:
        """
        Pre-filter candidate values by static unary constraints (capacity, type, lunch, unavail).
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
                    if period + session.duration > ts.periods_per_day:
                        continue

                    occupied_slots = [(day, period + offset) for offset in range(session.duration)]

                    if any(p in ts.break_periods for (_, p) in occupied_slots):
                        continue

                    if any(p == cls.lunch_break_period for (_, p) in occupied_slots):
                        continue

                    if teacher and any(s in teacher.unavailable_slots for s in occupied_slots):
                        continue

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
                    if period + session.duration > ts.periods_per_day:
                        continue

                    occupied_slots = [(day, period + offset) for offset in range(session.duration)]

                    if any(p in ts.break_periods for (_, p) in occupied_slots):
                        continue

                    if any(p == cls.lunch_break_period for (_, p) in occupied_slots):
                        continue

                    if teacher and any(s in teacher.unavailable_slots for s in occupied_slots):
                        continue

                    for room in allowed_rooms:
                        valid_assignments.append(Assignment(day=day, start_period=period, room_id=room.room_id))

            # Graceful fallback: allow non-lunch slots if all slots were blocked
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

    def _initial_assignment(self, timetable: Timetable) -> None:
        """
        Create a complete initial assignment by assigning each session a random value from its domain.
        """
        for session in self.sessions:
            vals = self.domains.get(session.session_id, [])
            if vals:
                timetable.assign(session.session_id, random.choice(vals))

    def _get_conflicted_variables(self, timetable: Timetable) -> List[str]:
        """
        Identify session IDs that are currently involved in at least one conflict.
        Checks teacher clashes, class clashes, and room clashes.
        """
        session_map = {s.session_id: s for s in self.sessions}

        # Index occupied slots by teacher, class, and room
        teacher_slots: Dict[Tuple[int, int, str], List[str]] = defaultdict(list)
        class_slots:   Dict[Tuple[int, int, str], List[str]] = defaultdict(list)
        room_slots:    Dict[Tuple[int, int, str], List[str]] = defaultdict(list)

        for sid, asgn in timetable.assignments.items():
            sess = session_map.get(sid)
            if not sess:
                continue
            for slot in asgn.occupied_slots(sess.duration):
                if sess.teacher_id:
                    teacher_slots[(slot[0], slot[1], sess.teacher_id)].append(sid)
                class_slots[(slot[0], slot[1], sess.class_id)].append(sid)
                room_slots[(slot[0], slot[1], asgn.room_id)].append(sid)

        conflicted: Set[str] = set()

        for sids in teacher_slots.values():
            if len(sids) > 1:
                conflicted.update(sids)
        for sids in class_slots.values():
            if len(sids) > 1:
                conflicted.update(sids)
        for sids in room_slots.values():
            if len(sids) > 1:
                conflicted.update(sids)

        return list(conflicted)

    def _count_conflicts_for_assignment(
        self,
        session: Session,
        candidate: Assignment,
        timetable: Timetable,
        session_map: Dict[str, Session],
    ) -> int:
        """
        Count how many clashes 'candidate' causes with other assigned sessions.
        """
        clashes = 0
        cand_slots = set(candidate.occupied_slots(session.duration))

        for other_sid, other_asgn in timetable.assignments.items():
            if other_sid == session.session_id:
                continue
            other_sess = session_map.get(other_sid)
            if not other_sess:
                continue

            other_slots = set(other_asgn.occupied_slots(other_sess.duration))
            common = cand_slots.intersection(other_slots)
            if not common:
                continue

            # Teacher clash
            if other_sess.teacher_id == session.teacher_id and session.teacher_id:
                clashes += len(common)
            # Class clash
            if other_sess.class_id == session.class_id:
                clashes += len(common)
            # Room clash
            if other_asgn.room_id == candidate.room_id:
                clashes += len(common)

        return clashes

    def solve(self) -> MinConflictsResult:
        """Run the Min-Conflicts local search loop."""
        random.seed(self.seed)
        start_time = time.time()
        session_map = {s.session_id: s for s in self.sessions}

        timetable = Timetable()
        self._initial_assignment(timetable)

        cost_history: List[int] = []
        best_timetable = timetable.copy()
        best_report = count_violations(best_timetable, self.sessions, self.dataset)
        best_cost = best_report.total_cost

        for iteration in range(self.max_iterations):
            report = count_violations(timetable, self.sessions, self.dataset)
            current_cost = report.total_cost
            cost_history.append(current_cost)

            # Keep track of best solution seen so far
            if current_cost < best_cost:
                best_cost = current_cost
                best_timetable = timetable.copy()
                best_report = report

            # Goal test: 0 hard violations found
            if report.hard_violations == 0:
                elapsed = time.time() - start_time
                return MinConflictsResult(
                    timetable=timetable,
                    success=True,
                    iterations=iteration + 1,
                    time_taken=elapsed,
                    cost_history=cost_history,
                    report=report,
                )

            # Get conflicted variables
            conflicted_sids = self._get_conflicted_variables(timetable)
            if not conflicted_sids:
                # If no slot clashes but other hard violations exist (e.g. teacher load)
                conflicted_sids = [s.session_id for s in self.sessions]

            # Choose one conflicted variable uniformly at random
            chosen_sid = random.choice(conflicted_sids)
            chosen_session = session_map[chosen_sid]
            candidate_vals = self.domains.get(chosen_sid, [])

            if not candidate_vals:
                continue

            # Plateau escape: random walk with small probability
            if random.random() < self.random_walk_prob:
                new_val = random.choice(candidate_vals)
            else:
                # Evaluate conflicts for each candidate value
                val_conflicts: List[Tuple[int, Assignment]] = []
                for val in candidate_vals:
                    c = self._count_conflicts_for_assignment(chosen_session, val, timetable, session_map)
                    val_conflicts.append((c, val))

                min_c = min(vc[0] for vc in val_conflicts)
                # Tie-breaking rule: pick uniformly at random among all values with min conflicts
                best_vals = [vc[1] for vc in val_conflicts if vc[0] == min_c]
                new_val = random.choice(best_vals)

            timetable.assign(chosen_sid, new_val)

        # Iteration limit reached
        elapsed = time.time() - start_time
        return MinConflictsResult(
            timetable=best_timetable,
            success=(best_report.hard_violations == 0),
            iterations=self.max_iterations,
            time_taken=elapsed,
            cost_history=cost_history,
            report=best_report,
        )
