"""
core/simulated_annealing.py
===========================
Simulated Annealing (SA) Metaheuristic Solver for Timetable Scheduling.

Key Components (Viva Defense):
------------------------------
1. State Representation:
   A complete timetable mapping every Session -> Assignment.
2. Objective Function:
   Cost = 1000 * hard_violations + soft_penalty (reused from core/cost_function.py).
3. Neighbourhood Operators:
   - Relocate Move (60% probability): Change a session's slot/room to a different valid domain value.
   - Swap Move (40% probability): Swap assignments between two sessions of identical duration.
4. Acceptance Criterion (Metropolis):
   - Delta = Cost(neighbor) - Cost(current)
   - If Delta < 0 (better): Always accept.
   - If Delta >= 0 (worse): Accept with probability P = exp(-Delta / Temperature).
5. Cooling Schedule (Geometric Cooling):
   T_{k+1} = alpha * T_k, where alpha is cooling rate (default 0.992).
6. Stopping Condition:
   T <= min_temp or iteration >= max_iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Dict, List, Optional, Tuple

from core.csp_model import Assignment, Dataset, Session, Timetable
from core.cost_function import count_violations, ViolationReport


@dataclass
class SimulatedAnnealingResult:
    """Statistics, cost trajectory, and final timetable produced by Simulated Annealing."""
    timetable: Timetable
    success: bool
    iterations: int
    time_taken: float
    cost_history: List[int] = field(default_factory=list)
    temperature_history: List[float] = field(default_factory=list)
    report: Optional[ViolationReport] = None


class SimulatedAnnealingSolver:
    """
    Simulated Annealing Solver for Timetable Scheduling.

    Parameters
    ----------
    dataset : Dataset
        The problem instance.
    initial_temp : float
        Starting temperature (default 1000.0). Higher allows more exploratory uphill moves.
    cooling_rate : float
        Geometric cooling rate alpha in T_{k+1} = alpha * T_k (default 0.992).
    min_temp : float
        Minimum temperature threshold for stopping (default 0.1).
    max_iterations : int
        Maximum search iterations allowed (default 3000).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.992,
        min_temp: float = 0.1,
        max_iterations: int = 3000,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.seed = seed

        self.sessions: List[Session] = self.dataset.get_sessions()
        self.domains: Dict[str, List[Assignment]] = self._build_domains()

    def _build_domains(self) -> Dict[str, List[Assignment]]:
        """Pre-filter candidate values by static unary constraints."""
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

    def _generate_initial_state(self) -> Timetable:
        """Create a complete initial timetable by picking a random domain value for each session."""
        timetable = Timetable()
        for session in self.sessions:
            vals = self.domains.get(session.session_id, [])
            if vals:
                timetable.assign(session.session_id, random.choice(vals))
        return timetable

    def _neighbor_move(self, timetable: Timetable) -> Tuple[Timetable, str]:
        """
        Generate a neighbor timetable via either Relocate or Swap.
        Returns: (neighbor_timetable, move_type)
        """
        neighbor = timetable.copy()
        if len(self.sessions) < 2:
            return neighbor, "none"

        # 60% chance Relocate, 40% chance Swap
        move_type = "relocate" if random.random() < 0.60 else "swap"

        if move_type == "relocate":
            # Pick a random session and change its assignment
            sess = random.choice(self.sessions)
            vals = self.domains.get(sess.session_id, [])
            if len(vals) > 1:
                current_asgn = neighbor.assignments.get(sess.session_id)
                alt_vals = [v for v in vals if v != current_asgn]
                if alt_vals:
                    neighbor.assign(sess.session_id, random.choice(alt_vals))
            return neighbor, "relocate"

        else:  # swap move
            # Pick two sessions with matching duration for clean exchange
            s1 = random.choice(self.sessions)
            s1_dur = s1.duration
            candidates_s2 = [s for s in self.sessions if s.session_id != s1.session_id and s.duration == s1_dur]

            if not candidates_s2:
                # Fallback to relocate if no equal duration partner
                return self._neighbor_move(timetable)

            s2 = random.choice(candidates_s2)
            asgn1 = neighbor.assignments.get(s1.session_id)
            asgn2 = neighbor.assignments.get(s2.session_id)

            if asgn1 and asgn2:
                # Check domain compatibility before swapping
                if asgn2 in self.domains.get(s1.session_id, []) and asgn1 in self.domains.get(s2.session_id, []):
                    neighbor.assign(s1.session_id, asgn2)
                    neighbor.assign(s2.session_id, asgn1)

            return neighbor, "swap"

    def solve(self, initial_timetable: Optional[Timetable] = None) -> SimulatedAnnealingResult:
        """
        Run Simulated Annealing.
        If `initial_timetable` is provided, starts from it (useful for Hybrid CSP + SA).
        Otherwise generates a random initial complete timetable.
        """
        random.seed(self.seed)
        start_time = time.time()

        if initial_timetable is not None:
            current_timetable = initial_timetable.copy()
        else:
            current_timetable = self._generate_initial_state()

        current_report = count_violations(current_timetable, self.sessions, self.dataset)
        current_cost = current_report.total_cost

        best_timetable = current_timetable.copy()
        best_cost = current_cost
        best_report = current_report

        cost_history: List[int] = [current_cost]
        temperature_history: List[float] = [self.initial_temp]

        temp = self.initial_temp
        iteration = 0

        while iteration < self.max_iterations and temp > self.min_temp:
            iteration += 1

            neighbor_timetable, _ = self._neighbor_move(current_timetable)
            neighbor_report = count_violations(neighbor_timetable, self.sessions, self.dataset)
            neighbor_cost = neighbor_report.total_cost

            delta_cost = neighbor_cost - current_cost

            # Metropolis Acceptance Criterion:
            # If delta < 0: neighbor is strictly better -> accept unconditionally
            # If delta >= 0: neighbor is worse -> accept with probability exp(-delta / T)
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                prob = math.exp(-delta_cost / max(temp, 1e-9))
                if random.random() < prob:
                    accept = True

            if accept:
                current_timetable = neighbor_timetable
                current_cost = neighbor_cost
                current_report = neighbor_report

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_timetable = current_timetable.copy()
                    best_report = current_report

            # Cool down
            temp *= self.cooling_rate

            cost_history.append(current_cost)
            temperature_history.append(temp)

            # Early exit if perfect solution (zero hard violations and low soft penalty)
            if best_report.hard_violations == 0 and best_report.soft_penalty == 0:
                break

        elapsed = time.time() - start_time
        return SimulatedAnnealingResult(
            timetable=best_timetable,
            success=(best_report.hard_violations == 0),
            iterations=iteration,
            time_taken=elapsed,
            cost_history=cost_history,
            temperature_history=temperature_history,
            report=best_report,
        )
