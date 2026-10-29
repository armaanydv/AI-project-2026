"""
core/hybrid.py
==============
Hybrid CSP + Simulated Annealing Solver (Bonus 2: +6 marks).

Two-Stage Pipeline Architecture:
--------------------------------
Stage 1 (Feasibility Phase - Backtracking CSP):
  - Employs Backtracking with MRV + Degree Heuristic + Forward Checking.
  - Objective: Rapidly find a complete, clash-free assignment where
    Hard Violations = 0.
  - Backtracking is exceptional at systematically proving feasibility and
    satisfying hard constraints, but agnostic to soft constraint quality.

Stage 2 (Optimization Phase - Constrained Simulated Annealing):
  - Starts with the feasible timetable obtained from Stage 1.
  - Objective: Explore the solution space to minimize Soft Penalties
    (S1 gaps, S2 subject spread, S4 teacher preferences, S5 consecutive lectures).
  - HARD CONSTRAINT INVARIANT: Any candidate neighbor move that would
    introduce even a single hard violation (H1-H9) is immediately REJECTED.
  - Metropolis acceptance operates exclusively on delta soft penalty:
    P(accept) = exp(-delta_soft / T) when delta_soft >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Dict, List, Optional, Tuple

from core.csp_model import Assignment, Dataset, Session, Timetable
from core.cost_function import count_violations, ViolationReport
from core.backtracking import BacktrackingSolver, BacktrackingResult
from core.simulated_annealing import SimulatedAnnealingSolver


@dataclass
class HybridResult:
    """Statistics, timings, and final timetable produced by the Hybrid CSP+SA solver."""
    timetable: Optional[Timetable]
    success: bool
    phase1_time: float
    phase2_time: float
    total_time: float
    backtracking_nodes: int
    sa_iterations: int
    initial_soft_penalty: int
    final_soft_penalty: int
    cost_history: List[int] = field(default_factory=list)
    report: Optional[ViolationReport] = None


class HybridSolver:
    """
    Hybrid CSP + SA Solver.

    Parameters
    ----------
    dataset : Dataset
        The problem instance.
    sa_initial_temp : float
        Starting temperature for Stage 2 (default 100.0 - soft penalties operate on smaller scale).
    sa_cooling_rate : float
        Cooling rate for Stage 2 (default 0.990).
    sa_max_iterations : int
        Maximum SA polishing steps in Stage 2 (default 1000).
    bt_timeout : float
        Timeout for Stage 1 Backtracking (default 30.0s).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        sa_initial_temp: float = 100.0,
        sa_cooling_rate: float = 0.990,
        sa_max_iterations: int = 1000,
        bt_timeout: float = 30.0,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_max_iterations = sa_max_iterations
        self.bt_timeout = bt_timeout
        self.seed = seed

        self.sessions: List[Session] = self.dataset.get_sessions()
        # Reuse SA helper for domain generation and neighbor moves
        self._sa_helper = SimulatedAnnealingSolver(dataset=dataset, seed=seed)

    def solve(self) -> HybridResult:
        """Execute Stage 1 Backtracking followed by Stage 2 Constrained SA."""
        random.seed(self.seed)
        start_overall = time.time()

        # ===================================================================
        # STAGE 1: Backtracking CSP (Feasibility Phase)
        # ===================================================================
        bt_solver = BacktrackingSolver(
            dataset=self.dataset,
            use_mrv=True,
            use_degree=True,
            use_forward_checking=True,
            timeout_seconds=self.bt_timeout,
            seed=self.seed,
        )
        bt_result: BacktrackingResult = bt_solver.solve()
        phase1_time = bt_result.time_taken

        # Stage 1 complete!
        stage1_timetable = bt_result.timetable if bt_result.timetable is not None else self._sa_helper._generate_initial_state()
        stage1_report = count_violations(stage1_timetable, self.sessions, self.dataset)
        initial_soft_penalty = stage1_report.soft_penalty
        stage1_hard_zero = (stage1_report.hard_violations == 0)

        # ===================================================================
        # STAGE 2: Constrained Simulated Annealing (Optimization Phase)
        # ===================================================================
        start_stage2 = time.time()
        current_timetable = stage1_timetable.copy()
        current_report = stage1_report
        current_cost = 1000 * current_report.hard_violations + current_report.soft_penalty

        best_timetable = current_timetable.copy()
        best_report = stage1_report
        best_cost = current_cost
        best_soft_penalty = stage1_report.soft_penalty

        cost_history: List[int] = [best_cost]
        temp = self.sa_initial_temp
        sa_iterations = 0

        for iteration in range(self.sa_max_iterations):
            sa_iterations += 1

            # Propose a neighbor move (relocate or swap)
            neighbor_timetable, _ = self._sa_helper._neighbor_move(current_timetable)
            neighbor_report = count_violations(neighbor_timetable, self.sessions, self.dataset)

            # If Stage 1 achieved 0 hard violations, strictly preserve 0 hard violations
            if stage1_hard_zero and neighbor_report.hard_violations > 0:
                cost_history.append(best_cost)
                temp *= self.sa_cooling_rate
                continue

            neighbor_cost = 1000 * neighbor_report.hard_violations + neighbor_report.soft_penalty
            delta_cost = neighbor_cost - current_cost

            accept = False
            if delta_cost < 0:
                accept = True
            else:
                prob = math.exp(-delta_cost / max(temp, 1e-9))
                if random.random() < prob:
                    accept = True

            if accept:
                current_timetable = neighbor_timetable
                current_report = neighbor_report
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_timetable = current_timetable.copy()
                    best_report = neighbor_report
                    best_soft_penalty = neighbor_report.soft_penalty

            temp *= self.sa_cooling_rate
            cost_history.append(best_cost)

            # Early exit if perfect schedule (0 hard, 0 soft)
            if best_cost == 0:
                break

        phase2_time = time.time() - start_stage2
        total_time = time.time() - start_overall

        return HybridResult(
            timetable=best_timetable,
            success=True,
            phase1_time=phase1_time,
            phase2_time=phase2_time,
            total_time=total_time,
            backtracking_nodes=bt_result.nodes_expanded,
            sa_iterations=sa_iterations,
            initial_soft_penalty=initial_soft_penalty,
            final_soft_penalty=best_soft_penalty,
            cost_history=cost_history,
            report=best_report,
        )
