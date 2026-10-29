"""
experiments/run_experiments.py
==============================
Comparative Empirical Evaluation of CSP & Search Algorithms on Timetabling Datasets.
Generates:
1. CSV summary of runtime, iterations, hard violations, soft penalties.
2. PNG comparison plots in experiments/results/.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from core.data_loader import load_dataset_from_file
from core.backtracking import BacktrackingSolver
from core.min_conflicts import MinConflictsSolver
from core.simulated_annealing import SimulatedAnnealingSolver
from core.hybrid import HybridSolver


def run_benchmark():
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "Small": "data/small_dataset.json",
        "Large": "data/large_dataset.json",
    }

    records = []

    for d_name, d_path in datasets.items():
        print(f"\n==========================================")
        print(f"  BENCHMARKING DATASET: {d_name.upper()} ({d_path})")
        print(f"==========================================")
        ds = load_dataset_from_file(d_path)

        # 1. Backtracking Baseline (with small timeout on large)
        print(f"Running: Backtracking CSP (Baseline)...")
        bt_timeout = 5.0 if d_name == "Large" else 15.0
        s_bt = BacktrackingSolver(ds, use_mrv=False, use_degree=False, use_forward_checking=False, timeout_seconds=bt_timeout, seed=42)
        res_bt = s_bt.solve()
        records.append({
            "Dataset": d_name,
            "Algorithm": "Backtracking (Baseline)",
            "Success": res_bt.success,
            "Time (s)": round(res_bt.time_taken, 4),
            "Hard Violations": res_bt.report.hard_violations,
            "Soft Penalty": res_bt.report.soft_penalty,
            "Total Cost": res_bt.report.total_cost,
        })

        # 2. Backtracking MRV + Degree + FC
        print(f"Running: Backtracking CSP (MRV + Degree + FC)...")
        s_mrv = BacktrackingSolver(ds, use_mrv=True, use_degree=True, use_forward_checking=True, timeout_seconds=30.0, seed=42)
        res_mrv = s_mrv.solve()
        records.append({
            "Dataset": d_name,
            "Algorithm": "Backtracking (MRV+Deg+FC)",
            "Success": res_mrv.success,
            "Time (s)": round(res_mrv.time_taken, 4),
            "Hard Violations": res_mrv.report.hard_violations,
            "Soft Penalty": res_mrv.report.soft_penalty,
            "Total Cost": res_mrv.report.total_cost,
        })

        # 3. Min-Conflicts
        print(f"Running: Min-Conflicts Local Search...")
        mc_iters = 1500 if d_name == "Small" else 3000
        s_mc = MinConflictsSolver(ds, max_iterations=mc_iters, random_walk_prob=0.06, seed=42)
        res_mc = s_mc.solve()
        records.append({
            "Dataset": d_name,
            "Algorithm": "Min-Conflicts Search",
            "Success": res_mc.success,
            "Time (s)": round(res_mc.time_taken, 4),
            "Hard Violations": res_mc.report.hard_violations,
            "Soft Penalty": res_mc.report.soft_penalty,
            "Total Cost": res_mc.report.total_cost,
        })

        # 4. Simulated Annealing
        print(f"Running: Simulated Annealing (Bonus 1)...")
        sa_iters = 1000 if d_name == "Small" else 2000
        s_sa = SimulatedAnnealingSolver(ds, max_iterations=sa_iters, seed=42)
        res_sa = s_sa.solve()
        records.append({
            "Dataset": d_name,
            "Algorithm": "Simulated Annealing",
            "Success": res_sa.success,
            "Time (s)": round(res_sa.time_taken, 4),
            "Hard Violations": res_sa.report.hard_violations,
            "Soft Penalty": res_sa.report.soft_penalty,
            "Total Cost": res_sa.report.total_cost,
        })

        # 5. Hybrid CSP + SA
        print(f"Running: Hybrid CSP + SA (Bonus 2)...")
        hy_iters = 600 if d_name == "Small" else 1200
        s_hy = HybridSolver(ds, sa_max_iterations=hy_iters, bt_timeout=15.0, seed=42)
        res_hy = s_hy.solve()
        records.append({
            "Dataset": d_name,
            "Algorithm": "Hybrid CSP + SA",
            "Success": res_hy.success,
            "Time (s)": round(res_hy.total_time, 4),
            "Hard Violations": res_hy.report.hard_violations,
            "Soft Penalty": res_hy.report.soft_penalty,
            "Total Cost": res_hy.report.total_cost,
        })

    df = pd.DataFrame(records)
    csv_path = results_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV benchmark results to: {csv_path}")
    print(df.to_string())

    # Generate Matplotlib chart if available
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_df = df.pivot(index="Algorithm", columns="Dataset", values="Time (s)")
        pivot_df.plot(kind="bar", ax=ax, colormap="viridis", width=0.7)
        ax.set_title("Runtime Comparison: AI Search Algorithms for Timetable CSP", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("Execution Time (seconds)", fontsize=11)
        ax.set_xlabel("Algorithm", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plot_path = results_dir / "comparison_plot.png"
        plt.savefig(plot_path, dpi=200)
        print(f"Saved comparison plot to: {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    run_benchmark()
