# AI-Based Timetable Scheduler

**B.Tech 2nd Year AI Assignment**

An AI-powered college timetable generator using Constraint Satisfaction Problem (CSP) techniques.

---

## Project Structure

```
AI ASSIGNMT/
├── core/                    # Algorithm & CSP logic (no UI code here)
│   ├── csp_model.py         # Data model: Session, Assignment, Timetable, Dataset
│   ├── cost_function.py     # Shared violation counter (H1-H9 + S1-S5)
│   ├── backtracking.py      # Backtracking CSP solver (plain + heuristics)
│   ├── min_conflicts.py     # Min-Conflicts local search solver
│   └── data_loader.py       # Load/validate JSON datasets
│
├── data/                    # Sample problem instances (JSON)
│   ├── small_dataset.json   # 3 classes, 6 teachers, 4 rooms, 5x6 grid
│   └── large_dataset.json   # 8 classes, 15 teachers, 8 rooms, 5x7 grid
│
├── experiments/             # Comparison scripts + result plots
│   ├── run_experiments.py   # Run both algorithms on both datasets, save CSV
│   └── results/             # Generated CSVs and PNG plots go here
│
├── app.py                   # Streamlit UI (imports from core/ only)
├── requirements.txt         # pip dependencies
└── README.md                # This file
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py

# 4. Run experiments (generates CSV + plots)
python experiments/run_experiments.py
```

---

## Algorithms Implemented

### 1. Backtracking CSP (core/backtracking.py)
- **Baseline**: plain recursive backtracking
- **Optimised**: MRV (Minimum Remaining Values) + Degree heuristic + Forward Checking

### 2. Min-Conflicts Local Search (core/min_conflicts.py)
- Starts from a random complete assignment
- Iteratively picks conflicted sessions and reassigns them to minimize conflicts
- Includes random-walk plateau escape

---

## Hard Constraints (H1–H9)
| ID | Rule |
|----|------|
| H1 | Teacher cannot teach two sessions simultaneously |
| H2 | Class cannot have two sessions simultaneously |
| H3 | Room cannot host two sessions simultaneously |
| H4 | Room capacity ≥ class strength |
| H5 | Labs in lab rooms, theory in classrooms |
| H6 | Teacher only scheduled in available slots |
| H7 | Every subject gets exactly its required weekly hours |
| H8 | Teacher load ≤ max weekly load |
| H9 | Class lunch break slot stays empty |

## Soft Constraints (S1–S5)
| ID | Rule |
|----|------|
| S1 | Minimize gaps in a class''s daily schedule |
| S2 | Spread subject lectures across different days |
| S3 | Lab sessions occupy consecutive periods |
| S4 | Respect teacher time preferences |
| S5 | Avoid more than 2 consecutive lectures for a teacher |

---

## Cost Function
```
Cost = 1000 × (hard violations) + (soft penalty score)
```
Both algorithms and the manual-edit feature use the **same** cost function in `core/cost_function.py`.
