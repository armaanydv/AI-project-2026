# AI-Based Timetable Scheduler

An intelligent academic timetable generation system that uses Artificial Intelligence, Constraint Satisfaction Problems (CSP), and search-based optimization algorithms to generate valid and conflict-free academic schedules.

## 1. Project Overview

The AI-Based Timetable Scheduler automates academic timetable generation by assigning classes, teachers, subjects, classrooms, and time slots while satisfying predefined scheduling constraints.

The system reduces manual scheduling effort, prevents resource conflicts, and improves timetable quality through intelligent search and optimization techniques.

## 2. Problem Statement

Manual academic timetable scheduling is a complex and time-consuming process involving multiple constraints, including teacher availability, classroom allocation, subject requirements, and time-slot conflicts.

This project addresses these challenges by developing an AI-based system capable of generating and evaluating academic timetables automatically.

## 3. Key Features

- Automated academic timetable generation.
- Teacher and classroom allocation.
- Scheduling conflict detection.
- Constraint-based timetable validation.
- Multiple algorithm-based scheduling approaches.
- Hard and soft constraint handling.
- Algorithm performance comparison.
- Interactive timetable visualization.
- Manual timetable editing.
- Cost-based timetable evaluation.

## 4. Algorithms & AI Techniques

The project models academic scheduling as a **Constraint Satisfaction Problem (CSP)**.

### 4.1 Backtracking Search

A systematic search algorithm that assigns values to scheduling variables and backtracks whenever a constraint violation occurs.

### 4.2 Minimum Remaining Values (MRV)

Selects the variable with the fewest available values to reduce the search space.

### 4.3 Degree Heuristic

Prioritizes variables that constrain the largest number of other unassigned variables.

### 4.4 Forward Checking

Eliminates inconsistent future assignments after each variable assignment.

### 4.5 Min-Conflicts

A local search algorithm that iteratively modifies variable assignments to reduce constraint violations.

## 5. Project Structure

```text
AI ASSIGNMENT/
│
├── core/
│   ├── csp_model.py
│   ├── cost_function.py
│   ├── backtracking.py
│   ├── min_conflicts.py
│   └── data_loader.py
│
├── data/
│   ├── small_dataset.json
│   └── large_dataset.json
│
├── experiments/
│   ├── run_experiments.py
│   └── results/
│
├── app.py
├── requirements.txt
└── README.md
```

### Folder Description

| File / Folder | Description |
|---|---|
| `core/` | Contains CSP models and scheduling algorithms. |
| `csp_model.py` | Defines scheduling data models. |
| `cost_function.py` | Calculates hard violations and soft penalties. |
| `backtracking.py` | Implements the Backtracking CSP solver with heuristics. |
| `min_conflicts.py` | Implements the Min-Conflicts local search solver. |
| `data_loader.py` | Loads and validates JSON datasets. |
| `data/` | Contains sample scheduling datasets. |
| `experiments/` | Contains algorithm comparison scripts. |
| `results/` | Stores generated CSV files and result plots. |
| `app.py` | Streamlit application interface. |
| `requirements.txt` | Lists required Python dependencies. |

## 6. Scheduling Constraints

The scheduler uses two categories of constraints to generate valid and efficient academic timetables.

### 6.1 Hard Constraints (H1–H9)

Hard constraints are mandatory rules that must be satisfied for a timetable to be considered valid.

| ID | Constraint | Description |
|---|---|---|
| H1 | Teacher Conflict | A teacher cannot teach two sessions simultaneously. |
| H2 | Class Conflict | A class cannot have two sessions simultaneously. |
| H3 | Room Conflict | A room cannot host two sessions simultaneously. |
| H4 | Room Capacity | Room capacity must be greater than or equal to class strength. |
| H5 | Room Suitability | Labs must be scheduled in lab rooms and theory sessions in classrooms. |
| H6 | Teacher Availability | Teachers can only be scheduled in their available time slots. |
| H7 | Weekly Subject Hours | Every subject must receive exactly its required weekly hours. |
| H8 | Teacher Workload | Teacher workload must not exceed the maximum weekly load. |
| H9 | Lunch Break | The class lunch-break slot must remain empty. |

### 6.2 Soft Constraints (S1–S5)

Soft constraints represent scheduling preferences that improve timetable quality.

| ID | Constraint | Description |
|---|---|---|
| S1 | Minimize Gaps | Minimize gaps in a class's daily schedule. |
| S2 | Subject Distribution | Spread subject lectures across different days. |
| S3 | Consecutive Labs | Lab sessions should occupy consecutive periods. |
| S4 | Teacher Preferences | Respect teacher time preferences. |
| S5 | Consecutive Lectures | Avoid more than two consecutive lectures for a teacher. |

## 7. Cost Function

The project uses a cost function to evaluate timetable quality by combining hard constraint violations and soft constraint penalties.

### Cost Formula

```text
Cost = 1000 × (Hard Violations) + (Soft Penalty Score)
```

### Explanation

- **Hard Violations:** Number of violated mandatory scheduling constraints.
- **Soft Penalty Score:** Total penalty resulting from violated scheduling preferences.
- **1000:** A large penalty weight that prioritizes satisfying hard constraints.

A lower cost indicates a better timetable.

The same cost function is used by both scheduling algorithms and the manual-edit feature through:

`core/cost_function.py`

## 8. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| Pandas | Data processing |
| Matplotlib | Data visualization |
| OpenPyXL | Excel file handling |
| Pytest | Automated testing |
| JSON | Dataset storage |

## 9. Installation & Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/armaanydv/AI-project-2026.git
```

### Step 2: Navigate to the Project Directory

```bash
cd AI-project-2026
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

For Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

## 10. Team Members

| S. No. | Name | GitHub Username |
|---|---|---|
| 1 | Animesh Agrawal | @animeshagrawal2402 |
| 2 | Samiksha Devi | @samikshadevi11 |
| 3 | Armaan Yadav | @armaanydv |


## Academic Project

**Project Title:** AI-Based Timetable Scheduler

**Repository:** AI-project-2026

**Academic Year:** 2026

Developed as a collaborative academic project demonstrating the practical application of Artificial Intelligence, Constraint Satisfaction Problems, search algorithms, and optimization techniques.

---

*End of Project Documentation*
