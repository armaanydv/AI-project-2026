"""
core/csp_model.py
=================
Defines the DATA MODEL for the AI Timetable Scheduler CSP.

Think of this file as the "schema" or "vocabulary" for the whole project.
Every other module (algorithms, UI, experiments) imports from here.

CSP Formulation Review
-----------------------
  Variable : one lecture/lab session that must be placed in the timetable.
             e.g. (ClassA, Maths, occurrence-1)
  Domain   : set of valid (day, period, room) combinations for that session.
  Constraints: rules the final assignment must satisfy (H1-H9, S1-S5).

Python note: we use @dataclass so we don't have to write __init__ by hand.
             'frozen=True' on SlotKey/Assignment makes them hashable so they
             can be used as dict keys and put in sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# PRIMITIVE / LEAF DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class TimeSlot:
    """
    Represents one cell in the timetable grid: (day, period).

    day    : integer 0..N-1  (0=Monday, 1=Tuesday, ...)
    period : integer 0..P-1  (0=first period, 1=second period, ...)
    """
    day: int
    period: int

    def __str__(self) -> str:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        d = day_names[self.day] if self.day < len(day_names) else f"D{self.day}"
        return f"{d}-P{self.period + 1}"


@dataclass(frozen=True)
class SlotKey:
    """
    An immutable (hashable) version of TimeSlot used as dict/set keys.
    frozen=True makes it usable as a dict key (needs __hash__).
    """
    day: int
    period: int


# ---------------------------------------------------------------------------
# INPUT ENTITIES  (what the user provides)
# ---------------------------------------------------------------------------

@dataclass
class Subject:
    """
    One taught subject.

    subject_id   : unique string id, e.g. "CS101"
    name         : human-readable name, e.g. "Data Structures"
    weekly_hours : how many periods per week this subject needs
    subject_type : "theory" or "lab"
                   -> theory sessions go in classrooms
                   -> lab sessions go in lab rooms (H5)
    lab_duration : if type=="lab", how many consecutive periods one lab
                   session occupies (usually 2 or 3). Ignored for theory.
    """
    subject_id: str
    name: str
    weekly_hours: int
    subject_type: str          # "theory" | "lab"
    lab_duration: int = 1      # for labs: periods per session block


@dataclass
class Teacher:
    """
    One faculty member.

    teacher_id       : unique string id, e.g. "T01"
    name             : full name
    subject_ids      : list of subject_ids this teacher can teach
    unavailable_slots: set of (day, period) tuples when teacher is NOT
                       available (H6 - teacher availability constraint)
    max_weekly_load  : max number of periods this teacher can teach in a week
                       (H8 - teacher load constraint)
    preferences      : dict of (day,period) -> penalty for soft constraint S4
                       e.g. {(0, 0): 5} means "teacher dislikes Mon period-1"
    """
    teacher_id: str
    name: str
    subject_ids: List[str]
    unavailable_slots: Set[Tuple[int, int]] = field(default_factory=set)
    max_weekly_load: int = 20
    preferences: Dict[Tuple[int, int], int] = field(default_factory=dict)


@dataclass
class Room:
    """
    One physical room.

    room_id   : unique string id, e.g. "R01"
    name      : e.g. "Lab 204"
    capacity  : max number of students (H4 - capacity constraint)
    room_type : "classroom" or "lab"  (H5 - type matching constraint)
    """
    room_id: str
    name: str
    capacity: int
    room_type: str    # "classroom" | "lab"


@dataclass
class Class:
    """
    One student batch/class.

    class_id           : unique string id, e.g. "CSE-A"
    name               : display name
    strength           : number of students (used for H4 capacity check)
    subject_ids        : list of subject_ids this class needs
    lunch_break_period : period index (0-based) that is the lunch break for
                         ALL days for this class. (H9)
    """
    class_id: str
    name: str
    strength: int
    subject_ids: List[str]
    lunch_break_period: int    # 0-based period index


@dataclass
class TimeStructure:
    """
    Defines the shape of the timetable grid.

    num_days        : how many working days (typically 5 for Mon-Fri)
    periods_per_day : how many periods in a day
    break_periods   : list of period indices that are non-teaching breaks
                      for ALL classes (e.g. a common assembly period)

    Example:
        TimeStructure(num_days=5, periods_per_day=6, break_periods=[])
        -> 5x6 grid = 30 total slots per week.
    """
    num_days: int
    periods_per_day: int
    break_periods: List[int] = field(default_factory=list)

    def all_slots(self) -> List[Tuple[int, int]]:
        """Return all (day, period) tuples that are NOT global break periods."""
        slots = []
        for d in range(self.num_days):
            for p in range(self.periods_per_day):
                if p not in self.break_periods:
                    slots.append((d, p))
        return slots

    def total_teaching_slots(self) -> int:
        """Total non-break slots in the entire week."""
        return len(self.all_slots())


# ---------------------------------------------------------------------------
# CSP VARIABLE
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """
    A CSP *Variable* - one session that must be placed in the timetable.

    session_id : unique id built as  "{class_id}__{subject_id}__{n:02d}"
                 e.g. "CSE-A__CS101__01" for the 1st DS lecture of CSE-A
    class_id   : which class this session belongs to
    subject_id : which subject
    teacher_id : which teacher is assigned (fixed in advance; this reduces
                 the domain size vs. treating teacher as a variable too)
    duration   : 1 for theory, 2+ for lab sessions (consecutive periods)

    WHY fixed teacher?
        The spec says "fix the teacher per subject in advance to keep the
        domain small". Without fixing the teacher, the domain would be
        (day x period x room x teacher) which is exponentially larger.
        By fixing it, domain = (day x period x room) only.
    """
    session_id: str
    class_id: str
    subject_id: str
    teacher_id: str
    duration: int = 1          # number of consecutive periods needed


# ---------------------------------------------------------------------------
# CSP VALUE (what gets assigned to a variable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Assignment:
    """
    A CSP *Value* - the (day, start_period, room_id) triple assigned to a
    Session variable.

    frozen=True so Assignment objects can be used as dict keys.

    day          : 0-based day index
    start_period : 0-based period index where the session starts
    room_id      : which room is used

    For a lab with duration=2: the session occupies (day, start_period) and
    (day, start_period+1). The constraint checkers handle this.
    """
    day: int
    start_period: int
    room_id: str

    def occupied_slots(self, duration: int) -> List[Tuple[int, int]]:
        """
        Return all (day, period) cells this assignment occupies.

        Example: Assignment(day=1, start_period=2, room_id="R1"), duration=2
                 -> [(1, 2), (1, 3)]
        """
        return [(self.day, self.start_period + offset)
                for offset in range(duration)]


# ---------------------------------------------------------------------------
# COMPLETE TIMETABLE (solution object)
# ---------------------------------------------------------------------------

@dataclass
class Timetable:
    """
    A complete (or partial) timetable.

    assignments : dict  session_id -> Assignment
                  If a session has no entry, it is unassigned (used during
                  backtracking where the solution is built incrementally).

    This is what both algorithms produce and what the UI displays.
    """
    assignments: Dict[str, Assignment] = field(default_factory=dict)

    def assign(self, session_id: str, value: Assignment) -> None:
        """Place a session into the timetable."""
        self.assignments[session_id] = value

    def unassign(self, session_id: str) -> None:
        """Remove a session from the timetable (used in backtracking)."""
        self.assignments.pop(session_id, None)

    def is_complete(self, sessions: List[Session]) -> bool:
        """True if every session has been assigned a value."""
        return all(s.session_id in self.assignments for s in sessions)

    def copy(self) -> "Timetable":
        """Shallow copy: new dict, same Assignment objects (they are frozen)."""
        return Timetable(assignments=dict(self.assignments))


# ---------------------------------------------------------------------------
# DATASET CONTAINER  (bundles everything into one problem instance)
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """
    All input data for one timetabling problem instance.

    This is loaded from a JSON file and passed to the algorithms.

    subjects           : dict  subject_id  -> Subject
    teachers           : dict  teacher_id  -> Teacher
    rooms              : dict  room_id     -> Room
    classes            : dict  class_id    -> Class
    time_structure     : the grid shape
    subject_teacher_map: subject_teacher_map[class_id][subject_id] = teacher_id
                         Built once during loading (the "fixed teacher" mapping).
    """
    subjects: Dict[str, Subject]
    teachers: Dict[str, Teacher]
    rooms: Dict[str, Room]
    classes: Dict[str, Class]
    time_structure: TimeStructure
    subject_teacher_map: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def get_sessions(self) -> List[Session]:
        """
        Expand the Dataset into a flat list of Session objects (CSP variables).

        For each class, for each subject of that class, we create
        (weekly_hours / lab_duration) Session objects.

        Example:
            Class CSE-A needs Maths (4 hrs/week, theory, duration=1)
            -> 4 Session objects: CSE-A__MATH__01 ... CSE-A__MATH__04

            Class CSE-A needs Networks Lab (2 hrs/week, lab, duration=2)
            -> 1 Session object: CSE-A__NLAB__01  (occupies 2 consecutive periods)
        """
        sessions: List[Session] = []

        for class_id, cls in self.classes.items():
            for subj_id in cls.subject_ids:
                subject = self.subjects[subj_id]
                duration = subject.lab_duration if subject.subject_type == "lab" else 1
                # number of sessions = total weekly hours / periods per session
                num_sessions = max(1, subject.weekly_hours // duration)

                # Look up the teacher for this class+subject combination
                teacher_id = self.subject_teacher_map.get(class_id, {}).get(subj_id, "")

                for i in range(1, num_sessions + 1):
                    sid = f"{class_id}__{subj_id}__{i:02d}"
                    sessions.append(Session(
                        session_id=sid,
                        class_id=class_id,
                        subject_id=subj_id,
                        teacher_id=teacher_id,
                        duration=duration,
                    ))

        return sessions
