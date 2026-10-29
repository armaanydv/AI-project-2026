"""
core/data_loader.py
===================
Loads and validates timetable datasets from JSON files or dictionaries.

Converts raw JSON data into the strong typed dataclasses defined in
core.csp_model (Dataset, Subject, Teacher, Room, Class, TimeStructure).
Includes sanity and solvability checks so invalid datasets are caught early.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from core.csp_model import (
    Class,
    Dataset,
    Room,
    Subject,
    Teacher,
    TimeStructure,
)


class DataValidationError(Exception):
    """Raised when the input dataset fails structural or consistency validation."""
    pass


def load_dataset_from_dict(data: Dict[str, Any]) -> Dataset:
    """
    Parse a dictionary representation of a dataset and construct a Dataset instance.
    Performs validation on references, constraints, and total capacity.
    """
    required_keys = ["time_structure", "rooms", "subjects", "teachers", "classes"]
    for key in required_keys:
        if key not in data:
            raise DataValidationError(f"Missing required top-level key: '{key}'")

    # 1. Time Structure
    ts_raw = data["time_structure"]
    time_structure = TimeStructure(
        num_days=int(ts_raw.get("num_days", 5)),
        periods_per_day=int(ts_raw.get("periods_per_day", 6)),
        break_periods=list(ts_raw.get("break_periods", [])),
    )

    # 2. Rooms
    rooms: Dict[str, Room] = {}
    for r in data["rooms"]:
        rid = str(r["room_id"])
        rooms[rid] = Room(
            room_id=rid,
            name=str(r.get("name", rid)),
            capacity=int(r.get("capacity", 40)),
            room_type=str(r.get("room_type", "classroom")).lower(),
        )

    # 3. Subjects
    subjects: Dict[str, Subject] = {}
    for s in data["subjects"]:
        sid = str(s["subject_id"])
        stype = str(s.get("subject_type", "theory")).lower()
        duration = int(s.get("lab_duration", 2 if stype == "lab" else 1))
        subjects[sid] = Subject(
            subject_id=sid,
            name=str(s.get("name", sid)),
            weekly_hours=int(s.get("weekly_hours", 3)),
            subject_type=stype,
            lab_duration=duration,
        )

    # 4. Teachers
    teachers: Dict[str, Teacher] = {}
    for t in data["teachers"]:
        tid = str(t["teacher_id"])
        unavail_raw = t.get("unavailable_slots", [])
        unavail_set: Set[Tuple[int, int]] = set()
        for slot in unavail_raw:
            if isinstance(slot, (list, tuple)) and len(slot) == 2:
                unavail_set.add((int(slot[0]), int(slot[1])))

        pref_raw = t.get("preferences", {})
        pref_dict: Dict[Tuple[int, int], int] = {}
        for k, v in pref_raw.items():
            if isinstance(k, str) and "," in k:
                parts = k.split(",")
                pref_dict[(int(parts[0].strip()), int(parts[1].strip()))] = int(v)
            elif isinstance(k, (list, tuple)) and len(k) == 2:
                pref_dict[(int(k[0]), int(k[1]))] = int(v)

        teachers[tid] = Teacher(
            teacher_id=tid,
            name=str(t.get("name", tid)),
            subject_ids=[str(sub) for sub in t.get("subject_ids", [])],
            unavailable_slots=unavail_set,
            max_weekly_load=int(t.get("max_weekly_load", 20)),
            preferences=pref_dict,
        )

    # 5. Classes
    classes: Dict[str, Class] = {}
    for c in data["classes"]:
        cid = str(c["class_id"])
        classes[cid] = Class(
            class_id=cid,
            name=str(c.get("name", cid)),
            strength=int(c.get("strength", 30)),
            subject_ids=[str(sub) for sub in c.get("subject_ids", [])],
            lunch_break_period=int(c.get("lunch_break_period", 3)),
        )

    # 6. Subject-Teacher Map
    # Either explicit in JSON, or auto-resolved from teacher.subject_ids
    st_map: Dict[str, Dict[str, str]] = {}
    explicit_map = data.get("subject_teacher_map", {})

    for cid, cls in classes.items():
        st_map[cid] = {}
        for sub_id in cls.subject_ids:
            if sub_id not in subjects:
                raise DataValidationError(f"Class '{cid}' references unknown subject '{sub_id}'")

            teacher_for_sub = ""
            if cid in explicit_map and sub_id in explicit_map[cid]:
                teacher_for_sub = explicit_map[cid][sub_id]
            else:
                # Fallback: find first teacher qualified for this subject
                for tid, teacher in teachers.items():
                    if sub_id in teacher.subject_ids:
                        teacher_for_sub = tid
                        break

            if not teacher_for_sub or teacher_for_sub not in teachers:
                raise DataValidationError(
                    f"No valid teacher assigned or qualified for subject '{sub_id}' in class '{cid}'"
                )
            st_map[cid][sub_id] = teacher_for_sub

    # 7. Basic Solvability & Feasibility Checks
    total_class_hours = sum(
        subjects[sid].weekly_hours for cls in classes.values() for sid in cls.subject_ids
    )
    total_room_slots = len(rooms) * time_structure.total_teaching_slots()
    if total_class_hours > total_room_slots:
        raise DataValidationError(
            f"Dataset unsolvable: Total required class hours ({total_class_hours}) "
            f"exceed total available room slots ({total_room_slots})"
        )

    # Check each class total hours vs available class slots
    for cid, cls in classes.items():
        cls_hours = sum(subjects[sid].weekly_hours for sid in cls.subject_ids)
        # Class cannot be scheduled during its lunch break or global breaks
        available_days = time_structure.num_days
        available_slots_per_day = time_structure.periods_per_day - len(time_structure.break_periods)
        if 0 <= cls.lunch_break_period < time_structure.periods_per_day:
            available_slots_per_day -= 1
        max_cls_slots = available_days * available_slots_per_day
        if cls_hours > max_cls_slots:
            raise DataValidationError(
                f"Class '{cid}' requires {cls_hours} hours, but has only {max_cls_slots} "
                f"usable slots in the week (considering lunch and breaks)"
            )

    return Dataset(
        subjects=subjects,
        teachers=teachers,
        rooms=rooms,
        classes=classes,
        time_structure=time_structure,
        subject_teacher_map=st_map,
    )


def load_dataset_from_file(file_path: str | Path) -> Dataset:
    """Read a JSON file from disk, parse it, and return a validated Dataset."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataValidationError(f"Invalid JSON in file '{path.name}': {e}") from e

    return load_dataset_from_dict(raw_data)
