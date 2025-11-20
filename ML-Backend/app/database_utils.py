# app/database_utils.py

import sqlite3
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "syllabus"
DB_PATH = DATA_DIR / "db.sqlite3"

def list_subjects():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT id, name, created_at FROM syllabus ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

def get_subject_by_name(subject_name: str):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT id, name, text, created_at, outline_json FROM syllabus WHERE LOWER(name)=LOWER(?)",
            (subject_name.strip(),)
        ).fetchone()
        return dict(row) if row else None
