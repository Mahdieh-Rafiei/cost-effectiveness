import sqlite3
from typing import Dict, Any, List
from pathlib import Path
import json
import re

# Use ONE DB path only (same file for read + write)
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ce_studies.sqlite3"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ce_studies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id TEXT UNIQUE,
  figure_group TEXT,
  condition TEXT,
  time_horizon TEXT,
  perspective TEXT,
  outcome_type TEXT,
  comparator_type TEXT,
  quadrant TEXT,
  notes TEXT,
  evidence_pages TEXT
);
"""

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = get_conn()
    conn.execute(SCHEMA_SQL)
    conn.commit()
    conn.close()

def _norm_simple(x: str) -> str:
    x = (x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x

def _to_text(v):
    if v is None:
        return "unknown"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def normalize_quadrant(v: str) -> str:
    x = _norm_simple(v).lower()

    # strict mappings
    if x in {"dominant", "dominance"}:
        return "dominant"
    if x in {"dominated"}:
        return "dominated"
    if x in {"ne", "north-east", "northeast"}:
        return "NE"
    if x in {"sw", "south-west", "southwest"}:
        return "SW"

    # soft mappings
    if "not cost-effective" in x:
        return "dominated"  # conservative approximation
    if "no significant difference" in x:
        return "unclear"
    if x == "effectiveness":
        return "unclear"
    if x in {"unknown", ""}:
        return "unknown"

    return "unclear"

def normalize_perspective(v: str) -> str:
    x = _norm_simple(v).lower()

    if x == "societal" or "societal" in x:
        return "societal"

    if x in {"healthcare", "health care"}:
        return "healthcare"
    if "healthcare" in x or "health care" in x:
        if "patient" in x or "individual" in x or "societal" in x:
            return "mixed"
        return "healthcare"

    if x == "patient" or "patient" in x or "individual" in x:
        return "patient"

    if x in {"unknown", ""}:
        return "unknown"

    return "unknown"

def upsert_row(row: Dict[str, Any]):
    conn = get_conn()
    conn.execute(SCHEMA_SQL)

    conn.execute("""
    INSERT INTO ce_studies
    (paper_id, figure_group, condition, time_horizon, perspective, outcome_type, comparator_type, quadrant, notes, evidence_pages)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(paper_id) DO UPDATE SET
      figure_group=excluded.figure_group,
      condition=excluded.condition,
      time_horizon=excluded.time_horizon,
      perspective=excluded.perspective,
      outcome_type=excluded.outcome_type,
      comparator_type=excluded.comparator_type,
      quadrant=excluded.quadrant,
      notes=excluded.notes,
      evidence_pages=excluded.evidence_pages
    """, (
        _to_text(row.get("paper_id")),
        _to_text(row.get("figure_group")),
        _to_text(row.get("condition")),
        _to_text(row.get("time_horizon")),
        normalize_perspective(_to_text(row.get("perspective"))),
        _to_text(row.get("outcome_type")),
        _to_text(row.get("comparator_type")),
        normalize_quadrant(_to_text(row.get("quadrant"))),
        _to_text(row.get("notes")),
        _to_text(row.get("evidence_pages")),
    ))

    conn.commit()
    conn.close()

def query_sql(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
