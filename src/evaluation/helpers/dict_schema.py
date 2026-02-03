from __future__ import annotations

import sqlite3
from typing import Dict, Set, List, Any


def extract_sqlite_schema(db_path: str) -> Dict[str, Any]:
    """
    Extract schema info from a SQLite database file.

    Returns on success:
      {
        "ok": True,
        "tables": {table_name -> {col1, col2, ...}},          # lowercased
        "primary_keys": {table_name -> {pk_col1, ...}},        # lowercased
        "foreign_keys": {table_name -> [ ... fk dicts ... ]},  # lowercased names
      }

    Returns on failure:
      {
        "ok": False,
        "error": "<exception type>: <message>",
        "db_path": "...",
      }
    """
    con = None
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row

        tables = [
            r["name"]
            for r in con.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]

        tables_map: Dict[str, Set[str]] = {}
        primary_keys: Dict[str, Set[str]] = {}
        foreign_keys: Dict[str, List[Dict[str, Any]]] = {}

        for t in tables:
            t_lc = t.lower()
            safe_t = t.replace("'", "''")

            # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
            ti_rows = con.execute(f"PRAGMA table_info('{safe_t}')").fetchall()
            tables_map[t_lc] = {row["name"].lower() for row in ti_rows}
            primary_keys[t_lc] = {row["name"].lower() for row in ti_rows if int(row["pk"]) > 0}

            # PRAGMA foreign_key_list:
            fk_rows = con.execute(f"PRAGMA foreign_key_list('{safe_t}')").fetchall()
            fk_list: List[Dict[str, Any]] = []
            for row in fk_rows:
                fk_list.append(
                    {
                        "id": int(row["id"]),
                        "seq": int(row["seq"]),
                        "ref_table": (row["table"] or "").lower(),
                        "from_column": (row["from"] or "").lower(),
                        "to_column": (row["to"] or "").lower(),
                        "on_update": row["on_update"],
                        "on_delete": row["on_delete"],
                        "match": row["match"],
                    }
                )
            foreign_keys[t_lc] = fk_list

        return {
            "ok": True,
            "tables": tables_map,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "db_path": db_path,
        }

    finally:
        if con is not None:
            con.close()
