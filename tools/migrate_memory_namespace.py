"""One-way migration for memories written before tenant/user namespaces."""

from __future__ import annotations

import argparse
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/long_term_memory.sqlite3")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user", default="system")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    suffix = f":tenant:{args.tenant}:user:{args.user}"
    with sqlite3.connect(args.db) as db:
        rows = db.execute("SELECT DISTINCT namespace FROM long_term_memories WHERE namespace NOT LIKE '%:tenant:%'").fetchall()
        print(f"legacy namespaces: {len(rows)}")
        for (old,) in rows:
            new = old + suffix
            count = db.execute("SELECT COUNT(*) FROM long_term_memories WHERE namespace = ?", (old,)).fetchone()[0]
            print(f"{old} -> {new}: {count} records")
            if args.apply:
                db.execute("UPDATE OR IGNORE long_term_memories SET namespace = ? WHERE namespace = ?", (new, old))
                db.execute(
                    "UPDATE long_term_memories SET is_active = 0, updated_at = datetime('now') "
                    "WHERE namespace = ?",
                    (old,),
                )
        if args.apply:
            db.commit()
            print("migration applied")
        else:
            print("dry run; pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
