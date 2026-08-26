"""Verify configured Text2SQL database identities are least-privilege readers."""

from __future__ import annotations

import re
import sys

from sqlalchemy import text

from app.db import get_engine
from app.schema import set_active_profile


WRITE_PRIVILEGES = re.compile(r"\b(INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRIGGER|EXECUTE|REFERENCES|INDEX)\b", re.I)


def check(profile: str) -> bool:
    set_active_profile(profile)
    with get_engine(profile).connect() as connection:
        identity = dict(connection.execute(text(
            "SELECT CURRENT_USER() AS db_user, DATABASE() AS db_name"
        )).mappings().one())
        grants = [str(row[0]) for row in connection.execute(text("SHOW GRANTS")).all()]
    unsafe = [grant for grant in grants if WRITE_PRIVILEGES.search(grant)]
    print({"profile": profile, **identity, "grants": grants, "read_only": not unsafe})
    return not unsafe


def main() -> int:
    profiles = sys.argv[1:] or ["resin", "steel_industry"]
    return 0 if all(check(profile) for profile in profiles) else 1


if __name__ == "__main__":
    raise SystemExit(main())
