"""KingbaseES database adapter (PostgreSQL-compatible mode).

Async runtime always uses ``postgresql+asyncpg`` (PostgreSQL wire protocol).

Alembic / sync URLs are controlled by ``KINGBASE_SYNC_DRIVER``:

- ``psycopg2`` (default): ``postgresql+psycopg2://...`` — no vendor wheel required.
- ``ksycopg2``: ``kingbase+ksycopg2://...`` — ``ksycopg2`` is a default dependency on
  Linux/Windows (``uv sync``); omitted on macOS (no PyPI wheel).
"""

import os

from memory.database.repository.middleware.adapters.postgresql_adapter import (
    PostgreSQLAdapter,
)

# Alembic and other sync SQLAlchemy entrypoints only.
_SYNC_DRIVER_PSYCPG2 = "psycopg2"
_SYNC_DRIVER_KSYCOPG2 = "ksycopg2"


class KingbaseAdapter(PostgreSQLAdapter):
    """KingbaseES adapter: configurable sync driver; async stays on asyncpg."""

    def get_db_type(self) -> str:
        return "kingbase"

    def get_env_prefix(self) -> str:
        return "KINGBASE"

    def get_default_port(self) -> int:
        return 54321

    def build_sync_url(
        self, user: str, password: str, host: str, port: int, database: str
    ) -> str:
        driver = os.getenv("KINGBASE_SYNC_DRIVER", _SYNC_DRIVER_PSYCPG2).lower().strip()
        if driver == _SYNC_DRIVER_KSYCOPG2:
            return f"kingbase+ksycopg2://{user}:{password}@{host}:{port}/{database}"
        if driver in (_SYNC_DRIVER_PSYCPG2, "postgresql", "pg"):
            return super().build_sync_url(user, password, host, port, database)
        raise ValueError(
            f"Unsupported KINGBASE_SYNC_DRIVER={driver!r}. "
            f"Use {_SYNC_DRIVER_PSYCPG2!r} or {_SYNC_DRIVER_KSYCOPG2!r}."
        )
