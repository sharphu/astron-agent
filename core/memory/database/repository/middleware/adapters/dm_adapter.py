"""Dameng DM database adapter implementation.

This adapter integrates DM with the project's SQLAlchemy + SQLGlot workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from memory.database.repository.middleware.adapters.base import DatabaseAdapter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import quoted_name


class DMAdapter(DatabaseAdapter):
    """Dameng DM-specific database adapter implementation.

    Notes:
    - The service creates and uses per-user schemas (modes) via `SET SCHEMA ...`.
    - The admin schema name is hardcoded to `sparkdb_manager` in this service.
    """

    def get_db_type(self) -> str:
        return "dm"

    def get_sqlglot_dialect(self) -> str:
        # DM8 native DDL uses IDENTITY(seed, increment) for auto-increment columns.
        # SQLGlot's T-SQL dialect provides robust parsing for IDENTITY syntax.
        return "tsql"

    def build_async_url(
        self, user: str, password: str, host: str, port: int, database: str
    ) -> str:
        # Use dmSQLAlchemy's async driver.
        return (
            f"dm+dmAsync://{user}:{password}@{host}:{port}/{database}"
            if database
            else f"dm+dmAsync://{user}:{password}@{host}:{port}/"
        )

    def build_sync_url(
        self, user: str, password: str, host: str, port: int, database: str
    ) -> str:
        # Alembic migrations use synchronous SQLAlchemy engine.
        return (
            f"dm+dmPython://{user}:{password}@{host}:{port}/{database}"
            if database
            else f"dm+dmPython://{user}:{password}@{host}:{port}/"
        )

    def get_engine_connect_args(self) -> dict:
        # Use dmSQLAlchemy default connection arguments
        return {}

    async def create_database_if_not_exists(self, base_url: str, db_name: str) -> None:
        # DM连接串在我们部署中要求必须带库名（/db_name），因此这里不做额外建库。
        # base_url/db_name 仍保留接口一致性。
        _ = base_url
        _ = db_name
        return

    async def create_admin_schema(self, database_url: str) -> None:
        # Ensure sparkdb_manager schema exists.
        schema_name = "sparkdb_manager"
        engine = create_async_engine(
            database_url,
            echo=False,
            isolation_level="AUTOCOMMIT",
            connect_args=self.get_engine_connect_args(),
        )
        try:
            async with engine.connect() as conn:
                # DM grammar for CREATE SCHEMA does not show IF NOT EXISTS in all contexts;
                # so we do best-effort create and ignore duplicate errors.
                await conn.execute(text(f'CREATE SCHEMA "{schema_name}"'))
                logger.info('DM admin schema "%s" created', schema_name)
        except Exception as e:  # pylint: disable=broad-except
            # Duplicate schema / already exists can be safely ignored.
            logger.info('DM admin schema "%s" ensure: %s', schema_name, e)
        finally:
            await engine.dispose()

    def safe_create_schema_sql(self, schema_name: str) -> Any:
        safe_name = quoted_name(schema_name, quote=True)
        # For DM8, CREATE SCHEMA supports creating objects under the current user.
        # We avoid AUTHORIZATION to keep privilege handling simple for "fixed ordinary user".
        return text(f'CREATE SCHEMA "{safe_name}"')

    def safe_drop_schema_sql(self, schema_name: str) -> Any:
        safe_name = quoted_name(schema_name, quote=True)
        # DM8 syntax: DROP SCHEMA schema_name [RESTRICT | CASCADE]
        return text(f'DROP SCHEMA "{safe_name}" CASCADE')

    def list_tables_sql(self) -> str:
        # List user tables in current schema (mode).
        # Query sys dictionary tables:
        # - SYSOBJECTS: object metadata
        # - SYSCOLUMNS: column metadata
        #
        # For current schema, use CURRENT_SCHID.
        return (
            "SELECT NAME "
            "FROM SYSOBJECTS "
            "WHERE TYPE$ = 'TABOBJ' "
            "AND SUBTYPE$ = 'UTAB' "
            "AND SCHID = CURRENT_SCHID "
            # The service layer passes :schema for PG/MySQL compatibility.
            # DM relies on `SET SCHEMA ...` so we don't need it for filtering,
            # but we must reference it to avoid SQLAlchemy bind-parameter errors.
            "AND (:schema IS NULL OR 1=1)"
        )

    def get_column_types_sql(self) -> str:
        # Return (column_name, data_type, specific_type).
        # We use SYSCOLUMNS.TYPE$ as both to keep the pipeline simple.
        return (
            "SELECT LOWER(C.NAME), C.TYPE$, C.TYPE$ "
            "FROM SYSCOLUMNS C "
            "WHERE C.ID = ("
            "  SELECT O.ID "
            "  FROM SYSOBJECTS O "
            "  WHERE UPPER(O.NAME) = UPPER(:table_name) "
            "    AND O.TYPE$ = 'TABOBJ' "
            "    AND O.SUBTYPE$ = 'UTAB' "
            "    AND O.SCHID = CURRENT_SCHID "
            # The service layer passes :table_schema for PG/MySQL compatibility.
            # DM uses current schema via `SET SCHEMA`, so it is not required for
            # filtering, but it must be referenced to keep bind params consistent.
            "    AND (:table_schema IS NULL OR 1=1)"
            ") "
            "ORDER BY C.COLID"
        )

    def get_reserved_keywords(self) -> List[str]:
        # Start conservative by reusing PostgreSQL reserved keywords list.
        # The goal is to prevent unsafe identifiers; false positives are acceptable.
        return [
            "all",
            "analyse",
            "analyze",
            "and",
            "any",
            "array",
            "as",
            "asc",
            "asymmetric",
            "authorization",
            "binary",
            "both",
            "case",
            "cast",
            "check",
            "collate",
            "collation",
            "column",
            "concurrently",
            "constraint",
            "create",
            "cross",
            "current_catalog",
            "current_date",
            "current_role",
            "current_schema",
            "current_time",
            "current_timestamp",
            "current_user",
            "default",
            "deferrable",
            "desc",
            "distinct",
            "do",
            "else",
            "end",
            "except",
            "false",
            "fetch",
            "for",
            "foreign",
            "freeze",
            "from",
            "full",
            "grant",
            "group",
            "having",
            "ilike",
            "in",
            "initially",
            "inner",
            "intersect",
            "into",
            "is",
            "isnull",
            "join",
            "lateral",
            "leading",
            "left",
            "like",
            "limit",
            "localtime",
            "localtimestamp",
            "natural",
            "not",
            "notnull",
            "null",
            "offset",
            "on",
            "only",
            "or",
            "order",
            "outer",
            "overlaps",
            "placing",
            "primary",
            "references",
            "returning",
            "right",
            "select",
            "session_user",
            "similar",
            "some",
            "symmetric",
            "table",
            "tablesample",
            "then",
            "to",
            "trailing",
            "true",
            "union",
            "unique",
            "user",
            "using",
            "variadic",
            "verbose",
            "when",
            "where",
            "window",
            "with",
        ]

    def get_dangerous_functions(self) -> List[str]:
        # Start conservative by reusing PostgreSQL dangerous functions list.
        return [
            "current_catalog",
            "current_database",
            "current_role",
            "current_schema",
            "current_schemas",
            "current_user",
            "inet_client_addr",
            "inet_client_port",
            "inet_server_addr",
            "inet_server_port",
            "pg_backend_pid",
            "pg_blocking_pids",
            "pg_conf_load_time",
            "pg_current_logfile",
            "pg_my_temp_schema",
            "pg_is_other_temp_schema",
            "pg_listening_channels",
            "pg_postmaster_start_time",
            "pg_safe_snapshot_blocking_pids",
            "pg_try_advisory_lock",
            "pg_terminate_backend",
            "pg_export_snapshot",
            "pg_is_wal_replay_paused",
            "pg_get_wal_replay_pause_state",
            "pg_advisory_lock",
            "pg_try_advisory_lock",
            "current_database",
            "current_role",
            "current_schema",
            "current_user",
            "session_user",
            "user",
            "version",
        ]

    def is_retryable_cache_error(self, exception: Exception) -> bool:
        # DM-specific retryable cache invalidation not implemented yet.
        _ = exception
        return False

    async def clear_statement_cache(self, session: Any) -> None:
        # No-op for now.
        _ = session
        return

    async def restore_search_path(self, session: Any) -> None:
        # Restore based on session._current_schema if present.
        current_schema = getattr(session, "_current_schema", None)
        if not current_schema:
            return
        try:
            await session.execute(text(self.set_search_path_sql(current_schema)))
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Failed to restore DM schema: %s", e)

    def set_search_path_sql(self, schema_name: str) -> str:
        safe_name = quoted_name(schema_name, quote=True)
        # DM supports `SET SCHEMA <模式名>;` as per official docs.
        return f'SET SCHEMA "{safe_name}"'

    def get_alembic_version_table_schema(self) -> Optional[str]:
        return "sparkdb_manager"

    def get_alembic_include_schemas(self) -> bool:
        return True

    def get_model_table_args(self) -> Dict[str, Any]:
        return {"schema": "sparkdb_manager"}

    def get_env_prefix(self) -> str:
        return "DM"

    def get_default_port(self) -> int:
        return 5236
