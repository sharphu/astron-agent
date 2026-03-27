import os
from typing import Literal

from loguru import logger
from sqlalchemy import engine_from_config, pool
from sqlalchemy.sql.schema import SchemaItem
from sqlmodel import SQLModel

from alembic import context  # type: ignore[attr-defined]

# Import all models for SQLModel metadata registration
from workflow.configs import workflow_config  # noqa: F401
from workflow.domain.models.ai_app import App  # noqa: F401
from workflow.domain.models.app_source import AppSource  # noqa: F401
from workflow.domain.models.flow import Flow  # noqa: F401
from workflow.domain.models.history import History  # noqa: F401
from workflow.domain.models.license import License  # noqa: F401

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
PG_FAMILY = {"kingbase", "postgresql", "postgres", "pg"}


def get_database_url() -> str:
    db_type = os.getenv("DB_TYPE", "mysql").lower().strip()
    if db_type in PG_FAMILY:
        host = os.getenv("KINGBASE_HOST")
        port = os.getenv("KINGBASE_PORT")
        user = os.getenv("KINGBASE_USER")
        password = os.getenv("KINGBASE_PASSWORD")
        db = os.getenv("KINGBASE_DB")
        missing = [
            key
            for key, value in [
                ("KINGBASE_HOST", host),
                ("KINGBASE_PORT", port),
                ("KINGBASE_USER", user),
                ("KINGBASE_PASSWORD", password),
                ("KINGBASE_DB", db),
            ]
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required environment variables for Alembic: "
                + ", ".join(missing)
            )
        sync_driver = os.getenv("KINGBASE_SYNC_DRIVER", "psycopg2").lower().strip()
        if sync_driver == "ksycopg2":
            return f"kingbase+ksycopg2://{user}:{password}@{host}:{port}/{db}"
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    host = os.getenv("MYSQL_HOST")
    port = os.getenv("MYSQL_PORT")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    db = os.getenv("MYSQL_DB")
    missing = [
        key
        for key, value in [
            ("MYSQL_HOST", host),
            ("MYSQL_PORT", port),
            ("MYSQL_USER", user),
            ("MYSQL_PASSWORD", password),
            ("MYSQL_DB", db),
        ]
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing required environment variables for Alembic: "
            + ", ".join(missing)
        )
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"


config.set_main_option("sqlalchemy.url", get_database_url())


def get_metadata():  # type: ignore[no-untyped-def]
    return SQLModel.metadata


def include_object(
    object: SchemaItem,
    name: str | None,
    type_: Literal[
        "schema",
        "table",
        "column",
        "index",
        "unique_constraint",
        "foreign_key_constraint",
    ],
    reflected: bool,
    compare_to: SchemaItem | None,
) -> bool:
    """
    Determine whether to include a schema object in migration.

    :param object: The schema object
    :param name: The name of the object
    :param type_: The type of schema object
    :param reflected: Whether the object was reflected from the database
    :param compare_to: The object to compare to (if any)
    :return: True if the object should be included, False otherwise
    """
    if type_ == "foreign_key_constraint":
        return False
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=get_metadata(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """

    # this callback is used to prevent an auto-migration from being generated
    # when there are no changes to the schema
    # reference: http://alembic.zzzcomputing.com/en/latest/cookbook.html
    def process_revision_directives(context: object, revision: object, directives: list) -> None:  # type: ignore[no-untyped-def]
        if getattr(config.cmd_opts, "autogenerate", False):
            script = directives[0]
            if script.upgrade_ops.is_empty():
                directives[:] = []
                logger.info("No changes in schema detected.")

    configuration = config.get_section(config.config_ini_section) or {}

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=get_metadata(),
            process_revision_directives=process_revision_directives,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
