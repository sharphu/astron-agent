import os
from typing import Optional

from plugin.link.consts import const
from plugin.link.domain.entity.tool_schema import Tools
from plugin.link.domain.models.utils import DatabaseService, RedisService

data_base_singleton: Optional[DatabaseService] = None
redis_singleton: Optional[RedisService] = None
PG_FAMILY = {"kingbase", "postgresql", "postgres", "pg"}


def init_data_base() -> None:
    """
    Initialize the database.
    """
    # Use global statement to modify module-level singleton instance
    global data_base_singleton
    db_type = (os.getenv(const.DB_TYPE_KEY, "mysql") or "mysql").lower().strip()
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
                "Missing required KingBase environment variables: "
                + ", ".join(missing)
            )
        sync_driver = os.getenv("KINGBASE_SYNC_DRIVER", "psycopg2").lower().strip()
        if sync_driver == "ksycopg2":
            db_url = f"kingbase+ksycopg2://{user}:{password}@{host}:{port}/{db}"
        else:
            db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    else:
        mysql_host = os.getenv(const.MYSQL_HOST_KEY)
        mysql_port = os.getenv(const.MYSQL_PORT_KEY)
        user = os.getenv(const.MYSQL_USER_KEY)
        password = os.getenv(const.MYSQL_PASSWORD_KEY)
        db = os.getenv(const.MYSQL_DB_KEY)
        missing = [
            key
            for key, value in [
                (const.MYSQL_HOST_KEY, mysql_host),
                (const.MYSQL_PORT_KEY, mysql_port),
                (const.MYSQL_USER_KEY, user),
                (const.MYSQL_PASSWORD_KEY, password),
                (const.MYSQL_DB_KEY, db),
            ]
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required MySQL environment variables: "
                + ", ".join(missing)
            )
        db_url = (
            f"mysql+pymysql://{user}:{password}@{mysql_host}:{mysql_port}/{db}"
            "?charset=utf8mb4"
        )
    data_base_singleton = DatabaseService(database_url=db_url)
    data_base_singleton.create_db_and_tables()

    # Initialize Redis service using global singleton pattern
    # Use global statement to modify module-level singleton instance
    global redis_singleton
    if not (
        addr := os.getenv(const.REDIS_CLUSTER_ADDR_KEY)
        or os.getenv(const.REDIS_ADDR_KEY)
    ):
        raise ValueError("Redis address is not set in environment variables")

    password = os.getenv(const.REDIS_PASSWORD_KEY)
    redis_singleton = RedisService(cluster_addr=addr, password=password)


def get_db_engine() -> Optional[DatabaseService]:
    """
    Get the global database service singleton instance.

    Returns:
        DatabaseService: The initialized database service instance
    """
    return data_base_singleton


def get_redis_engine() -> Optional[RedisService]:
    """
    Get the global Redis service singleton instance.

    Returns:
        RedisService: The initialized Redis service instance
    """
    return redis_singleton


if __name__ == "__main__":
    os.environ[const.MYSQL_HOST_KEY] = "mysql.mysql-hf04-2oc97b.svc.hfb.ipaas.cn"
    os.environ[const.MYSQL_PORT_KEY] = "8066"
    os.environ[const.MYSQL_USER_KEY] = "admin"
    os.environ[const.MYSQL_PASSWORD_KEY] = "EdgeAIGo!"
    os.environ[const.MYSQL_DB_KEY] = "spark_link"

    os.environ[const.REDIS_CLUSTER_ADDR_KEY] = (
        "172.29.100.22:7301,172.29.100.23:7301,172.29.100.24:7301,"
        "172.29.100.22:7302,172.29.100.23:7302,172.29.100.24:7302"
    )
    os.environ[const.REDIS_PASSWORD_KEY] = "0EHYkZSsk1NoQQGH"

    init_data_base()
    add_test1 = Tools(
        app_id="123231",
        tool_id="tool@1q331",
        name="航班查询",
        description="查询航班信息",
        open_api_schema="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    )

    if redis_engine := get_redis_engine():
        res = redis_engine.get("spark_bot:bot_config:0059649e52bb4c97a9f32a4d4bfcceea")
        print(res)
