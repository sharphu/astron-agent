from datetime import datetime
from typing import Optional

from common.utils.snowfake import get_id
from sqlmodel import Field, SQLModel


class Bot(SQLModel, table=True):  # type: ignore[valid-type,misc]
    __tablename__ = "bot"

    id: str = Field(
        default_factory=get_id,
        primary_key=True,
        description="Primary key ID, snowflake ID",
    )
    app_id: str = Field(..., description="Tenant application identifier")
    dsl: str = Field(..., description="Assistant orchestration protocol")
    create_at: datetime = Field(
        default_factory=datetime.now, description="Creation time"
    )
    update_at: datetime = Field(default_factory=datetime.now, description="Update time")


class BotTenant(SQLModel, table=True):  # type: ignore[valid-type,misc]
    __tablename__ = "bot_tenant"

    id: str = Field(
        default_factory=get_id,
        primary_key=True,
        description="Primary key ID, snowflake ID",
    )
    name: str = Field(..., max_length=64, description="Application name")
    alias_id: str = Field(
        ..., max_length=32, unique=True, description="Application identifier ID"
    )
    description: Optional[str] = Field(
        default=None, max_length=255, description="Tenant description"
    )
    api_key: Optional[str] = Field(
        default=None, max_length=128, description="Tenant API key"
    )
    api_secret: Optional[str] = Field(
        default=None, max_length=128, description="Tenant API secret"
    )
    create_at: datetime = Field(
        default_factory=datetime.now, description="Creation time"
    )
    update_at: datetime = Field(default_factory=datetime.now, description="Update time")


class BotRelease(SQLModel, table=True):  # type: ignore[valid-type,misc]
    __tablename__ = "bot_release"

    id: str = Field(
        default_factory=get_id,
        primary_key=True,
        description="Primary key ID, snowflake ID",
    )
    bot_id: str = Field(
        ..., description="Business foreign key, assistant table primary key"
    )
    version: str = Field(..., max_length=64, description="Version")
    description: Optional[str] = Field(
        default=None, max_length=255, description="Version description"
    )
    dsl: str = Field(
        ..., description="Current version assistant orchestration protocol"
    )
    create_at: datetime = Field(
        default_factory=datetime.now, description="Creation time"
    )
    update_at: datetime = Field(default_factory=datetime.now, description="Update time")
