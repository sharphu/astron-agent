from typing import Literal, Optional

from pydantic import BaseModel, Field

from agent.api.schemas.base_inputs import BaseInputs
from agent.api.schemas_v2.bot_dsl import Dsl


class MessageInputs(BaseModel):
    content: str = Field(...)
    content_type: Literal["text"] = Field(...)
    role: Literal["user", "assistant", "system"]


class DebugChat(BaseInputs):
    bot_id: str = Field(...)
    conversation_id: str = Field(...)
    stream: bool = Field(...)
    messages: list[MessageInputs] = Field(...)
    dsl: Optional[Dsl] = Field(default=None)


class Chat(BaseInputs):
    bot_id: str = Field(...)
    uid: str = Field(...)
    conversation_id: str = Field(...)
    stream: bool = Field(...)
    messages: list[MessageInputs] = Field(...)
    version_name: str = Field(...)
