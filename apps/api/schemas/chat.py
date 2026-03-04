from pydantic import BaseModel, ConfigDict, Field


class CreateChatSessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
