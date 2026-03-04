from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: str
    source_url: HttpUrl
    options: dict[str, Any] = Field(default_factory=dict)


class CreateJobResponse(BaseModel):
    job_id: str


class JobRecord(BaseModel):
    job_id: str
    state: Literal["QUEUED"]
    source_type: str
    source_url: str
    options: dict[str, Any]

