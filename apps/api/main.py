import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from redis import Redis
from redis.exceptions import RedisError

from schemas.job import CreateJobRequest, CreateJobResponse, JobRecord

app = FastAPI(title="video-course-analyzer-api")

DATA_ROOT = os.getenv("DATA_ROOT", "/data/jobs")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
JOB_CREATED_CHANNEL = "jobs.created"


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/jobs", response_model=CreateJobResponse)
def create_job(payload: CreateJobRequest) -> CreateJobResponse:
    job_id = str(uuid.uuid4())
    input_dir = Path(DATA_ROOT) / job_id / "input"
    job_file = input_dir / "job.json"

    job = JobRecord(
        job_id=job_id,
        state="QUEUED",
        source_type=payload.source_type,
        source_url=str(payload.source_url),
        options=payload.options,
    )

    try:
        input_dir.mkdir(parents=True, exist_ok=False)
        job_file.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"failed_to_persist_job: {exc}") from exc

    event_payload: dict[str, Any] = {
        "event": "job created",
        "job_id": job_id,
        "state": "QUEUED",
    }

    try:
        redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.publish(JOB_CREATED_CHANNEL, json.dumps(event_payload))
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"failed_to_publish_job_event: {exc}") from exc

    return CreateJobResponse(job_id=job_id)
