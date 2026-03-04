import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from redis import Redis
from redis.exceptions import RedisError

from schemas.job import CreateJobRequest, CreateJobResponse, JobRecord
from storage import (
    build_default_state,
    get_job_dir,
    list_artifacts,
    load_job_json,
    load_or_create_state,
    resolve_artifact,
    save_state,
)

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
    job_dir = Path(DATA_ROOT) / job_id
    input_dir = job_dir / "input"
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
        save_state(job_dir, build_default_state(job.model_dump()))
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


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job_dir = get_job_dir(DATA_ROOT, job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"job_not_found: {job_id}")

    job_json = load_job_json(job_dir)
    state = load_or_create_state(job_dir, job_json)
    artifacts = list_artifacts(job_dir)

    return {
        "job_id": job_id,
        "status": state["status"],
        "progress": state["progress"],
        "current_step": state["current_step"],
        "steps": state["steps"],
        "updated_at": state["updated_at"],
        "job": job_json,
        "artifacts": artifacts,
    }


@app.get("/jobs/{job_id}/artifacts")
def get_job_artifacts(job_id: str) -> dict[str, Any]:
    job_dir = get_job_dir(DATA_ROOT, job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"job_not_found: {job_id}")

    artifacts = list_artifacts(job_dir)
    return {"job_id": job_id, "artifacts": artifacts}


@app.get("/jobs/{job_id}/artifacts/{key}")
def get_job_artifact(job_id: str, key: str) -> FileResponse:
    job_dir = get_job_dir(DATA_ROOT, job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"job_not_found: {job_id}")

    try:
        artifact_path, artifact_type, relative_path = resolve_artifact(job_dir, key)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"artifact_key_not_found: {key}")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"artifact_file_not_found: {key}")

    media_type_map = {
        "json": "application/json",
        "csv": "text/csv; charset=utf-8",
        "markdown": "text/markdown; charset=utf-8",
    }
    media_type = media_type_map.get(artifact_type, "text/plain; charset=utf-8")

    return FileResponse(path=artifact_path, media_type=media_type, filename=Path(relative_path).name)