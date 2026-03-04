import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from celery import Celery
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from qdrant_client import QdrantClient
from redis import Redis
from redis.exceptions import RedisError
from sentence_transformers import SentenceTransformer

from schemas.job import CreateJobRequest, CreateJobResponse, JobRecord
from schemas.search import SearchRequest
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
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3").strip().lower()
JOB_CREATED_CHANNEL = "jobs.created"
celery_client = Celery("api_client", broker=REDIS_URL, backend=REDIS_URL)

MODEL_MAP = {
    "bge-m3": "BAAI/bge-m3",
    "e5-large": "intfloat/e5-large-v2",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_sse(event_type: str, payload: Any) -> str:
    message = {
        "type": event_type,
        "ts": _utc_now_iso(),
        "payload": payload,
    }
    return f"event: {event_type}\ndata: {json.dumps(message, ensure_ascii=False)}\n\n"


def _resolve_model_name(model_key: str) -> str:
    if model_key in MODEL_MAP:
        return MODEL_MAP[model_key]
    return model_key


@lru_cache(maxsize=1)
def _get_encoder() -> SentenceTransformer:
    return SentenceTransformer(_resolve_model_name(EMBEDDING_MODEL))


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


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

    try:
        celery_client.send_task("pipeline_run", args=[job_id])
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"failed_to_enqueue_pipeline_task: {exc}") from exc

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


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    job_dir = get_job_dir(DATA_ROOT, job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"job_not_found: {job_id}")

    state_file = job_dir / "state.json"
    log_file = job_dir / "logs" / "live.log"

    async def event_stream() -> Any:
        job_json = load_job_json(job_dir)
        state = load_or_create_state(job_dir, job_json)
        state_signature = json.dumps(state, sort_keys=True)
        log_offset = 0

        yield _format_sse("state", state)

        while True:
            if await request.is_disconnected():
                break

            try:
                if state_file.exists():
                    next_state = load_or_create_state(job_dir, load_job_json(job_dir))
                    next_signature = json.dumps(next_state, sort_keys=True)
                    if next_signature != state_signature:
                        state_signature = next_signature
                        yield _format_sse("state", next_state)
            except OSError:
                pass

            if log_file.exists() and log_file.is_file():
                try:
                    current_size = log_file.stat().st_size
                    if current_size < log_offset:
                        log_offset = 0

                    if current_size > log_offset:
                        with log_file.open("r", encoding="utf-8", errors="replace") as handle:
                            handle.seek(log_offset)
                            for line in handle:
                                line_text = line.rstrip("\r\n")
                                if line_text:
                                    yield _format_sse("log", {"line": line_text})
                            log_offset = handle.tell()
                except OSError:
                    pass

            await asyncio.sleep(1)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.get("/jobs/{job_id}/artifacts")
def get_job_artifacts(job_id: str) -> dict[str, Any]:
    job_dir = get_job_dir(DATA_ROOT, job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"job_not_found: {job_id}")

    artifacts = list_artifacts(job_dir)
    return {"job_id": job_id, "artifacts": artifacts}


@app.post("/jobs/{job_id}/search")
def search_job(job_id: str, payload: SearchRequest) -> dict[str, Any]:
    collection_name = f"video_{job_id}"

    try:
        encoder = _get_encoder()
        vector = encoder.encode([payload.query], normalize_embeddings=True, show_progress_bar=False)[0]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"embedding_failed: {exc}") from exc

    try:
        client = _get_qdrant_client()
        hits = client.search(
            collection_name=collection_name,
            query_vector=vector.tolist(),
            limit=payload.top_k,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"search_failed_or_collection_not_found: {collection_name} ({exc})",
        ) from exc

    results: list[dict[str, Any]] = []
    for hit in hits:
        data = hit.payload or {}
        snippet = str(data.get("text") or "").strip()
        if len(snippet) > 320:
            snippet = snippet[:317] + "..."
        results.append(
            {
                "t0": data.get("t0"),
                "t1": data.get("t1"),
                "snippet": snippet,
                "chunk_id": data.get("chunk_id"),
                "score": float(hit.score),
            }
        )

    return {"results": results}


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
