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
from transformers import pipeline

from schemas.chat import ChatRequest, CreateChatSessionResponse
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
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
CHAT_PROMPT_PATH = os.getenv("CHAT_PROMPT_PATH", "/shared/models/prompts/chat_answer.txt")
JOB_CREATED_CHANNEL = "jobs.created"
celery_client = Celery("api_client", broker=REDIS_URL, backend=REDIS_URL)
CPU_PIPELINE_QUEUE = "cpu_pipeline"

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


@lru_cache(maxsize=1)
def _get_chat_generator() -> Any:
    return pipeline("text-generation", model=CHAT_MODEL, device_map="auto")


@lru_cache(maxsize=1)
def _get_chat_prompt_template() -> str:
    prompt_file = Path(CHAT_PROMPT_PATH)
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    return (
        "You are a video-only assistant. Answer only using provided video context. "
        "If the answer is not present, respond with 'Bu bilgi videoda yok.'"
    )


def _chat_dir(job_id: str) -> Path:
    return Path(DATA_ROOT) / job_id / "chat"


def _chat_session_file(job_id: str, session_id: str) -> Path:
    return _chat_dir(job_id) / f"{session_id}.json"


def _load_chat_session(job_id: str, session_id: str) -> dict[str, Any]:
    session_file = _chat_session_file(job_id, session_id)
    if not session_file.exists():
        raise HTTPException(status_code=404, detail=f"chat_session_not_found: {session_id}")
    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"chat_session_corrupted: {session_id}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail=f"chat_session_invalid_format: {session_id}")
    if not isinstance(data.get("messages"), list):
        data["messages"] = []
    return data


def _save_chat_session(job_id: str, session_id: str, data: dict[str, Any]) -> None:
    session_file = _chat_session_file(job_id, session_id)
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _retrieve_sources(job_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
    collection_name = f"video_{job_id}"
    encoder = _get_encoder()
    query_vector = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

    hits = _get_qdrant_client().search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=max(top_k, 2),
        with_payload=True,
        with_vectors=False,
    )

    sources: list[dict[str, Any]] = []
    for hit in hits:
        payload = hit.payload or {}
        snippet = str(payload.get("text") or "").strip()
        if len(snippet) > 500:
            snippet = snippet[:497] + "..."
        sources.append(
            {
                "t0": payload.get("t0"),
                "t1": payload.get("t1"),
                "snippet": snippet,
                "chunk_id": payload.get("chunk_id"),
                "score": float(hit.score),
            }
        )
    return sources


def _generate_chat_answer(message: str, sources: list[dict[str, Any]], history: list[dict[str, Any]]) -> str:
    if not sources:
        return "Bu bilgi videoda yok."

    prompt_template = _get_chat_prompt_template()
    history_tail = history[-6:]
    history_text = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in history_tail])
    context_text = "\n".join(
        [f"[{s.get('t0')} - {s.get('t1')}] ({s.get('chunk_id')}): {s.get('snippet')}" for s in sources]
    )
    prompt = (
        f"{prompt_template}\n\n"
        f"SOHBET_GECMISI:\n{history_text}\n\n"
        f"VIDEO_BAGLAM:\n{context_text}\n\n"
        f"KULLANICI_SORUSU:\n{message}\n\n"
        "YANIT:"
    )

    try:
        generator = _get_chat_generator()
        output = generator(prompt, max_new_tokens=220, do_sample=False, temperature=0.1)
        generated = str(output[0].get("generated_text", "")).strip()
        answer = generated[len(prompt) :].strip() if generated.startswith(prompt) else generated
        if not answer:
            return "Bu bilgi videoda yok."
        return answer
    except Exception:
        if sources:
            return f"Videoya göre: {sources[0].get('snippet') or 'Bu bilgi videoda yok.'}"
        return "Bu bilgi videoda yok."


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
        celery_client.send_task("pipeline_run", args=[job_id], queue=CPU_PIPELINE_QUEUE)
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


@app.post("/jobs/{job_id}/chat/sessions", response_model=CreateChatSessionResponse)
def create_chat_session(job_id: str) -> CreateChatSessionResponse:
    job_dir = get_job_dir(DATA_ROOT, job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"job_not_found: {job_id}")

    session_id = str(uuid.uuid4())
    data = {
        "session_id": session_id,
        "job_id": job_id,
        "created_at": _utc_now_iso(),
        "messages": [],
    }
    _save_chat_session(job_id, session_id, data)
    return CreateChatSessionResponse(session_id=session_id)


@app.post("/jobs/{job_id}/chat")
def chat_with_job(job_id: str, payload: ChatRequest) -> dict[str, Any]:
    job_dir = get_job_dir(DATA_ROOT, job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"job_not_found: {job_id}")

    session = _load_chat_session(job_id, payload.session_id)
    if str(session.get("job_id")) != job_id:
        raise HTTPException(status_code=400, detail="chat_session_job_mismatch")

    try:
        sources = _retrieve_sources(job_id, payload.message, payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"chat_retrieve_failed: {exc}") from exc

    answer = _generate_chat_answer(payload.message, sources, session.get("messages", []))

    session["messages"].append({"role": "user", "content": payload.message, "ts": _utc_now_iso()})
    session["messages"].append({"role": "assistant", "content": answer, "ts": _utc_now_iso(), "sources": sources})
    session["updated_at"] = _utc_now_iso()
    _save_chat_session(job_id, payload.session_id, session)

    response_sources = [{"t0": s["t0"], "t1": s["t1"], "snippet": s["snippet"], "chunk_id": s["chunk_id"]} for s in sources]
    return {"answer": answer, "sources": response_sources}


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
