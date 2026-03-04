import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from celery import Celery
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
DATA_ROOT = os.getenv("DATA_ROOT", "/data/jobs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3").strip().lower()

MODEL_MAP = {
    "bge-m3": "BAAI/bge-m3",
    "e5-large": "intfloat/e5-large-v2",
}

celery_app = Celery("rag_indexer", broker=REDIS_URL, backend=REDIS_URL)
WRITER_QUEUE = "writer_llm"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    return Path(DATA_ROOT) / job_id


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "QUEUED",
            "progress": 0.0,
            "current_step": None,
            "steps": [],
            "updated_at": _now_iso(),
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "QUEUED",
            "progress": 0.0,
            "current_step": None,
            "steps": [],
            "updated_at": _now_iso(),
        }


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _upsert_step(steps: list[dict[str, Any]], name: str, step_state: str, progress: float) -> list[dict[str, Any]]:
    updated = False
    for step in steps:
        if step.get("name") == name:
            step["state"] = step_state
            step["progress"] = progress
            updated = True
            break
    if not updated:
        steps.append({"name": name, "state": step_state, "progress": progress})
    return steps


def append_live_log(job_id: str, line: str) -> None:
    log_file = _job_dir(job_id) / "logs" / "live.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_now_iso()}] {line}\n")


def update_step_state(job_id: str, status: str, current_step: str | None, step_name: str, step_state: str, progress: float) -> None:
    state_file = _job_dir(job_id) / "state.json"
    state = _read_state(state_file)
    steps = state.get("steps")
    if not isinstance(steps, list):
        steps = []

    normalized_progress = max(0.0, min(float(progress), 100.0))
    state["status"] = status
    state["progress"] = normalized_progress
    state["current_step"] = current_step
    state["steps"] = _upsert_step(steps, step_name, step_state, normalized_progress)
    state["updated_at"] = _now_iso()
    _write_state(state_file, state)


def _resolve_model_name(model_key: str) -> str:
    if model_key in MODEL_MAP:
        return MODEL_MAP[model_key]
    return model_key


def _load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    if not chunks_path.exists():
        raise RuntimeError(f"rag_index_failed: missing_chunks_file: {chunks_path}")

    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                chunks.append(item)
    if not chunks:
        raise RuntimeError("rag_index_failed: no_chunks_to_index")
    return chunks


def _chunk_text(chunk: dict[str, Any]) -> str:
    transcript = str(chunk.get("transcript", "") or "").strip()
    ocr = str(chunk.get("ocr", "") or "").strip()
    if transcript and ocr:
        return f"{transcript}\n{ocr}"
    return transcript or ocr or ""


@celery_app.task(name="rag_index")
def rag_index(job_id: str) -> dict[str, Any]:
    try:
        append_live_log(job_id, "rag_index started")
        update_step_state(job_id, "RAG_INDEX_RUNNING", "rag_index", "rag_index", "RUNNING", 99.0)
        job_dir = _job_dir(job_id)
        chunks_path = job_dir / "rag" / "chunks.jsonl"
        chunks = _load_chunks(chunks_path)

        hf_model_name = _resolve_model_name(EMBEDDING_MODEL)
        encoder = SentenceTransformer(hf_model_name)

        texts = [_chunk_text(chunk) for chunk in chunks]
        vectors = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        vector_size = len(vectors[0])
        collection_name = f"video_{job_id}"
        client = QdrantClient(url=QDRANT_URL)

        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )

        points: list[qmodels.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
            payload = {
                "chunk_id": chunk.get("chunk_id"),
                "t0": chunk.get("t0"),
                "t1": chunk.get("t1"),
                "text": _chunk_text(chunk),
                "source_refs": chunk.get("source_refs", []),
            }
            points.append(
                qmodels.PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points, wait=True)
        update_step_state(job_id, "WRITER_QUEUED", "writer_llm", "rag_index", "COMPLETED", 99.0)
        update_step_state(job_id, "WRITER_QUEUED", "writer_llm", "writer_llm", "QUEUED", 99.0)
        append_live_log(job_id, "rag_index completed")
        append_live_log(job_id, "writer_llm enqueue started")
        celery_app.send_task("writer_llm", args=[job_id], queue=WRITER_QUEUE)
        append_live_log(job_id, "writer_llm enqueue completed")
        return {"job_id": job_id, "collection": collection_name, "points_upserted": len(points), "model": hf_model_name}
    except Exception as exc:
        update_step_state(job_id, "FAILED", "rag_index", "rag_index", "FAILED", 99.0)
        append_live_log(job_id, f"rag_index failed: {exc}")
        raise
