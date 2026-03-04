import json
import os
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
    job_dir = Path(DATA_ROOT) / job_id
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
    return {"job_id": job_id, "collection": collection_name, "points_upserted": len(points), "model": hf_model_name}
