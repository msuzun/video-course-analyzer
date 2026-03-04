import csv
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _read_scenes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    scenes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                scene_id = int(row.get("scene_id", "0") or 0)
                start_sec = float(row.get("start_sec", "0") or 0.0)
                end_sec = float(row.get("end_sec", "0") or 0.0)
            except ValueError:
                continue
            scenes.append(
                {
                    "scene_id": scene_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                }
            )
    return scenes


def _timeline_end(transcript_rows: list[dict[str, Any]], ocr_rows: list[dict[str, Any]], scenes: list[dict[str, Any]]) -> float:
    candidates = [0.0]
    for row in transcript_rows:
        try:
            candidates.append(float(row.get("t1", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
    for row in ocr_rows:
        try:
            candidates.append(float(row.get("ts", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
    for scene in scenes:
        try:
            candidates.append(float(scene.get("end_sec", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
    return max(candidates)


def _text_from_transcript(rows: list[dict[str, Any]], t0: float, t1: float) -> tuple[str, list[dict[str, Any]]]:
    snippets: list[str] = []
    refs: list[dict[str, Any]] = []
    for row in rows:
        try:
            seg_t0 = float(row.get("t0", 0.0) or 0.0)
            seg_t1 = float(row.get("t1", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if seg_t1 < t0 or seg_t0 > t1:
            continue
        text = str(row.get("text", "")).strip()
        if text:
            snippets.append(text)
        refs.append({"type": "transcript", "t0": seg_t0, "t1": seg_t1})
    return " ".join(snippets).strip(), refs


def _text_from_ocr(rows: list[dict[str, Any]], t0: float, t1: float) -> tuple[str, list[dict[str, Any]]]:
    snippets: list[str] = []
    refs: list[dict[str, Any]] = []
    for row in rows:
        try:
            ts = float(row.get("ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if ts < t0 or ts > t1:
            continue
        text = str(row.get("text", "")).strip()
        if text:
            snippets.append(text)
        refs.append(
            {
                "type": "ocr",
                "ts": ts,
                "scene_id": row.get("scene_id"),
                "frame": row.get("frame"),
            }
        )
    return " ".join(snippets).strip(), refs


def _refs_from_scenes(scenes: list[dict[str, Any]], t0: float, t1: float) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for scene in scenes:
        start_sec = float(scene.get("start_sec", 0.0) or 0.0)
        end_sec = float(scene.get("end_sec", 0.0) or 0.0)
        if end_sec < t0 or start_sec > t1:
            continue
        refs.append(
            {
                "type": "scene",
                "scene_id": scene.get("scene_id"),
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )
    return refs


def run_chunking(job_id: str, data_root: str) -> dict[str, str | int]:
    job_dir = Path(data_root) / job_id
    transcript_path = job_dir / "asr" / "transcript.jsonl"
    ocr_path = job_dir / "vision" / "ocr.jsonl"
    scenes_path = job_dir / "vision" / "scenes.csv"

    transcript_rows = _read_jsonl(transcript_path)
    ocr_rows = _read_jsonl(ocr_path)
    scenes = _read_scenes(scenes_path)

    timeline_end = _timeline_end(transcript_rows, ocr_rows, scenes)
    if timeline_end <= 0:
        raise RuntimeError("chunking_failed: no_timeline_data")

    rag_dir = job_dir / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = rag_dir / "chunks.jsonl"

    window = 60.0
    overlap = 15.0
    step = window - overlap
    chunk_count = 0

    with chunks_path.open("w", encoding="utf-8") as handle:
        cursor = 0.0
        while cursor < timeline_end:
            chunk_t0 = round(cursor, 3)
            chunk_t1 = round(min(cursor + window, timeline_end), 3)

            transcript_text, transcript_refs = _text_from_transcript(transcript_rows, chunk_t0, chunk_t1)
            ocr_text, ocr_refs = _text_from_ocr(ocr_rows, chunk_t0, chunk_t1)
            scene_refs = _refs_from_scenes(scenes, chunk_t0, chunk_t1)

            chunk_count += 1
            chunk = {
                "chunk_id": f"{job_id}_c{chunk_count:04d}",
                "t0": chunk_t0,
                "t1": chunk_t1,
                "transcript": transcript_text,
                "ocr": ocr_text,
                "vlm": "",
                "source_refs": transcript_refs + ocr_refs + scene_refs,
            }
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            if chunk_t1 >= timeline_end:
                break
            cursor += step

    return {
        "chunks_jsonl": str(chunks_path),
        "chunk_count": chunk_count,
        "window_sec": int(window),
        "overlap_sec": int(overlap),
    }
