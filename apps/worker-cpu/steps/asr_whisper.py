import json
import math
import os
from pathlib import Path
from typing import Any

import whisper


def _format_ts(seconds: float) -> str:
    total = int(max(0.0, seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _segment_confidence(segment: dict[str, Any]) -> float:
    avg_logprob = segment.get("avg_logprob")
    try:
        score = float(avg_logprob)
    except (TypeError, ValueError):
        return 0.0
    confidence = math.exp(score)
    return round(max(0.0, min(confidence, 1.0)), 4)


def _read_language_hint(job_dir: Path) -> str | None:
    job_json_path = job_dir / "input" / "job.json"
    if not job_json_path.exists():
        return None
    try:
        job_payload = json.loads(job_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    options = job_payload.get("options")
    if not isinstance(options, dict):
        return None
    hint = options.get("language_hint")
    if hint is None:
        return None
    hint_value = str(hint).strip().lower()
    return hint_value or None


def run_asr_whisper(job_id: str, data_root: str) -> dict[str, Any]:
    job_dir = Path(data_root) / job_id
    audio_path = job_dir / "normalized" / "audio.wav"
    if not audio_path.exists():
        raise RuntimeError("asr_failed: missing_audio: normalized/audio.wav")

    asr_dir = job_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    transcript_jsonl = asr_dir / "transcript.jsonl"
    transcript_md = asr_dir / "transcript.md"

    model_name = os.getenv("WHISPER_MODEL", "base")
    language_hint = _read_language_hint(job_dir)

    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), language=language_hint, verbose=False)

    segments = result.get("segments", [])
    with transcript_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        for segment in segments:
            t0 = float(segment.get("start", 0.0) or 0.0)
            t1 = float(segment.get("end", 0.0) or 0.0)
            text = str(segment.get("text", "")).strip()
            row = {
                "t0": round(t0, 3),
                "t1": round(t1, 3),
                "text": text,
                "confidence": _segment_confidence(segment),
            }
            jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    md_lines: list[str] = ["# Transcript", ""]
    for segment in segments:
        t0 = float(segment.get("start", 0.0) or 0.0)
        t1 = float(segment.get("end", 0.0) or 0.0)
        text = str(segment.get("text", "")).strip()
        md_lines.append(f"## [{_format_ts(t0)} - {_format_ts(t1)}]")
        md_lines.append(text)
        md_lines.append("")
    transcript_md.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    return {
        "audio_path": str(audio_path),
        "transcript_jsonl": str(transcript_jsonl),
        "transcript_md": str(transcript_md),
        "language_hint": language_hint,
        "model": model_name,
        "segments": len(segments),
    }
