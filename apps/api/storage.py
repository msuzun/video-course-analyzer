import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACT_REGISTRY: dict[str, dict[str, str]] = {
    "transcript_md": {"path": "outputs/transcript.md", "type": "markdown"},
    "summary_md": {"path": "outputs/summary.md", "type": "markdown"},
    "summary_json": {"path": "outputs/summary.json", "type": "json"},
    "chapters_json": {"path": "outputs/chapters.json", "type": "json"},
    "segments_csv": {"path": "outputs/segments.csv", "type": "csv"},
    "embeddings_manifest": {"path": "rag/embeddings_manifest.json", "type": "json"},
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_job_dir(data_root: str, job_id: str) -> Path:
    return Path(data_root) / job_id


def load_job_json(job_dir: Path) -> dict[str, Any] | None:
    job_file = job_dir / "input" / "job.json"
    if not job_file.exists():
        return None
    return _read_json(job_file)


def build_default_state(job_json: dict[str, Any] | None = None) -> dict[str, Any]:
    status = "QUEUED"
    if isinstance(job_json, dict):
        status = str(job_json.get("state") or status)

    return {
        "status": status,
        "progress": 0.0,
        "current_step": None,
        "steps": [],
        "updated_at": _utc_now_iso(),
    }


def _normalize_step(step: Any) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {"name": "unknown", "state": "UNKNOWN", "progress": 0.0}

    name = str(step.get("name") or step.get("step") or "unknown")
    state = str(step.get("state") or step.get("status") or "UNKNOWN")

    try:
        progress = float(step.get("progress", 0.0))
    except (TypeError, ValueError):
        progress = 0.0
    progress = max(0.0, min(progress, 100.0))

    return {"name": name, "state": state, "progress": progress}


def _normalize_state(raw_state: dict[str, Any], fallback_status: str = "QUEUED") -> dict[str, Any]:
    status = str(raw_state.get("status") or fallback_status)

    try:
        progress = float(raw_state.get("progress", 0.0))
    except (TypeError, ValueError):
        progress = 0.0
    progress = max(0.0, min(progress, 100.0))

    current_step_raw = raw_state.get("current_step")
    current_step = str(current_step_raw) if current_step_raw is not None else None

    raw_steps = raw_state.get("steps")
    steps_input = raw_steps if isinstance(raw_steps, list) else []
    steps = [_normalize_step(step) for step in steps_input]

    updated_at = raw_state.get("updated_at")
    updated_at_value = str(updated_at) if updated_at else _utc_now_iso()

    return {
        "status": status,
        "progress": progress,
        "current_step": current_step,
        "steps": steps,
        "updated_at": updated_at_value,
    }


def save_state(job_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_state(state)
    state_file = job_dir / "state.json"
    state_file.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def load_or_create_state(job_dir: Path, job_json: dict[str, Any] | None = None) -> dict[str, Any]:
    state_file = job_dir / "state.json"
    fallback_status = str((job_json or {}).get("state") or "QUEUED")

    if not state_file.exists():
        return save_state(job_dir, build_default_state(job_json))

    try:
        raw_state = _read_json(state_file)
    except (OSError, json.JSONDecodeError):
        return save_state(job_dir, build_default_state(job_json))

    normalized = _normalize_state(raw_state, fallback_status=fallback_status)

    if normalized != raw_state:
        return save_state(job_dir, normalized)
    return normalized


def list_artifacts(job_dir: Path) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for key, item in ARTIFACT_REGISTRY.items():
        artifact_path = job_dir / item["path"]
        if artifact_path.exists() and artifact_path.is_file():
            artifacts.append(
                {
                    "key": key,
                    "path": item["path"],
                    "type": item["type"],
                }
            )
    return artifacts


def resolve_artifact(job_dir: Path, key: str) -> tuple[Path, str, str]:
    item = ARTIFACT_REGISTRY.get(key)
    if item is None:
        raise KeyError(key)

    artifact_path = job_dir / item["path"]
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(item["path"])

    return artifact_path, item["type"], item["path"]