import json
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


def get_job_dir(data_root: str, job_id: str) -> Path:
    return Path(data_root) / job_id


def load_job_json(job_dir: Path) -> dict[str, Any] | None:
    job_file = job_dir / "input" / "job.json"
    if not job_file.exists():
        return None
    return _read_json(job_file)


def load_steps(job_dir: Path) -> list[dict[str, Any]]:
    steps_dir = job_dir / "steps"
    if not steps_dir.exists():
        return []

    steps: list[dict[str, Any]] = []
    for step_file in sorted(steps_dir.rglob("*.json")):
        try:
            payload = _read_json(step_file)
        except (OSError, json.JSONDecodeError):
            payload = {}

        step_name = str(payload.get("step") or step_file.stem)
        status = str(payload.get("status") or "UNKNOWN")
        progress_raw = payload.get("progress", 0)
        try:
            progress_value = float(progress_raw)
        except (TypeError, ValueError):
            progress_value = 0.0
        progress_value = max(0.0, min(progress_value, 100.0))

        steps.append(
            {
                "step": step_name,
                "status": status,
                "progress": progress_value,
                "path": str(step_file.relative_to(job_dir)).replace("\\", "/"),
            }
        )

    return steps


def compute_progress(steps: list[dict[str, Any]], status: str) -> float:
    if not steps:
        return 100.0 if status == "COMPLETED" else 0.0

    total = 0.0
    for step in steps:
        try:
            total += float(step.get("progress", 0))
        except (TypeError, ValueError):
            total += 0.0
    return round(total / len(steps), 2)


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