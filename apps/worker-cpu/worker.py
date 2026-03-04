import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
DATA_ROOT = os.getenv("DATA_ROOT", "/data/jobs")

celery_app = Celery("worker_cpu", broker=REDIS_URL, backend=REDIS_URL)


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


def update_ingest_running(job_id: str, progress: float = 5.0) -> dict[str, Any]:
    state_file = _job_dir(job_id) / "state.json"
    state = _read_state(state_file)

    normalized_progress = max(0.0, min(float(progress), 100.0))
    steps = state.get("steps")
    if not isinstance(steps, list):
        steps = []

    state["status"] = "INGEST_RUNNING"
    state["progress"] = normalized_progress
    state["current_step"] = "ingest"
    state["steps"] = _upsert_step(steps, "ingest", "RUNNING", normalized_progress)
    state["updated_at"] = _now_iso()

    _write_state(state_file, state)
    return state


@celery_app.task(name="pipeline_run")
def pipeline_run(job_id: str) -> dict[str, Any]:
    append_live_log(job_id, f"pipeline_run started job_id={job_id}")
    state = update_ingest_running(job_id, progress=10.0)
    return {"job_id": job_id, "status": state["status"], "progress": state["progress"]}
