import csv
import subprocess
from pathlib import Path


def _run_command(command: list[str], error_prefix: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{error_prefix}: command_not_found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"{error_prefix}: {stderr}") from exc


def _load_scenes(scenes_csv: Path) -> list[dict[str, float | int]]:
    scenes: list[dict[str, float | int]] = []
    with scenes_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                scene_id = int(row.get("scene_id", "0") or 0)
                start_sec = float(row.get("start_sec", "0") or 0.0)
                end_sec = float(row.get("end_sec", "0") or 0.0)
            except ValueError:
                continue
            scenes.append({"scene_id": scene_id, "start_sec": start_sec, "end_sec": end_sec})
    return scenes


def run_keyframes(job_id: str, data_root: str) -> dict[str, str | int]:
    job_dir = Path(data_root) / job_id
    video_path = job_dir / "normalized" / "video.mp4"
    scenes_csv = job_dir / "vision" / "scenes.csv"
    if not video_path.exists():
        raise RuntimeError("keyframes_failed: missing_video: normalized/video.mp4")
    if not scenes_csv.exists():
        raise RuntimeError("keyframes_failed: missing_scenes_csv: vision/scenes.csv")

    keyframes_dir = job_dir / "vision" / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    scenes = _load_scenes(scenes_csv)
    written = 0
    for scene in scenes:
        scene_id = int(scene["scene_id"])
        start_sec = float(scene["start_sec"])
        end_sec = float(scene["end_sec"])
        mid_sec = start_sec + max(0.0, (end_sec - start_sec) / 2.0)
        output_path = keyframes_dir / f"scene_{scene_id:04d}_kf1.jpg"
        _run_command(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{mid_sec:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(output_path),
            ],
            f"keyframe_extract_failed_scene_{scene_id}",
        )
        written += 1

    return {
        "video_path": str(video_path),
        "scenes_csv": str(scenes_csv),
        "keyframes_dir": str(keyframes_dir),
        "keyframe_count": written,
    }
