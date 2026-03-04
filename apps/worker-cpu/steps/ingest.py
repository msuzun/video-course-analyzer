import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen


def _is_direct_mp4_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    return parsed.path.lower().endswith(".mp4")


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:  # nosec B310
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def _run_command(command: list[str], error_prefix: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{error_prefix}: command_not_found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"{error_prefix}: {stderr}") from exc


def _parse_fps(value: str) -> float:
    if "/" in value:
        num, denom = value.split("/", 1)
        try:
            numerator = float(num)
            denominator = float(denom)
            if denominator == 0:
                return 0.0
            return round(numerator / denominator, 3)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_meta(video_path: Path, meta_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("meta_extract_failed: command_not_found: ffprobe") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"meta_extract_failed: {stderr}") from exc

    raw = json.loads(result.stdout or "{}")
    streams = raw.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})

    duration_raw = (raw.get("format", {}) or {}).get("duration")
    try:
        duration = float(duration_raw) if duration_raw is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    width = int(video_stream.get("width", 0) or 0)
    height = int(video_stream.get("height", 0) or 0)
    fps = _parse_fps(str(video_stream.get("r_frame_rate", "0/1")))

    meta = {
        "duration_sec": round(duration, 3),
        "fps": fps,
        "resolution": f"{width}x{height}" if width and height else None,
        "width": width,
        "height": height,
        "source_video": str(video_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def run_ingest(job_id: str, data_root: str) -> dict[str, Any]:
    job_dir = Path(data_root) / job_id
    input_dir = job_dir / "input"
    normalized_dir = job_dir / "normalized"

    source_file = input_dir / "source.mp4"
    job_json_file = input_dir / "job.json"

    if not source_file.exists():
        if not job_json_file.exists():
            raise RuntimeError("ingest_failed: missing_input_file: input/source.mp4 and input/job.json")

        job_payload = json.loads(job_json_file.read_text(encoding="utf-8"))
        source_url = str(job_payload.get("source_url") or "")
        if not source_url:
            raise RuntimeError("ingest_failed: missing_source_url_in_job_json")
        if not _is_direct_mp4_url(source_url):
            raise RuntimeError("ingest_failed: only_direct_mp4_url_supported")
        _download_file(source_url, source_file)

    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_video = normalized_dir / "video.mp4"
    normalized_audio = normalized_dir / "audio.wav"
    normalized_meta = normalized_dir / "meta.json"

    _run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_file),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            str(normalized_video),
        ],
        "video_normalize_failed",
    )

    _run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(normalized_video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(normalized_audio),
        ],
        "audio_extract_failed",
    )

    meta = _extract_meta(normalized_video, normalized_meta)
    return {
        "source_path": str(source_file),
        "video_path": str(normalized_video),
        "audio_path": str(normalized_audio),
        "meta_path": str(normalized_meta),
        "meta": meta,
    }
