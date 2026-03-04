import csv
from pathlib import Path

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


def run_scene_detect(job_id: str, data_root: str) -> dict[str, str | int]:
    job_dir = Path(data_root) / job_id
    video_path = job_dir / "normalized" / "video.mp4"
    if not video_path.exists():
        raise RuntimeError("scenedetect_failed: missing_video: normalized/video.mp4")

    vision_dir = job_dir / "vision"
    vision_dir.mkdir(parents=True, exist_ok=True)
    scenes_csv = vision_dir / "scenes.csv"

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()

    with scenes_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene_id",
                "start_sec",
                "end_sec",
                "duration_sec",
                "start_frame",
                "end_frame",
            ],
        )
        writer.writeheader()
        for idx, (start_time, end_time) in enumerate(scenes, start=1):
            start_sec = float(start_time.get_seconds())
            end_sec = float(end_time.get_seconds())
            writer.writerow(
                {
                    "scene_id": idx,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "duration_sec": round(max(0.0, end_sec - start_sec), 3),
                    "start_frame": start_time.get_frames(),
                    "end_frame": end_time.get_frames(),
                }
            )

    return {
        "video_path": str(video_path),
        "scenes_csv": str(scenes_csv),
        "scene_count": len(scenes),
    }
