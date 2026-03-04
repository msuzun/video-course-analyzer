import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import pytesseract

SCENE_ID_PATTERN = re.compile(r"scene_(\d+)_kf1\.jpg$", re.IGNORECASE)


def _load_scene_midpoints(scenes_csv: Path) -> dict[int, float]:
    mapping: dict[int, float] = {}
    if not scenes_csv.exists():
        return mapping

    with scenes_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                scene_id = int(row.get("scene_id", "0") or 0)
                start_sec = float(row.get("start_sec", "0") or 0.0)
                end_sec = float(row.get("end_sec", "0") or 0.0)
            except ValueError:
                continue
            mapping[scene_id] = round(start_sec + ((end_sec - start_sec) / 2.0), 3)
    return mapping


def _extract_scene_id(frame_name: str) -> int:
    match = SCENE_ID_PATTERN.search(frame_name)
    if not match:
        return 0
    return int(match.group(1))


def _edge_density(image: Any) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    non_zero = cv2.countNonZero(edges)
    total = max(1, edges.shape[0] * edges.shape[1])
    return float(non_zero) / float(total)


def _run_tesseract(image: Any) -> tuple[str, float]:
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--oem 1 --psm 6",
    )
    words: list[str] = []
    confs: list[float] = []
    for text, conf_raw in zip(data.get("text", []), data.get("conf", [])):
        token = str(text or "").strip()
        if not token:
            continue
        try:
            conf_value = float(conf_raw)
        except (TypeError, ValueError):
            continue
        if conf_value < 0:
            continue
        words.append(token)
        confs.append(conf_value)

    if not words:
        return "", 0.0
    score = round(sum(confs) / (len(confs) * 100.0), 4)
    return " ".join(words), max(0.0, min(score, 1.0))


def run_ocr(job_id: str, data_root: str) -> dict[str, str | int]:
    job_dir = Path(data_root) / job_id
    keyframes_dir = job_dir / "vision" / "keyframes"
    scenes_csv = job_dir / "vision" / "scenes.csv"
    ocr_jsonl = job_dir / "vision" / "ocr.jsonl"
    ocr_jsonl.parent.mkdir(parents=True, exist_ok=True)

    threshold = float(os.getenv("OCR_EDGE_DENSITY_THRESHOLD", "0.02"))
    ts_map = _load_scene_midpoints(scenes_csv)
    frames = sorted(keyframes_dir.glob("*.jpg")) if keyframes_dir.exists() else []

    rows_written = 0
    with ocr_jsonl.open("w", encoding="utf-8") as handle:
        for frame_path in frames:
            scene_id = _extract_scene_id(frame_path.name)
            ts = ts_map.get(scene_id, 0.0)
            text = ""
            score = 0.0

            image = cv2.imread(str(frame_path))
            if image is not None:
                density = _edge_density(image)
                if density >= threshold:
                    try:
                        text, score = _run_tesseract(image)
                    except pytesseract.TesseractError:
                        text = ""
                        score = 0.0

            row = {
                "ts": ts,
                "scene_id": scene_id,
                "frame": f"vision/keyframes/{frame_path.name}",
                "text": text,
                "score": score,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

    return {
        "ocr_jsonl": str(ocr_jsonl),
        "frames_total": len(frames),
        "rows_written": rows_written,
    }
