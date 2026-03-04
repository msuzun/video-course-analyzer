import json
import os
from pathlib import Path
from typing import Any

from transformers import pipeline

DEFAULT_PROMPT_PATH = "/shared/models/prompts/video_brief.txt"


def _read_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    if not chunks_path.exists():
        raise RuntimeError(f"writer_llm_failed: missing_chunks_file: {chunks_path}")
    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                chunks.append(row)
    if not chunks:
        raise RuntimeError("writer_llm_failed: no_chunks_to_summarize")
    return chunks


def _load_prompt_template() -> str:
    prompt_path = Path(os.getenv("VIDEO_BRIEF_PROMPT_PATH", DEFAULT_PROMPT_PATH))
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return (
        "You are an expert course analyst. Generate a JSON object with keys: "
        "title, one_liner, chapters, key_concepts, suggested_questions."
    )


def _collect_context(chunks: list[dict[str, Any]], max_chars: int = 12000) -> str:
    lines: list[str] = []
    size = 0
    for chunk in chunks:
        piece = f"[{chunk.get('t0')} - {chunk.get('t1')}] {chunk.get('transcript', '')} {chunk.get('ocr', '')}".strip()
        if not piece:
            continue
        if size + len(piece) > max_chars:
            break
        lines.append(piece)
        size += len(piece)
    return "\n".join(lines)


def _extract_json_block(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    raw = text[start : end + 1]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _normalize_brief(data: dict[str, Any], fallback_title: str) -> dict[str, Any]:
    title = str(data.get("title") or fallback_title).strip()
    one_liner = str(data.get("one_liner") or "").strip()

    chapters_raw = data.get("chapters")
    chapters = chapters_raw if isinstance(chapters_raw, list) else []
    chapters = [str(ch).strip() for ch in chapters if str(ch).strip()]

    concepts_raw = data.get("key_concepts")
    key_concepts = concepts_raw if isinstance(concepts_raw, list) else []
    key_concepts = [str(c).strip() for c in key_concepts if str(c).strip()]

    questions_raw = data.get("suggested_questions")
    suggested_questions = questions_raw if isinstance(questions_raw, list) else []
    suggested_questions = [str(q).strip() for q in suggested_questions if str(q).strip()]

    return {
        "title": title,
        "one_liner": one_liner,
        "chapters": chapters,
        "key_concepts": key_concepts,
        "suggested_questions": suggested_questions,
    }


def _fallback_brief(job_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    preview = []
    for chunk in chunks[:5]:
        text = str(chunk.get("transcript", "")).strip()
        if text:
            preview.append(text[:180])
    return {
        "title": f"Video Brief {job_id}",
        "one_liner": "Auto-generated brief from transcript and OCR chunks.",
        "chapters": [f"Segment {idx + 1}" for idx in range(min(5, len(chunks)))],
        "key_concepts": preview[:8],
        "suggested_questions": [
            "What are the main topics covered in this video?",
            "Which sections should be reviewed first?",
            "What are the actionable takeaways?",
        ],
    }


def _render_markdown(brief: dict[str, Any]) -> str:
    lines: list[str] = [f"# {brief['title']}", "", brief["one_liner"], ""]
    lines.append("## Chapters")
    for chapter in brief["chapters"]:
        lines.append(f"- {chapter}")
    lines.append("")
    lines.append("## Key Concepts")
    for concept in brief["key_concepts"]:
        lines.append(f"- {concept}")
    lines.append("")
    lines.append("## Suggested Questions")
    for question in brief["suggested_questions"]:
        lines.append(f"- {question}")
    lines.append("")
    return "\n".join(lines)


def run_writer_llm(job_id: str, data_root: str) -> dict[str, str | int]:
    job_dir = Path(data_root) / job_id
    chunks = _read_chunks(job_dir / "rag" / "chunks.jsonl")

    model_name = os.getenv("WRITER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    prompt_template = _load_prompt_template()
    context = _collect_context(chunks)
    prompt = (
        f"{prompt_template}\n\n"
        "Return ONLY valid JSON.\n\n"
        f"JOB_ID: {job_id}\n"
        f"CONTEXT:\n{context}\n"
    )

    generated_json: dict[str, Any] | None = None
    try:
        generator = pipeline("text-generation", model=model_name, device_map="auto")
        output = generator(prompt, max_new_tokens=700, do_sample=False, temperature=0.1)
        text = output[0]["generated_text"]
        generated_json = _extract_json_block(text)
    except Exception:
        generated_json = None

    if generated_json is None:
        brief = _fallback_brief(job_id, chunks)
    else:
        brief = _normalize_brief(generated_json, fallback_title=f"Video Brief {job_id}")

    outputs_dir = job_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    json_path = outputs_dir / "video_brief.json"
    md_path = outputs_dir / "video_brief.md"
    json_path.write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(brief), encoding="utf-8")

    return {
        "job_id": job_id,
        "model": model_name,
        "video_brief_json": str(json_path),
        "video_brief_md": str(md_path),
        "chunks_used": len(chunks),
    }

