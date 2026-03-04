import json
import os
from collections import Counter
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
    """
    Base system prompt. If a custom template exists, it is used as a prefix,
    otherwise we fall back to a sensible default. The concrete task instruction
    "Create a structured educational video brief" is always appended later.
    """
    prompt_path = Path(os.getenv("VIDEO_BRIEF_PROMPT_PATH", DEFAULT_PROMPT_PATH))
    if prompt_path.exists():
        base = prompt_path.read_text(encoding="utf-8").strip()
    else:
        base = (
            "You are an expert educational video analyst.\n"
            "You analyze transcripts, on-screen text (OCR), and chunk-level context "
            "to create high‑quality learning materials."
        )
    return base


def _collect_context(chunks: list[dict[str, Any]], max_chars: int = 12000) -> str:
    """
    Build a compact, chunk-aware context that includes transcript, OCR, and any
    available per-chunk summary fields.
    """
    lines: list[str] = []
    size = 0

    for idx, chunk in enumerate(chunks, start=1):
        t0 = chunk.get("t0")
        t1 = chunk.get("t1")
        transcript = str(chunk.get("transcript", "") or "").strip()
        ocr = str(chunk.get("ocr", "") or "").strip()

        # Prefer explicit summary fields when present (future‑proof for additional steps)
        summary = str(
            chunk.get("summary")
            or chunk.get("vlm")
            or ""
        ).strip()

        parts: list[str] = []
        if transcript:
            parts.append(f"TRANSCRIPT: {transcript}")
        if ocr:
            parts.append(f"OCR: {ocr}")
        if summary:
            parts.append(f"SUMMARY: {summary}")

        if not parts:
            continue

        header = f"CHUNK {idx} [{t0} - {t1}]"
        piece = header + "\n" + "\n".join(parts)

        if size + len(piece) > max_chars:
            break

        lines.append(piece)
        size += len(piece)

    return "\n\n".join(lines)


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


def _timeline_bounds(chunks: list[dict[str, Any]]) -> tuple[float, float]:
    if not chunks:
        return 0.0, 0.0
    starts: list[float] = []
    ends: list[float] = []
    for ch in chunks:
        try:
            starts.append(float(ch.get("t0", 0.0) or 0.0))
            ends.append(float(ch.get("t1", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    if not starts or not ends:
        return 0.0, 0.0
    return min(starts), max(ends)


def _auto_chapters(chunks: list[dict[str, Any]], target: int = 6) -> list[dict[str, Any]]:
    start, end = _timeline_bounds(chunks)
    if end <= start:
        # Fallback to simple numbered chapters without timing
        return [{"title": f"Chapter {i+1}", "t0": 0.0, "t1": 0.0} for i in range(target)]

    duration = max(end - start, 1.0)
    step = duration / float(target)
    chapters: list[dict[str, Any]] = []
    for i in range(target):
        t0 = start + step * i
        t1 = start + step * (i + 1)
        chapters.append(
            {
                "title": f"Chapter {i+1}",
                "t0": round(t0, 3),
                "t1": round(min(t1, end), 3),
            }
        )
    return chapters


def _default_summary_points(chunks: list[dict[str, Any]], target: int = 8) -> list[str]:
    points: list[str] = []
    for ch in chunks:
        text = str(ch.get("transcript", "") or "").strip()
        if not text:
            continue
        snippet = text.split(". ")[0].strip()
        if not snippet:
            continue
        points.append(snippet)
        if len(points) >= target:
            break
    while len(points) < target:
        points.append(f"Key insight {len(points) + 1}.")
    return points[:target]


def _default_key_concepts(chunks: list[dict[str, Any]], target: int = 10) -> list[dict[str, str]]:
    text_parts: list[str] = []
    for ch in chunks:
        text_parts.append(str(ch.get("transcript", "") or ""))
        text_parts.append(str(ch.get("ocr", "") or ""))
    full_text = " ".join(text_parts)[:8000]

    tokens = [
        token.strip(".,!?;:()[]{}\"'")
        for token in full_text.split()
    ]
    counter: Counter[str] = Counter(
        t.lower()
        for t in tokens
        if len(t) >= 4 and not t.isdigit()
    )
    concepts: list[dict[str, str]] = []
    for term, _ in counter.most_common(target):
        concepts.append(
            {
                "term": term.capitalize(),
                "definition": "Important concept in the video related to this term.",
            }
        )
        if len(concepts) >= target:
            break

    while len(concepts) < target:
        idx = len(concepts) + 1
        concepts.append(
            {
                "term": f"Concept {idx}",
                "definition": "Key idea discussed in the video.",
            }
        )
    return concepts[:target]


def _normalize_brief(data: dict[str, Any], job_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Normalize LLM output into the strict video brief structure:
    {
      "title": str,
      "one_liner": str,
      "summary_points": [8 x str],
      "chapters": [6 x {title,str,t0,float,t1,float}],
      "key_concepts": [10 x {term,str,definition,str}],
      "suggested_questions": [str]
    }
    """
    title = str(data.get("title") or f"Video Brief {job_id}").strip()
    one_liner = str(data.get("one_liner") or "").strip()

    # Summary bullets (8)
    raw_summary = data.get("summary_points")
    if isinstance(raw_summary, list):
        summary_points = [str(x).strip() for x in raw_summary if str(x).strip()]
    else:
        summary_points = []
    if len(summary_points) < 8:
        auto_points = _default_summary_points(chunks, target=8)
        # Prefer LLM bullets first, then auto-fill
        needed = 8 - len(summary_points)
        summary_points.extend(auto_points[:needed])
    summary_points = summary_points[:8]

    # Chapters (6) with timings
    raw_chapters = data.get("chapters")
    chapters: list[dict[str, Any]] = []
    if isinstance(raw_chapters, list):
        for ch in raw_chapters:
            if isinstance(ch, dict):
                title_val = str(ch.get("title") or "").strip()
                try:
                    t0 = float(ch.get("t0", 0.0) or 0.0)
                    t1 = float(ch.get("t1", 0.0) or 0.0)
                except (TypeError, ValueError):
                    t0, t1 = 0.0, 0.0
            else:
                title_val = str(ch).strip()
                t0, t1 = 0.0, 0.0
            if not title_val:
                continue
            chapters.append(
                {
                    "title": title_val,
                    "t0": t0,
                    "t1": t1,
                }
            )
    if len(chapters) < 6:
        auto = _auto_chapters(chunks, target=6)
        needed = 6 - len(chapters)
        chapters.extend(auto[:needed])
    chapters = chapters[:6]

    # Key concepts (10) as term/definition objects
    raw_concepts = data.get("key_concepts")
    key_concepts: list[dict[str, str]] = []
    if isinstance(raw_concepts, list):
        for item in raw_concepts:
            term = ""
            definition = ""
            if isinstance(item, dict):
                term = str(item.get("term") or "").strip()
                definition = str(item.get("definition") or "").strip()
            else:
                text = str(item).strip()
                if ":" in text:
                    term, definition = [p.strip() for p in text.split(":", 1)]
                else:
                    term, definition = text, ""
            if term:
                key_concepts.append({"term": term, "definition": definition or "Key concept from the video."})

    if len(key_concepts) < 10:
        auto_concepts = _default_key_concepts(chunks, target=10)
        needed = 10 - len(key_concepts)
        key_concepts.extend(auto_concepts[:needed])
    key_concepts = key_concepts[:10]

    # Suggested questions (no strict count)
    raw_questions = data.get("suggested_questions")
    if isinstance(raw_questions, list):
        suggested_questions = [str(q).strip() for q in raw_questions if str(q).strip()]
    else:
        suggested_questions = []
    if not suggested_questions:
        suggested_questions = [
            "What are the most important ideas to remember from this video?",
            "Which parts of the video should I rewatch to deepen my understanding?",
            "How can I apply the main concepts from this video in practice?",
        ]

    return {
        "title": title,
        "one_liner": one_liner,
        "summary_points": summary_points,
        "chapters": chapters,
        "key_concepts": key_concepts,
        "suggested_questions": suggested_questions,
    }


def _fallback_brief(job_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Deterministic fallback when the LLM output cannot be parsed.
    Produces a brief that already matches the required schema.
    """
    summary_points = _default_summary_points(chunks, target=8)
    chapters = _auto_chapters(chunks, target=6)
    key_concepts = _default_key_concepts(chunks, target=10)

    return {
        "title": f"Video Brief {job_id}",
        "one_liner": "Auto-generated brief from transcript, OCR, and timeline chunks.",
        "summary_points": summary_points,
        "chapters": chapters,
        "key_concepts": key_concepts,
        "suggested_questions": [
            "What are the main topics covered in this video?",
            "Which sections should be reviewed first?",
            "What are the actionable takeaways?",
        ],
    }


def _render_markdown(brief: dict[str, Any]) -> str:
    """
    Render the structured brief into a human-friendly markdown summary.
    """
    lines: list[str] = [f"# {brief['title']}", ""]

    if brief.get("one_liner"):
        lines.append(f"**One‑liner:** {brief['one_liner']}")
        lines.append("")

    # Summary bullets
    summary_points = brief.get("summary_points") or []
    if summary_points:
        lines.append("## Summary")
        for point in summary_points:
            lines.append(f"- {point}")
        lines.append("")

    # Chapters
    chapters = brief.get("chapters") or []
    if chapters:
        lines.append("## Chapters")
        for idx, ch in enumerate(chapters, start=1):
            title = ch.get("title") or f"Chapter {idx}"
            t0 = ch.get("t0")
            t1 = ch.get("t1")
            if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and (t0 or t1):
                lines.append(f"- **{title}** ({t0:.1f}s – {t1:.1f}s)")
            else:
                lines.append(f"- **{title}**")
        lines.append("")

    # Key concepts
    key_concepts = brief.get("key_concepts") or []
    if key_concepts:
        lines.append("## Key Concepts")
        for concept in key_concepts:
            if isinstance(concept, dict):
                term = concept.get("term") or ""
                definition = concept.get("definition") or ""
                if term and definition:
                    lines.append(f"- **{term}**: {definition}")
                elif term:
                    lines.append(f"- **{term}**")
            else:
                lines.append(f"- {concept}")
        lines.append("")

    # Suggested questions
    questions = brief.get("suggested_questions") or []
    if questions:
        lines.append("## Suggested Questions")
        for question in questions:
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
        "Create a structured educational video brief.\n"
        "Use the transcript, OCR text, and chunk-level context below.\n\n"
        "Output a single JSON object with the following structure:\n"
        "{\n"
        '  "title": string,\n'
        '  "one_liner": string,\n'
        '  "summary_points": [8 strings],\n'
        '  "chapters": [\n'
        '    {"title": string, "t0": number, "t1": number},\n'
        "    ... exactly 6 items total ...\n"
        "  ],\n"
        '  "key_concepts": [\n'
        '    {"term": string, "definition": string},\n'
        "    ... exactly 10 items total ...\n"
        "  ],\n"
        '  "suggested_questions": [string]\n'
        "}\n\n"
        "Return ONLY valid JSON. Do not include any explanations.\n\n"
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
        brief = _normalize_brief(generated_json, job_id, chunks)

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

