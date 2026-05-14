#!/usr/bin/env python3
"""
Stage 0: Translate a fixed SQuAD v1.1 train subset to Indonesian and Swahili
via OpenRouter (google/gemini-3-flash-preview), preserving extractive gold spans.

Outputs (JSONL, one object per line):
  {id, src_question_en, src_answer_en, question, context, gold, span_found}

Strategy (per REFACTOR_BRIEF): translate context + question, then ask the model
to copy the shortest verbatim answer span from the translated context using the
English question + answer as reference. Rows with span_found=false are kept for
auditing; downstream prepare stage can filter.

Env:
  OPENROUTER_API_KEY — required
  OPENROUTER_TRANSLATE_MODEL — optional; default google/gemini-3-flash-preview
  TRANSLATE_CONCURRENCY — optional; default 16 (parallel row workers)

Run from repo root `multilingual-evaluation/`:
  python translate_squad.py
  python translate_squad.py --concurrency 32
  python translate_squad.py --limit 3
  python translate_squad.py --lang id
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DATA_DIR = Path("data")
OUT_ID = DATA_DIR / "squad_id_gemini.jsonl"
OUT_SW = DATA_DIR / "squad_sw_gemini.jsonl"

SEED = 42
SUBSET_N = 500
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_CONCURRENCY = 16

LANG_META = {
    "id": {"name": "Indonesian", "path": OUT_ID},
    "sw": {"name": "Swahili", "path": OUT_SW},
}


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object in model output: {text[:300]!r}")
    return json.loads(text[start : end + 1])


def _openrouter_client():
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY for OpenRouter.")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


def _generate(client, model: str, system: str, user: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=8192,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:  # noqa: BLE001 — rate limits / transport
            wait = min(60, 2**attempt)
            if attempt == max_retries - 1:
                raise RuntimeError(f"OpenRouter call failed after retries: {e}") from e
            time.sleep(wait)
    return ""


def load_squad_subset(n: int, seed: int) -> list[dict[str, Any]]:
    ds = load_dataset("squad", split="train")

    def ok(ex: dict) -> bool:
        texts = ex.get("answers") or {}
        tlist = texts.get("text") or []
        return bool(tlist) and bool(str(tlist[0]).strip())

    ds = ds.filter(ok)
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    rows: list[dict[str, Any]] = []
    for ex in ds:
        ans = ex["answers"]["text"][0].strip()
        rows.append(
            {
                "id": ex["id"],
                "src_question_en": ex["question"].strip(),
                "src_context_en": ex["context"].strip(),
                "src_answer_en": ans,
            }
        )
    return rows


def translate_cq(
    client, model: str, target_lang_name: str, context_en: str, question_en: str
) -> tuple[str, str]:
    system = (
        "You are a professional translator. "
        "Follow instructions exactly and output valid JSON only — no markdown, no commentary."
    )
    user = f"""Translate the following reading-comprehension passage and its question into {target_lang_name}.

Rules:
- Preserve meaning; natural {target_lang_name} is required.
- Do not add titles, labels, or explanations.
- The passage and question must stay faithful to the English originals.

Return a single JSON object with exactly these keys:
  "context": <translated passage as a single string>
  "question": <translated question as a single string>

--- ENGLISH PASSAGE ---
{context_en}

--- ENGLISH QUESTION ---
{question_en}
"""
    raw = _generate(client, model, system, user)
    data = _parse_json_object(raw)
    ctx = str(data.get("context", "")).strip()
    q = str(data.get("question", "")).strip()
    if not ctx or not q:
        raise ValueError(f"Translation JSON missing context/question: {raw[:400]!r}")
    return ctx, q


def locate_span(
    client,
    model: str,
    target_lang_name: str,
    translated_context: str,
    translated_question: str,
    src_question_en: str,
    src_answer_en: str,
) -> tuple[str, bool]:
    system = (
        "You locate answer spans for extractive reading comprehension. "
        "Output valid JSON only — no markdown, no commentary."
    )
    user = f"""The passage below is in {target_lang_name}. An English reference question and answer are given.
Find the **shortest contiguous substring** of the {target_lang_name} passage that answers the {target_lang_name} question
with the **same semantic role** as the English answer (i.e. the translated equivalent of that answer).

Hard rules:
- If such a substring exists, set "span_found": true and set "gold" to that substring **copied verbatim** from the passage (exact characters).
- If no exact substring exists (e.g. answer is only implied), set "span_found": false and "gold": "".
- "gold" must either be empty or be an **exact** slice of the passage (no paraphrase).

Return JSON: {{"gold": "...", "span_found": true/false}}

--- {target_lang_name.upper()} PASSAGE ---
{translated_context}

--- {target_lang_name.upper()} QUESTION ---
{translated_question}

--- ENGLISH REFERENCE QUESTION ---
{src_question_en}

--- ENGLISH REFERENCE ANSWER (for alignment) ---
{src_answer_en}
"""
    raw = _generate(client, model, system, user)
    data = _parse_json_object(raw)
    gold = str(data.get("gold", ""))
    span_found = bool(data.get("span_found"))
    if span_found:
        if gold not in translated_context:
            span_found = False
            gold = ""
    else:
        gold = ""
    return gold, span_found


def _translate_one_row(
    client: Any,
    model: str,
    lang_name: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    ctx, q = translate_cq(client, model, lang_name, row["src_context_en"], row["src_question_en"])
    gold, ok = locate_span(
        client,
        model,
        lang_name,
        ctx,
        q,
        row["src_question_en"],
        row["src_answer_en"],
    )
    return {
        "id": row["id"],
        "src_question_en": row["src_question_en"],
        "src_answer_en": row["src_answer_en"],
        "question": q,
        "context": ctx,
        "gold": gold,
        "span_found": ok,
    }


def process_language(
    client,
    model: str,
    lang_code: str,
    subset: list[dict[str, Any]],
    out_path: Path,
    concurrency: int,
) -> None:
    meta = LANG_META[lang_code]
    name = meta["name"]
    items = subset
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(items)
    if concurrency <= 1:
        written_list = [
            _translate_one_row(client, model, name, row)
            for row in tqdm(items, desc=f"translate_{lang_code}")
        ]
    else:
        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_i = {
                pool.submit(_translate_one_row, client, model, name, row): i
                for i, row in enumerate(items)
            }
            for fut in tqdm(
                as_completed(future_to_i),
                total=n,
                desc=f"translate_{lang_code}",
            ):
                i = future_to_i[fut]
                results[i] = fut.result()
        written_list = [results[i] for i in range(n)]

    with out_path.open("w", encoding="utf-8") as f:
        for obj in written_list:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    n_ok = sum(1 for x in written_list if x["span_found"])
    print(f"{name}: {n_ok}/{len(written_list)} span_found ({100 * n_ok / max(1, len(written_list)):.1f}%) -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0: SQuAD -> id/sw via OpenRouter")
    parser.add_argument("--limit", type=int, default=None, help="Max items (for smoke tests)")
    parser.add_argument(
        "--lang",
        choices=("id", "sw", "both"),
        default="both",
        help="Which language file(s) to produce",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="RNG seed for SQuAD subset")
    parser.add_argument("--n", type=int, default=SUBSET_N, help="Subset size from SQuAD train")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_TRANSLATE_MODEL", DEFAULT_MODEL),
        help="OpenRouter model id (default: google/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("TRANSLATE_CONCURRENCY", DEFAULT_CONCURRENCY)),
        help=f"Parallel workers (default {DEFAULT_CONCURRENCY}, or TRANSLATE_CONCURRENCY). Use 1 for sequential.",
    )
    args = parser.parse_args()

    subset = load_squad_subset(args.n, args.seed)
    if args.limit is not None:
        subset = subset[: args.limit]

    print(f"SQuAD subset: {len(subset)} examples (seed={args.seed}, n={args.n})")
    client = _openrouter_client()
    model = args.model
    print(f"OpenRouter model: {model}")
    print(f"Concurrency: {args.concurrency}")

    if args.lang in ("id", "both"):
        process_language(client, model, "id", subset, OUT_ID, args.concurrency)
    if args.lang in ("sw", "both"):
        process_language(client, model, "sw", subset, OUT_SW, args.concurrency)


if __name__ == "__main__":
    main()
