"""
Diagnostic pilot v2: native vs translated extractive QA, Indonesian + Swahili.

One command:
  python run_pipeline.py            # full run
  python run_pipeline.py --limit 5  # smoke test
  python run_pipeline.py --reset    # start fresh

Prerequisite: Stage 0 translation → data/squad_id_gemini.jsonl, data/squad_sw_gemini.jsonl

Single output: data/results.jsonl — one record per item with everything:
  {id, dataset, lang, question, context, gold, tags,
   predictions: {model: {condition: {prediction, verdict, reason, correct}}}}

Datasets (500 items each when data permits; --limit caps each source):
  indoqa      — Indonesian native (IndoQA val, answerable)
  squad_id    — Indonesian translated SQuAD (Gemini jsonl, span_found only)
  tydiqa_sw   — Swahili native (TyDi QA secondary_task, HF `tydiqa`)
  squad_sw    — Swahili translated SQuAD (Gemini jsonl, span_found only)

Conditions: openbook + closedbook only (extractive QA).

Fully resumable: each stage skips work already present per field.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS = DATA_DIR / "results.jsonl"

INDOQA_VAL_URL = "https://drive.google.com/uc?id=1mq_foV72riXb1KVBirJzTFZEe7oa8f4f"
INDOQA_VAL_CACHE = DATA_DIR / "indoqa_val.json"
SQUAD_ID_GEMINI = DATA_DIR / "squad_id_gemini.jsonl"
SQUAD_SW_GEMINI = DATA_DIR / "squad_sw_gemini.jsonl"

N_PER_BENCHMARK = 500
PREPARE_SEED = 42
V2_DATASETS = frozenset({"indoqa", "squad_id", "tydiqa_sw", "squad_sw"})

# (label, openrouter_model_id)
MODELS = [
    ("gemini_3_pro",    "google/gemini-3.1-pro-preview"),
    ("deepseek_v4_pro", "deepseek/deepseek-v4-pro"),
    ("gemma_4_31b_it",  "google/gemma-4-31b-it"),
    ("llama_31_8b",     "meta-llama/llama-3.1-8b-instruct"),
]
JUDGE_MODEL = ("gpt_4o_mini", "openai/gpt-4o-mini")
TAGGER       = ("claude_sonnet_4_6", "anthropic/claude-sonnet-4.6")

CONDITIONS = {
    "indoqa": ["openbook", "closedbook"],
    "squad_id": ["openbook", "closedbook"],
    "tydiqa_sw": ["openbook", "closedbook"],
    "squad_sw": ["openbook", "closedbook"],
}

CONCURRENCY = 64


# ---------- IO ----------

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_jsonl(path: Path, items: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------- OpenRouter ----------

def _client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

def call(model_id: str, system: str, user: str, max_tokens: int = 512) -> str:
    resp = _client().chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=8192,
    )
    return resp.choices[0].message.content or ""

def _parse_json_strict(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        raise ValueError(f"no JSON object in: {text[:200]}")
    return json.loads(text[a:b+1])


# ---------- Prompts ----------

TAG_SYSTEM = "You are a linguistic annotator for reading-comprehension items in the specified language. Output strict JSON only — no prose, no markdown."

TAG_USER_TMPL: dict[str, str] = {
"id": """Annotate the following **Indonesian** RC item along three axes. The CONTEXT is provided as background; for `linguistic_phenomena` annotate **only what appears in the QUESTION sentence itself**.

1) linguistic_phenomena — list of phenomena present **in the QUESTION sentence only**:
   - "voice_meN": active voice prefix meN- with any allomorph (me-, mem-, men-, meng-, meny-, menge-). Examples: meminta, membaca, menulis, mengirim, menyapu.
   - "voice_di": passive voice prefix di-. Examples: diminta, dibaca, ditulis, dikirim, ditemukan. Note: do NOT confuse with the locative preposition "di" (written as separate word, meaning "at/in").
   - "reduplication": full or partial reduplication of a base word, written with a hyphen. Examples: anak-anak (children), jalan-jalan (walk around), kekanak-kanakan (childish). Compound words with hyphens that are NOT reduplications (e.g., "Bandar-Udara") do NOT count.
   - "complex_NP_yang": noun phrase containing a relative clause introduced by `yang` modifying a head noun. Examples: "orang yang datang" (the person who came), "buku yang dibaca" (the book that was read). Bare `yang` as a focus marker without a clear noun-head also does NOT count.
   Return [] if none apply to the question.

2) paraphrase_distance — one of:
   - "literal_match": gold answer appears as a near-exact span in the context, question rephrases the context only superficially
   - "paraphrase": gold answer is in the context but the question is paraphrased
   - "requires_inference": answer requires combining facts, counting, or inferential step

3) domain — one of:
   - "indonesian_history": Indonesian history, independence, kings, presidents (e.g., Soekarno, Hatta, Majapahit, kerajaan)
   - "indonesian_geography": Indonesia as a country (geography, regions, demographics)
   - "general": anything else (science, world history, religion not specific to Indonesia)

Item:
Context: {context}
Question: {question}
Gold answer: {gold}

Output exactly: {{"linguistic_phenomena": [...], "paraphrase_distance": "...", "domain": "..."}}""",

"sw": """Annotate the following **Swahili** RC item along three axes. The CONTEXT is provided as background; for `linguistic_phenomena` annotate **only what appears in the QUESTION sentence itself**.

1) linguistic_phenomena — list of phenomena present **in the QUESTION sentence only**:
   - "noun_class_concord": noun class agreement marker on adjective, verb, or possessive (M-Wa class: m-/wa- prefix; Ki-Vi class: ki-/vi-; N class: n-; Ji-Ma class: ji-/ma-; U class: u-/n-; etc.). Examples: "watoto wadogo" (small children, M-Wa concord), "vitabu vyangu" (my books, Ki-Vi concord).
   - "applicative": verb with applicative extension -i-/-e- adding a benefactive or directional argument. Examples: andikia (write to/for), pikia (cook for).
   - "passive": verb with passive extension -w-. Examples: andikwa (be written), pikwa (be cooked). Do NOT confuse with question word "wapi" (where).
   - "locative_ni": noun + locative suffix -ni indicating location/direction. Examples: nyumbani (at home), shuleni (at school), mjini (in town). Bare "ni" as copula ("ni mtoto", "is a child") does NOT count.
   Return [] if none apply to the question.

2) paraphrase_distance — one of:
   - "literal_match": gold answer appears as a near-exact span in the context, question rephrases the context only superficially
   - "paraphrase": gold answer is in the context but the question is paraphrased
   - "requires_inference": answer requires combining facts, counting, or inferential step

3) domain — one of:
   - "east_african_history": Tanzania/Kenya/Uganda historical figures, events
   - "east_african_geography": Tanzania/Kenya/Uganda/Swahili-coast geography
   - "general": anything else

Item:
Context: {context}
Question: {question}
Gold answer: {gold}

Output exactly: {{"linguistic_phenomena": [...], "paraphrase_distance": "...", "domain": "..."}}""",
}


INFER_SYSTEM = {
    "id": "Anda adalah asisten. Jawab dengan singkat, tanpa penjelasan tambahan.",
    "sw": "Wewe ni msaidizi. Jibu kwa ufupi, bila maelezo ya ziada.",
}

OPENBOOK_TMPL = {
"id": """Berdasarkan bacaan berikut, jawab pertanyaannya.

Bacaan: {context}

Pertanyaan: {question}

Jawaban:""",
"sw": """Kulingana na kifungu kifuatacho, jibu swali.

Kifungu: {context}

Swali: {question}

Jibu:""",
}

CLOSEDBOOK_TMPL = {
"id": """Jawab pertanyaan berikut.

Pertanyaan: {question}

Jawaban:""",
"sw": """Jibu swali lifuatalo.

Swali: {question}

Jibu:""",
}
JUDGE_SYSTEM = "You are evaluating a multilingual QA system. Output strict JSON only."

JUDGE_TMPL = """Compare the predicted answer to the gold answer. The question is in {language_name}. Verdict "correct" if the prediction conveys the same factual content as the gold (allowing paraphrasing, normalization differences, partial-vs-full names, dates in different formats). Otherwise "incorrect".

Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}

Output: {{"verdict": "correct" | "incorrect", "reason": "<one short sentence>"}}"""

LANG_NAME = {"id": "Indonesian", "sw": "Swahili"}


def build_prompt(item: dict, condition: str) -> str:
    lang = item["lang"]
    if condition == "openbook":
        return OPENBOOK_TMPL[lang].format(context=item["context"], question=item["question"])
    if condition == "closedbook":
        return CLOSEDBOOK_TMPL[lang].format(question=item["question"])
    raise ValueError(condition)


# ---------- Stage 1: prepare ----------


def _gemini_squad_items(path: Path, dataset: str, lang: str, n_take: int) -> list[dict]:
    if not path.exists():
        print(f"[prepare] Missing {path}; run translate_squad.py for {dataset}.", file=sys.stderr)
        return []
    rows = [r for r in read_jsonl(path) if r.get("span_found")]
    rows = rows[:n_take]
    out: list[dict] = []
    for r in rows:
        sid = str(r["id"]).replace("/", "_")
        out.append(
            {
                "id": f"{dataset}_{sid}",
                "dataset": dataset,
                "lang": lang,
                "question": r["question"].strip(),
                "context": r["context"].strip(),
                "gold": r["gold"].strip(),
            }
        )
    if len(out) < n_take:
        print(
            f"[prepare] {dataset}: only {len(out)}/{n_take} rows with span_found.",
            file=sys.stderr,
        )
    return out


def _tydiqa_sw_items(n_take: int) -> list[dict]:
    from datasets import concatenate_datasets, load_dataset

    tr = load_dataset("tydiqa", "secondary_task", split="train")
    va = load_dataset("tydiqa", "secondary_task", split="validation")
    sw_tr = tr.filter(lambda ex: ex["id"].startswith("swahili"))
    sw_va = va.filter(lambda ex: ex["id"].startswith("swahili"))
    sw = concatenate_datasets([sw_tr, sw_va])
    sw = sw.shuffle(seed=PREPARE_SEED)
    sw = sw.select(range(min(n_take, len(sw))))
    out: list[dict] = []
    for ex in sw:
        texts = ex["answers"]["text"]
        gold = texts[0].strip() if texts else ""
        oid = str(ex["id"]).replace("/", "_")
        out.append(
            {
                "id": f"tydiqa_sw_{oid}",
                "dataset": "tydiqa_sw",
                "lang": "sw",
                "question": ex["question"].strip(),
                "context": ex["context"].strip(),
                "gold": gold,
            }
        )
    return out


def _ensure_indoqa() -> list[dict]:
    if not INDOQA_VAL_CACHE.exists():
        import gdown
        gdown.download(INDOQA_VAL_URL, str(INDOQA_VAL_CACHE), quiet=True)
    return json.loads(INDOQA_VAL_CACHE.read_text(encoding="utf-8"))


def prepare(limit: int | None = None) -> list[dict]:
    """Build v2 item list (4 benchmarks); merge with existing results.jsonl by id."""
    existing = {it["id"]: it for it in read_jsonl(RESULTS)}
    items: list[dict] = []

    n_take = min(limit, N_PER_BENCHMARK) if limit is not None else N_PER_BENCHMARK

    # ---- Indonesian IndoQA (native, extractive) ----
    indoqa_rows = _ensure_indoqa()
    indoqa_rows = [r for r in indoqa_rows if r.get("answer") and r.get("category") != "UNANSWERABLE"]
    rng_id = random.Random(PREPARE_SEED)
    id_order = list(range(len(indoqa_rows)))
    rng_id.shuffle(id_order)
    id_order = id_order[: min(n_take, len(indoqa_rows))]
    for k, i in enumerate(id_order):
        r = indoqa_rows[i]
        items.append(
            {
                "id": f"indoqa_{k:04d}",
                "dataset": "indoqa",
                "lang": "id",
                "question": r["question"].strip(),
                "context": r["context"].strip(),
                "gold": r["answer"].strip(),
            }
        )
    if len(id_order) < n_take:
        print(
            f"[prepare] indoqa: only {len(id_order)}/{n_take} answerable items in cache.",
            file=sys.stderr,
        )

    items.extend(_gemini_squad_items(SQUAD_ID_GEMINI, "squad_id", "id", n_take))
    items.extend(_tydiqa_sw_items(n_take))
    items.extend(_gemini_squad_items(SQUAD_SW_GEMINI, "squad_sw", "sw", n_take))

    new_ids = {it["id"] for it in items}
    merged = [existing.get(it["id"], it) for it in items]
    extras = [it for it in existing.values() if it["id"] not in new_ids and it.get("dataset") in V2_DATASETS]
    return merged + extras


# ---------- Stage 2: tag ----------

def tag_one(item: dict) -> dict:
    user = TAG_USER_TMPL[item["lang"]].format(context=item["context"], question=item["question"], gold=item["gold"])
    return _parse_json_strict(call(TAGGER[1], TAG_SYSTEM, user, max_tokens=300))

def stage_tag(items: list[dict]) -> None:
    by_id = {it["id"]: it for it in items}
    todo = [it for it in items if "tags" not in it]
    if not todo:
        print("Tag: all done")
        return
    print(f"Tag: {len(todo)} items")
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(tag_one, it): it["id"] for it in todo}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            iid = futs[fut]
            try:
                by_id[iid]["tags"] = fut.result()
            except Exception as e:
                print(f"[tag fail] {iid}: {e}", file=sys.stderr)


# ---------- Stage 3: infer ----------

def stage_infer(items: list[dict]) -> None:
    by_id = {it["id"]: it for it in items}
    jobs: list[tuple[str, str, str, str]] = []
    for it in items:
        it.setdefault("predictions", {})
        for ml, mid in MODELS:
            it["predictions"].setdefault(ml, {})
            for cond in CONDITIONS[it["dataset"]]:
                cur = it["predictions"][ml].get(cond, {})
                if "prediction" not in cur:
                    jobs.append((it["id"], cond, ml, mid))
    if not jobs:
        print("Infer: all done")
        return
    print(f"Infer: {len(jobs)} calls")

    def worker(j):
        iid, cond, ml, mid = j
        prompt = build_prompt(by_id[iid], cond)
        sys_msg = INFER_SYSTEM[by_id[iid]["lang"]]
        return j, call(mid, sys_msg, prompt, max_tokens=200).strip()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(worker, j): j for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            iid, cond, ml, _ = futs[fut]
            try:
                _, pred = fut.result()
                by_id[iid]["predictions"][ml].setdefault(cond, {})["prediction"] = pred
            except Exception as e:
                print(f"[infer fail] {ml}/{cond}/{iid}: {e}", file=sys.stderr)


# ---------- Stage 4: judge ----------

def judge_one(item: dict, prediction: str) -> dict:
    user = JUDGE_TMPL.format(
        language_name=LANG_NAME[item["lang"]],
        question=item["question"], gold=item["gold"], prediction=prediction,
    )
    return _parse_json_strict(call(JUDGE_MODEL[1], JUDGE_SYSTEM, user, max_tokens=256))

def stage_judge(items: list[dict]) -> None:
    by_id = {it["id"]: it for it in items}

    jobs: list[tuple[str, str, str]] = []
    for it in items:
        if it["dataset"] not in V2_DATASETS:
            continue
        for ml, _ in MODELS:
            for cond in ("openbook", "closedbook"):
                entry = it.get("predictions", {}).get(ml, {}).get(cond)
                if entry and "prediction" in entry and "verdict" not in entry:
                    jobs.append((it["id"], cond, ml))
    if not jobs:
        print("Judge: all done")
        return
    print(f"Judge: {len(jobs)} calls")

    def worker(j):
        iid, cond, ml = j
        pred = by_id[iid]["predictions"][ml][cond]["prediction"]
        return j, judge_one(by_id[iid], pred)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(worker, j): j for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            iid, cond, ml = futs[fut]
            try:
                _, judged = fut.result()
                entry = by_id[iid]["predictions"][ml][cond]
                entry["verdict"] = judged["verdict"]
                entry["reason"]  = judged.get("reason", "")
                entry["correct"] = (judged["verdict"] == "correct")
            except Exception as e:
                print(f"[judge fail] {ml}/{cond}/{iid}: {e}", file=sys.stderr)


# ---------- Main ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke test: limit each dataset to N items")
    p.add_argument("--reset", action="store_true",
                   help="Delete results.jsonl and start fresh")
    args = p.parse_args()

    if args.reset and RESULTS.exists():
        RESULTS.unlink()
        print(f"Reset: removed {RESULTS}")

    items = prepare(limit=args.limit)
    write_jsonl(RESULTS, items)
    by_lang = {}
    for it in items:
        by_lang.setdefault((it["lang"], it["dataset"]), 0)
        by_lang[(it["lang"], it["dataset"])] += 1
    print("Prepared:", ", ".join(f"{l}/{d}={n}" for (l, d), n in sorted(by_lang.items())))

    stage_tag(items);    write_jsonl(RESULTS, items)
    stage_infer(items);  write_jsonl(RESULTS, items)
    stage_judge(items);  write_jsonl(RESULTS, items)

    print(f"\nDone. {len(items)} items in {RESULTS}")

if __name__ == "__main__":
    main()
