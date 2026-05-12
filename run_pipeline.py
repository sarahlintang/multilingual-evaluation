"""
Diagnostic pilot: native vs translated, Urdu + Indonesian.

One command:
  python run_pipeline.py            # full run
  python run_pipeline.py --limit 5  # smoke test
  python run_pipeline.py --reset    # start fresh

Single output: data/results.jsonl — one record per item with everything:
  {id, dataset, lang, question, context, gold, [choices], tags,
   predictions: {model: {condition: {prediction, verdict, reason, correct}}}}

Datasets:
  Urdu native:        UQuAD (extractive QA, 139 items)         openbook + closedbook
  Urdu translated:    Belebele urd_Arab (MCQ)                  mcq
  Indonesian native:  IndoQA validation (extractive QA, 150)   openbook + closedbook
  Indonesian transl.: Belebele ind_Latn (MCQ)                  mcq

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

UQUAD_RAW = Path("/Users/sarahlintang/Documents/evaluation_research/multilingual-evaluation/urdu_uquad.jsonl")
INDOQA_VAL_URL = "https://drive.google.com/uc?id=1mq_foV72riXb1KVBirJzTFZEe7oa8f4f"
INDOQA_VAL_CACHE = DATA_DIR / "indoqa_val.json"

# (label, openrouter_model_id)
MODELS = [
    ("claude_opus_4_7", "anthropic/claude-opus-4.7"),
    ("qwen_25_7b",      "qwen/qwen-2.5-7b-instruct"),
    ("llama_31_8b",     "meta-llama/llama-3.1-8b-instruct"),
]
JUDGE_MODEL = ("gpt_4o_mini", "openai/gpt-4o-mini")
TAGGER       = ("deepseek_v4_pro", "deepseek/deepseek-v4-pro")

CONDITIONS = {"uquad": ["openbook", "closedbook"],
              "indoqa": ["openbook", "closedbook"],
              "belebele": ["mcq"]}

BELEBELE_CONFIG = {"ur": "urd_Arab", "id": "ind_Latn"}

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
"ur": """Annotate the following **Urdu** RC item along three axes. The CONTEXT is provided as background; for `linguistic_phenomena` annotate **only what appears in the QUESTION sentence itself**.

1) linguistic_phenomena — list of phenomena present **in the QUESTION sentence only**:
   - "ergative_ne": ergative case marker نے on the SUBJECT of a past-tense transitive verb. INCLUDES interrogative subjects: "کس نے" (kis ne, "who-ERG"), "کن لوگوں نے", etc.
   - "izafat": izafat / ezāfe construction (linker -e- between noun and modifier, e.g. خلافتِ راشدہ, دارُالحکومت). Linker is often a kasra ـِ but may also be unwritten.
   - "light_verb": compound / light verb construction (noun or adjective + کرنا/ہونا/لینا/دینا/پانا, e.g. قتل کرنا, پیدا ہونا, وفات پانا)
   - "complex_NP": noun phrase containing EITHER (a) a finite relative clause introduced by جو/جس/جن/کہ, OR (b) a non-finite participial relative such as V-stem + والے/والا/والی/والوں. Bare adjectival modifiers do NOT count.
   Return [] if none apply to the question.

2) paraphrase_distance — one of:
   - "literal_match": gold answer appears as a near-exact span in the context, question rephrases the context only superficially
   - "paraphrase": gold answer is in the context but the question is paraphrased
   - "requires_inference": answer requires combining facts, counting, or inferential step

3) domain — one of:
   - "islamic_history": Islamic religious figures, caliphs, early Islamic events
   - "pakistan_geography": Pakistan as a country (geography, borders, demographics)
   - "general": anything else

Item:
Context: {context}
Question: {question}
Gold answer: {gold}

Output exactly: {{"linguistic_phenomena": [...], "paraphrase_distance": "...", "domain": "..."}}""",

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
}


INFER_SYSTEM = {
    "ur": "آپ ایک معاون ہیں۔ صرف جواب لکھیں، اضافی وضاحت کے بغیر۔",
    "id": "Anda adalah asisten. Jawab dengan singkat, tanpa penjelasan tambahan.",
}

OPENBOOK_TMPL = {
"ur": """درج ذیل اقتباس کی بنیاد پر سوال کا جواب دیں۔

اقتباس: {context}

سوال: {question}

جواب:""",
"id": """Berdasarkan bacaan berikut, jawab pertanyaannya.

Bacaan: {context}

Pertanyaan: {question}

Jawaban:""",
}

CLOSEDBOOK_TMPL = {
"ur": """درج ذیل سوال کا جواب دیں۔

سوال: {question}

جواب:""",
"id": """Jawab pertanyaan berikut.

Pertanyaan: {question}

Jawaban:""",
}

BELEBELE_TMPL = {
"ur": """درج ذیل اقتباس کو پڑھیں اور سوال کا درست جواب منتخب کریں۔ صرف اختیار کا نمبر (1، 2، 3، یا 4) لکھیں۔

اقتباس: {context}

سوال: {question}

اختیارات:
1. {c1}
2. {c2}
3. {c3}
4. {c4}

جواب:""",
"id": """Baca bacaan berikut dan pilih jawaban yang benar. Tulis hanya nomor pilihan (1, 2, 3, atau 4).

Bacaan: {context}

Pertanyaan: {question}

Pilihan:
1. {c1}
2. {c2}
3. {c3}
4. {c4}

Jawaban:""",
}


JUDGE_SYSTEM = "You are evaluating a multilingual QA system. Output strict JSON only."

JUDGE_TMPL = """Compare the predicted answer to the gold answer. The question is in {language_name}. Verdict "correct" if the prediction conveys the same factual content as the gold (allowing paraphrasing, normalization differences, partial-vs-full names, dates in different formats). Otherwise "incorrect".

Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}

Output: {{"verdict": "correct" | "incorrect", "reason": "<one short sentence>"}}"""

LANG_NAME = {"ur": "Urdu", "id": "Indonesian"}


def build_prompt(item: dict, condition: str) -> str:
    lang = item["lang"]
    if condition == "openbook":
        return OPENBOOK_TMPL[lang].format(context=item["context"], question=item["question"])
    if condition == "closedbook":
        return CLOSEDBOOK_TMPL[lang].format(question=item["question"])
    if condition == "mcq":
        ch = item["choices"]
        return BELEBELE_TMPL[lang].format(
            context=item["context"], question=item["question"],
            c1=ch["1"], c2=ch["2"], c3=ch["3"], c4=ch["4"],
        )
    raise ValueError(condition)


# ---------- Stage 1: prepare ----------

def _ensure_indoqa() -> list[dict]:
    if not INDOQA_VAL_CACHE.exists():
        import gdown
        gdown.download(INDOQA_VAL_URL, str(INDOQA_VAL_CACHE), quiet=True)
    return json.loads(INDOQA_VAL_CACHE.read_text(encoding="utf-8"))


def prepare(limit: int | None = None) -> list[dict]:
    """Build complete item list (Urdu + Indonesian, native + translated), merging with existing results."""
    existing = {it["id"]: it for it in read_jsonl(RESULTS)}
    items: list[dict] = []

    # ---- Urdu UQuAD (native, extractive) ----
    if UQUAD_RAW.exists():
        raw = read_jsonl(UQUAD_RAW)
        n = limit if limit is not None else len(raw)
        for i, r in enumerate(raw[:n]):
            items.append({
                "id": f"uquad_{i:04d}", "dataset": "uquad", "lang": "ur",
                "question": r["question_urdu"].strip(),
                "context":  r["context_urdu"].strip(),
                "gold":     r["answer_urdu"].strip(),
            })

    from datasets import load_dataset
    rng = random.Random(0)

    # ---- Urdu Belebele (translated, MCQ) ----
    ds_ur = load_dataset("facebook/belebele", BELEBELE_CONFIG["ur"], split="test")
    n_bel = limit if limit is not None else 139
    for k, i in enumerate(rng.sample(range(len(ds_ur)), min(n_bel, len(ds_ur)))):
        r = ds_ur[i]
        items.append({
            "id": f"belebele_ur_{k:04d}", "dataset": "belebele", "lang": "ur",
            "question": r["question"].strip(),
            "context":  r["flores_passage"].strip(),
            "gold":     str(r["correct_answer_num"]).strip(),
            "choices":  {str(j+1): r[f"mc_answer{j+1}"].strip() for j in range(4)},
        })

    # ---- Indonesian IndoQA (native, extractive) ----
    indoqa_rows = _ensure_indoqa()
    # Filter out unanswerable items (answer is None / category UNANSWERABLE)
    indoqa_rows = [r for r in indoqa_rows if r.get("answer") and r.get("category") != "UNANSWERABLE"]
    n_id = limit if limit is not None else 150
    rng_id = random.Random(1)
    idxs = rng_id.sample(range(len(indoqa_rows)), min(n_id, len(indoqa_rows)))
    for k, i in enumerate(idxs):
        r = indoqa_rows[i]
        items.append({
            "id": f"indoqa_{k:04d}", "dataset": "indoqa", "lang": "id",
            "question": r["question"].strip(),
            "context":  r["context"].strip(),
            "gold":     r["answer"].strip(),
        })

    # ---- Indonesian Belebele (translated, MCQ) ----
    ds_id = load_dataset("facebook/belebele", BELEBELE_CONFIG["id"], split="test")
    rng_bel_id = random.Random(2)
    n_bel_id = limit if limit is not None else 150
    for k, i in enumerate(rng_bel_id.sample(range(len(ds_id)), min(n_bel_id, len(ds_id)))):
        r = ds_id[i]
        items.append({
            "id": f"belebele_id_{k:04d}", "dataset": "belebele", "lang": "id",
            "question": r["question"].strip(),
            "context":  r["flores_passage"].strip(),
            "gold":     str(r["correct_answer_num"]).strip(),
            "choices":  {str(j+1): r[f"mc_answer{j+1}"].strip() for j in range(4)},
        })

    # Merge: prefer existing items (which have tags / predictions / verdicts).
    # Also preserve any existing items not generated this round (e.g., when --limit is set).
    new_ids = {it["id"] for it in items}
    merged = [existing.get(it["id"], it) for it in items]
    extras = [it for it in existing.values() if it["id"] not in new_ids]
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

    # Belebele MCQ: exact-match in-process
    for it in items:
        if it["dataset"] != "belebele":
            continue
        for ml, _ in MODELS:
            entry = it.get("predictions", {}).get(ml, {}).get("mcq")
            if entry and "prediction" in entry and "correct" not in entry:
                chosen = next((c for c in entry["prediction"] if c in "1234"), "")
                entry["chosen"] = chosen
                entry["correct"] = (chosen == it["gold"])

    # Extractive QA (UQuAD + IndoQA): LLM judge
    jobs: list[tuple[str, str, str]] = []
    for it in items:
        if it["dataset"] not in ("uquad", "indoqa"):
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
