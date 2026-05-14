"""
MT extension to the QA diagnostic pilot.

Filter FLORES-200 by tagging target-side references with Sonnet 4.6,
keep ~50 items per linguistic phenomenon bucket per language,
translate EN -> ID/SW with 4 models, score with chrF + COMET-22 + gpt-5.4-mini judge.

Single command:
  python run_mt.py                   # full run
  python run_mt.py --per-bucket 30   # smaller buckets for smoke test
  python run_mt.py --skip-comet      # skip COMET stage (no GPU)
  python run_mt.py --reset           # start fresh

Output: data/mt_results.jsonl — one record per item with everything:
  {id, lang, src_en, ref, tags, translations: {model: text}, scores: {model: {chrf, comet, adequacy, fluency}}}

Stages are idempotent: each skips work already present per field.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# ---------- Config ----------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS = DATA_DIR / "mt_results.jsonl"

FLORES_DATASET = "openlanguagedata/flores_plus"
FLORES_LANGS = {"eng": "eng_Latn", "id": "ind_Latn", "sw": "swh_Latn", "fr": "fra_Latn"}
FLORES_SPLIT = "devtest"

LANG_NAME = {"id": "Indonesian", "sw": "Swahili", "fr": "French"}

# Required buckets — sampling continues until each has >= per_bucket items
BUCKETS = {
    "id": ["voice_meN", "voice_di", "complex_NP_yang"],
    "sw": ["noun_class_concord", "passive", "locative_ni"],
    "fr": ["clitic_pronoun", "subjunctive", "complex_NP_relative"],
}
# Bonus buckets — tracked but don't gate sampling
BONUS = {"id": ["reduplication"], "sw": ["applicative"], "fr": ["past_participle_agreement"]}

# (label, openrouter_model_id) — translation models (same lineup as QA pilot)
MODELS = [
    ("gemini_3_pro",    "google/gemini-3.1-pro-preview"),
    ("deepseek_v4_pro", "deepseek/deepseek-v4-pro"),
    ("gemma_4_31b_it",  "google/gemma-4-31b-it"),
    ("llama_31_8b",     "meta-llama/llama-3.1-8b-instruct"),
]
TAGGER = ("claude_sonnet_4_6", "anthropic/claude-sonnet-4.6")     # via OpenRouter
JUDGE  = ("gpt_5_4_mini",      "gpt-5.4-mini")                     # via OpenAI direct
COMET_MODEL = "Unbabel/wmt22-comet-da"

CONCURRENCY = 32
RAND_SEED = 42


# ---------- IO ----------

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, items) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------- Clients ----------

def _openrouter():
    return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")


def _openai_direct():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_openrouter(model_id: str, system: str, user: str, max_tokens: int = 400) -> str:
    resp = _openrouter().chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def call_openai_direct(model_id: str, system: str, user: str, max_tokens: int = 200) -> str:
    # GPT-5.x family uses `max_completion_tokens` instead of `max_tokens`.
    resp = _openai_direct().chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def parse_json_strict(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        raise ValueError(f"no JSON object in: {text[:200]}")
    return json.loads(text[a:b + 1])


# ---------- Prompts ----------

TAG_SYSTEM = "You are a linguistic annotator. Output strict JSON only — no prose, no markdown."

TAG_USER_TMPL = {
    "id": """Annotate the following Indonesian sentence for the linguistic phenomena listed below.

linguistic_phenomena — list any present in the sentence:
   - "voice_meN": active voice prefix meN- with any allomorph (me-, mem-, men-, meng-, meny-, menge-). Examples: meminta, membaca, menulis, mengirim, menyapu.
   - "voice_di": passive voice prefix di- on a verb. Examples: diminta, dibaca, ditulis, ditemukan. Do NOT confuse with the locative preposition "di" written as a separate word (meaning "at/in").
   - "reduplication": full or partial reduplication of a base word, written with a hyphen. Examples: anak-anak (children), jalan-jalan (walk around), kekanak-kanakan (childish). Compound words with hyphens that are NOT reduplications do NOT count.
   - "complex_NP_yang": noun phrase containing a relative clause introduced by `yang` modifying a head noun. Examples: "orang yang datang", "buku yang dibaca". Bare `yang` as a focus marker without a clear noun-head does NOT count.

Return [] if none apply.

Sentence: {text}

Output exactly: {{"linguistic_phenomena": [...]}}""",

    "sw": """Annotate the following Swahili sentence for the linguistic phenomena listed below.

linguistic_phenomena — list any present in the sentence:
   - "noun_class_concord": noun class agreement marker on adjective, verb, or possessive (M-Wa: m-/wa-; Ki-Vi: ki-/vi-; N-: n-; Ji-Ma: ji-/ma-; U-: u-/n-; etc.). Examples: "watoto wadogo" (small children, M-Wa concord), "vitabu vyangu" (my books, Ki-Vi concord).
   - "applicative": verb with applicative extension -i-/-e- adding benefactive or directional argument. Examples: andikia (write to/for), pikia (cook for).
   - "passive": verb with passive extension -w-. Examples: andikwa (be written), pikwa (be cooked). Do NOT confuse with question word "wapi" (where).
   - "locative_ni": noun + locative suffix -ni indicating location. Examples: nyumbani (at home), shuleni (at school), mjini (in town). Bare "ni" as copula does NOT count.

Return [] if none apply.

Sentence: {text}

Output exactly: {{"linguistic_phenomena": [...]}}""",

    "fr": """Annotate the following French sentence for the linguistic phenomena listed below.

linguistic_phenomena — list any present in the sentence:
   - "clitic_pronoun": object/dative/locative pronoun clitic attached pre-verbally. Counts: "le, la, l', les, lui, leur, y, en" and "me, te, se, nous, vous" when functioning as object/dative/reflexive (NOT when these are subject pronouns). Examples: "je le vois" (I see him), "il lui parle" (he speaks to her), "j'y vais" (I go there), "elle s'en va" (she leaves). Do NOT count subject pronouns "je/tu/il/elle/nous/vous/ils/elles" or stressed forms "moi/toi/lui/elle/eux/elles" used after prepositions.
   - "subjunctive": verb in subjunctive mood (present or imperfect). Examples: "qu'il soit" (that he be), "que nous fassions" (that we do), "qu'ils aient" (that they have), "bien qu'elle puisse" (although she can). Triggered by expressions of doubt/wish/necessity/emotion, or conjunctions like "pour que", "avant que", "bien que".
   - "complex_NP_relative": noun phrase containing a relative clause introduced by `qui, que, qu', dont, où, lequel/laquelle/lesquels/lesquelles, duquel/desquels, auquel/auxquels`. Examples: "l'homme qui marche", "le livre que je lis", "la ville où j'habite", "la raison pour laquelle".
   - "past_participle_agreement": past participle showing gender/number agreement with subject (in être passé composé) or with a preceding direct object (in avoir passé composé). Examples: "elle est partie" (subject agreement), "les pommes que j'ai mangées" (preceding DO agreement). Look for participles ending in -e/-es/-s where a base form -é would also be possible.

Return [] if none apply.

Sentence: {text}

Output exactly: {{"linguistic_phenomena": [...]}}""",
}


TRANSLATE_SYSTEM = {
    "id": "Anda adalah penerjemah profesional. Terjemahkan ke bahasa Indonesia dengan akurat dan alami. Berikan hanya hasil terjemahan, tanpa penjelasan tambahan.",
    "sw": "Wewe ni mtafsiri mtaalamu. Tafsiri kwa Kiswahili kwa usahihi na asili. Toa tafsiri tu, bila ufafanuzi wa ziada.",
    "fr": "Vous êtes un traducteur professionnel. Traduisez en français de manière précise et naturelle. Donnez uniquement la traduction, sans explication.",
}

TRANSLATE_USER_TMPL = "Translate the following English text to {target_lang}:\n\n{text}"


JUDGE_SYSTEM = "You are an MT quality evaluator. Output strict JSON only — no prose, no markdown."

JUDGE_USER_TMPL = """Rate this {target_lang} translation against the reference on two 1-5 scales.

ADEQUACY (does the hypothesis preserve the source meaning?):
1 = none / unrelated; 2 = poor / major omissions; 3 = partial; 4 = mostly preserved; 5 = perfect

FLUENCY (does the hypothesis read naturally in {target_lang}?):
1 = unintelligible; 2 = many grammar errors; 3 = some errors but understandable; 4 = mostly fluent; 5 = native-like

Source (English): {src}
Reference {target_lang}: {ref}
Hypothesis {target_lang}: {hyp}

Output exactly: {{"adequacy": <1-5>, "fluency": <1-5>, "note": "<one short sentence>"}}"""


# ---------- FLORES load ----------

def load_flores_aligned(lang_code: str) -> list[dict]:
    """Load EN and target-language splits, align by FLORES id."""
    from datasets import load_dataset
    eng = load_dataset(FLORES_DATASET, FLORES_LANGS["eng"], split=FLORES_SPLIT)
    tgt = load_dataset(FLORES_DATASET, FLORES_LANGS[lang_code], split=FLORES_SPLIT)
    by_id_eng = {r["id"]: r for r in eng}
    items = []
    for r in tgt:
        if r["id"] in by_id_eng:
            items.append({
                "id": f"flores_{lang_code}_{r['id']}",
                "flores_id": r["id"],
                "lang": lang_code,
                "src_en": by_id_eng[r["id"]]["text"].strip(),
                "ref":    r["text"].strip(),
                "domain": r.get("domain"),
                "topic":  r.get("topic"),
            })
    return items


# ---------- Stage 1: filter + tag ----------

import time as _time

def tag_one(item: dict, max_retries: int = 2) -> dict:
    """Tag one item with retry on empty/invalid response (transient OpenRouter flakiness)."""
    user = TAG_USER_TMPL[item["lang"]].format(text=item["ref"])
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = call_openrouter(TAGGER[1], TAG_SYSTEM, user, max_tokens=300)
            if not resp.strip():
                raise ValueError("empty response")
            return parse_json_strict(resp)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < max_retries:
                _time.sleep(1.0 + attempt * 0.5)
    raise last_err if last_err else ValueError("tag failed")


def stage_filter_and_tag(per_bucket: int, batch_size: int = 32) -> list[dict]:
    """Sample FLORES, tag in parallel batches, fill required buckets to per_bucket.

    Saves incrementally after each language so Ctrl+C never loses tag progress.
    """
    all_items_dict = {it["id"]: it for it in read_jsonl(RESULTS)}

    for lang in ["id", "sw", "fr"]:
        bucket_targets = BUCKETS[lang]
        bonus_targets  = BONUS[lang]

        # Bootstrap from cache
        existing_lang = [it for it in all_items_dict.values() if it["lang"] == lang and "tags" in it]
        buckets: dict[str, list[dict]] = defaultdict(list)
        for it in existing_lang:
            phenomena = it["tags"].get("linguistic_phenomena") or ["none"]
            for ph in phenomena:
                buckets[ph].append(it)

        if all(len(buckets[b]) >= per_bucket for b in bucket_targets):
            print(f"{LANG_NAME[lang]}: buckets already full from cache, n={len(existing_lang)}")
            continue

        # Candidates not yet in cache
        flores = load_flores_aligned(lang)
        candidates = [r for r in flores if r["id"] not in all_items_dict]
        random.Random(RAND_SEED).shuffle(candidates)

        print(f"\n{LANG_NAME[lang]}: filling {len(bucket_targets)} buckets to {per_bucket} from {len(candidates)} candidates (batch={batch_size}, parallel)")
        print(f"  Required: {bucket_targets}")
        print(f"  Bonus:    {bonus_targets}")
        print(f"  Existing: " + ", ".join(f"{b}={len(buckets[b])}" for b in bucket_targets))

        idx = 0
        pbar = tqdm(total=len(candidates), desc=f"tag {lang}")
        while idx < len(candidates) and not all(len(buckets[b]) >= per_bucket for b in bucket_targets):
            batch = candidates[idx:idx + batch_size]
            idx += len(batch)

            # Tag the batch concurrently
            with ThreadPoolExecutor(max_workers=batch_size) as ex:
                futs = {ex.submit(tag_one, c): c for c in batch}
                for fut in as_completed(futs):
                    c = futs[fut]
                    try:
                        c["tags"] = fut.result()
                    except Exception as e:
                        print(f"[tag fail] {c['id']}: {e}", file=sys.stderr)

            # Accept items that advance any non-full required bucket or any bonus bucket
            for c in batch:
                if "tags" not in c:
                    continue
                phenomena = c["tags"].get("linguistic_phenomena") or ["none"]
                advances = any(
                    ph in bucket_targets and len(buckets[ph]) < per_bucket
                    for ph in phenomena
                ) or any(ph in bonus_targets for ph in phenomena)
                if advances:
                    all_items_dict[c["id"]] = c
                    for ph in phenomena:
                        if ph in bucket_targets or ph in bonus_targets:
                            buckets[ph].append(c)

            pbar.update(len(batch))
            pbar.set_postfix({b: len(buckets[b]) for b in bucket_targets})
        pbar.close()

        # Report
        print(f"\n{LANG_NAME[lang]} final bucket counts:")
        for b in bucket_targets:
            n = len(buckets[b])
            mark = "✓" if n >= per_bucket else "✗"
            print(f"  {mark} {b}: {n}")
        for b in bonus_targets:
            if buckets[b]:
                print(f"  (bonus) {b}: {len(buckets[b])}")

        # Incremental save after each language
        write_jsonl(RESULTS, list(all_items_dict.values()))
        print(f"  Saved {len(all_items_dict)} items to {RESULTS}")

    return list(all_items_dict.values())


# ---------- Stage 2: translate ----------

def stage_translate(items: list[dict]) -> None:
    by_id = {it["id"]: it for it in items}
    jobs: list[tuple[str, str, str]] = []
    for it in items:
        it.setdefault("translations", {})
        for ml, mid in MODELS:
            if ml not in it["translations"]:
                jobs.append((it["id"], ml, mid))
    if not jobs:
        print("Translate: all done")
        return
    print(f"Translate: {len(jobs)} calls")

    def worker(j):
        iid, ml, mid = j
        it = by_id[iid]
        user = TRANSLATE_USER_TMPL.format(target_lang=LANG_NAME[it["lang"]], text=it["src_en"])
        out = call_openrouter(mid, TRANSLATE_SYSTEM[it["lang"]], user, max_tokens=400)
        return j, out.strip()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(worker, j): j for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            iid, ml, _ = futs[fut]
            try:
                _, text = fut.result()
                by_id[iid]["translations"][ml] = text
            except Exception as e:
                print(f"[translate fail] {ml}/{iid}: {e}", file=sys.stderr)


# ---------- Stage 3a: chrF (sacrebleu, free) ----------

def stage_chrf(items: list[dict]) -> None:
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        print("Install sacrebleu first: pip install sacrebleu", file=sys.stderr)
        return
    chrf = CHRF()
    n = 0
    for it in items:
        it.setdefault("scores", {})
        for ml, _ in MODELS:
            it["scores"].setdefault(ml, {})
            hyp = it.get("translations", {}).get(ml)
            if hyp and "chrf" not in it["scores"][ml]:
                it["scores"][ml]["chrf"] = chrf.sentence_score(hyp, [it["ref"]]).score
                n += 1
    print(f"chrF: scored {n} hypotheses")


# ---------- Stage 3b: COMET-22 ----------

def stage_comet(items: list[dict]) -> None:
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("Install COMET first: pip install unbabel-comet", file=sys.stderr)
        return

    triplets = []
    keys = []  # parallel (item_id, model_label)
    for it in items:
        for ml, _ in MODELS:
            entry = it.get("scores", {}).get(ml, {})
            hyp = it.get("translations", {}).get(ml)
            if hyp and "comet" not in entry:
                triplets.append({"src": it["src_en"], "mt": hyp, "ref": it["ref"]})
                keys.append((it["id"], ml))
    if not triplets:
        print("COMET: all done")
        return

    print(f"COMET: loading {COMET_MODEL}")
    model_path = download_model(COMET_MODEL)
    model = load_from_checkpoint(model_path)
    print(f"COMET: scoring {len(triplets)} triplets")
    out = model.predict(triplets, batch_size=64, gpus=1)

    by_id = {it["id"]: it for it in items}
    for (iid, ml), sc in zip(keys, out.scores):
        by_id[iid].setdefault("scores", {}).setdefault(ml, {})["comet"] = float(sc)
    print(f"COMET: system-level avg = {sum(out.scores) / len(out.scores):.3f}")


# ---------- Stage 3c: gpt-5.4-mini judge (adequacy + fluency) ----------

def judge_one(item: dict, hyp: str) -> dict:
    user = JUDGE_USER_TMPL.format(
        target_lang=LANG_NAME[item["lang"]],
        src=item["src_en"], ref=item["ref"], hyp=hyp,
    )
    return parse_json_strict(call_openai_direct(JUDGE[1], JUDGE_SYSTEM, user, max_tokens=200))


def stage_judge(items: list[dict]) -> None:
    by_id = {it["id"]: it for it in items}
    jobs: list[tuple[str, str]] = []
    for it in items:
        for ml, _ in MODELS:
            entry = it.get("scores", {}).get(ml, {})
            hyp = it.get("translations", {}).get(ml)
            if hyp and "adequacy" not in entry:
                jobs.append((it["id"], ml))
    if not jobs:
        print("Judge: all done")
        return
    print(f"Judge: {len(jobs)} calls (gpt-5.4-mini via OpenAI direct)")

    def worker(j):
        iid, ml = j
        hyp = by_id[iid]["translations"][ml]
        return j, judge_one(by_id[iid], hyp)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(worker, j): j for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            iid, ml = futs[fut]
            try:
                _, judged = fut.result()
                entry = by_id[iid].setdefault("scores", {}).setdefault(ml, {})
                entry["adequacy"]   = judged.get("adequacy")
                entry["fluency"]    = judged.get("fluency")
                entry["judge_note"] = judged.get("note", "")
            except Exception as e:
                print(f"[judge fail] {ml}/{iid}: {e}", file=sys.stderr)


# ---------- Main ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-bucket", type=int, default=50, help="Target items per required bucket")
    p.add_argument("--skip-comet", action="store_true", help="Skip COMET stage (e.g. no GPU)")
    p.add_argument("--reset", action="store_true", help="Delete mt_results.jsonl and start fresh")
    args = p.parse_args()

    if args.reset and RESULTS.exists():
        RESULTS.unlink()
        print(f"Reset: removed {RESULTS}")

    # Quick env sanity check
    missing = [k for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"Missing env vars: {missing}", file=sys.stderr)
        sys.exit(1)

    items = stage_filter_and_tag(args.per_bucket)
    write_jsonl(RESULTS, items)

    stage_translate(items);  write_jsonl(RESULTS, items)
    stage_chrf(items);       write_jsonl(RESULTS, items)
    if not args.skip_comet:
        stage_comet(items);  write_jsonl(RESULTS, items)
    stage_judge(items);      write_jsonl(RESULTS, items)

    by_lang = defaultdict(int)
    for it in items:
        by_lang[it["lang"]] += 1
    print(f"\nDone. {len(items)} items ({dict(by_lang)}) in {RESULTS}")


if __name__ == "__main__":
    main()
