"""
Extract Swahili `has_passive` items for manual inspection.

Produces data/passive_inspection.jsonl_to_json with two arrays:
  - native_wrong: items from tydiqa_sw (native SW) tagged passive,
                  including per-model openbook predictions/verdicts and
                  a `wrong_models` list.
  - translated_correct: items from squad_sw (translated SW) tagged passive,
                        including per-model openbook predictions/verdicts and
                        a `correct_models` list.

Each item carries question / context / gold / tags so you can eyeball them.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("data/results.jsonl")
OUT = Path("data/passive_inspection.json")

MODELS = ["gemini_3_pro", "deepseek_v4_pro", "gemma_4_31b_it", "llama_31_8b"]


def load_results() -> list[dict]:
    return [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]


def slim_item(item: dict) -> dict:
    openbook = {}
    wrong_models, correct_models = [], []
    preds = item.get("predictions", {})
    for m in MODELS:
        ob = preds.get(m, {}).get("openbook", {})
        openbook[m] = {
            "prediction": ob.get("prediction"),
            "verdict": ob.get("verdict"),
            "reason": ob.get("reason"),
            "correct": ob.get("correct"),
        }
        if ob.get("correct") is True:
            correct_models.append(m)
        elif ob.get("correct") is False:
            wrong_models.append(m)
    return {
        "id": item["id"],
        "dataset": item["dataset"],
        "lang": item["lang"],
        "question": item.get("question"),
        "context": item.get("context"),
        "gold": item.get("gold"),
        "tags": item.get("tags", {}),
        "openbook": openbook,
        "wrong_models": wrong_models,
        "correct_models": correct_models,
        "n_wrong": len(wrong_models),
        "n_correct": len(correct_models),
    }


def main() -> None:
    items = load_results()

    native_wrong: list[dict] = []
    translated_correct: list[dict] = []

    for it in items:
        ds = it.get("dataset")
        phenomena = it.get("tags", {}).get("linguistic_phenomena", []) or []
        if "passive" not in phenomena:
            continue

        slim = slim_item(it)

        if ds == "tydiqa_sw":
            # Keep items where at least one model got the openbook answer wrong.
            if slim["n_wrong"] > 0:
                native_wrong.append(slim)
        elif ds == "squad_sw":
            # Keep items where at least one model got the openbook answer correct.
            if slim["n_correct"] > 0:
                translated_correct.append(slim)

    payload = {
        "description": (
            "Swahili passive items for manual inspection. "
            "`native_wrong` = TyDi QA (native SW) passive items where at least one model "
            "answered incorrectly in open-book. "
            "`translated_correct` = SQuAD-SW (translated SW) passive items where at least "
            "one model answered correctly in open-book. "
            "Each item lists per-model open-book predictions plus `wrong_models` / `correct_models`."
        ),
        "models": MODELS,
        "counts": {
            "native_wrong": len(native_wrong),
            "translated_correct": len(translated_correct),
        },
        "native_wrong": native_wrong,
        "translated_correct": translated_correct,
    }

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}  (native_wrong={len(native_wrong)}, translated_correct={len(translated_correct)})")


if __name__ == "__main__":
    main()
