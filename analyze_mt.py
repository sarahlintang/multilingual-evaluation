"""
Analysis + plots for MT extension pilot.

Reads:
  data/mt_results.jsonl       — MT translations + scores
  data/results.jsonl          — QA results (for cross-task correlation)

Outputs to outputs/mt/:
  mt_summary.txt              — system-level + per-phenomenon tables
  01_system_level.png         — chrF / COMET / adequacy / fluency per (lang, model)
  02_per_tag_chrf.png         — chrF per (lang, phenomenon, model)
  03_per_tag_comet.png
  04_per_tag_adequacy.png
  05_per_tag_fluency.png
  06_cross_task.png           — QA accuracy vs MT chrF per phenomenon
  07_metric_agreement.png     — pairwise Pearson correlation across 4 metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MT_RESULTS = Path("data/mt_results.jsonl")
QA_RESULTS = Path("data/results.jsonl")
OUT_DIR = Path("outputs/mt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gemini_3_pro", "deepseek_v4_pro", "gemma_4_31b_it", "llama_31_8b"]
MODEL_DISPLAY = {
    "gemini_3_pro":    "Gemini 3.1 Pro",
    "deepseek_v4_pro": "DeepSeek V4 Pro",
    "gemma_4_31b_it":  "Gemma 4 31B",
    "llama_31_8b":     "Llama 3.1 8B",
}
MODEL_COLOR = {
    "gemini_3_pro":    "#4285F4",
    "deepseek_v4_pro": "#E5793A",
    "gemma_4_31b_it":  "#34A853",
    "llama_31_8b":     "#9C27B0",
}
LANGS = ["id", "sw"]
LANG_NAME = {"id": "Indonesian", "sw": "Swahili"}
METRICS = ["chrf", "comet", "adequacy", "fluency"]

# Per-language required phenomena (must match run_mt.py BUCKETS)
PHENOMENA = {
    "id": ["voice_meN", "voice_di", "complex_NP_yang"],
    "sw": ["noun_class_concord", "passive", "locative_ni"],
}
QA_NATIVE_DATASET = {"id": "indoqa", "sw": "tydiqa_sw"}


# ---------- Load ----------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flatten_mt(items: list[dict]) -> pd.DataFrame:
    """One row per (item, model) with all 4 metrics + phenomenon list."""
    rows = []
    for it in items:
        for ml in MODELS:
            entry = it.get("scores", {}).get(ml, {})
            rows.append({
                "id":        it["id"],
                "lang":      it["lang"],
                "model":     ml,
                "phenomena": it.get("tags", {}).get("linguistic_phenomena") or ["none"],
                "chrf":      entry.get("chrf"),
                "comet":     entry.get("comet"),
                "adequacy":  entry.get("adequacy"),
                "fluency":   entry.get("fluency"),
            })
    return pd.DataFrame(rows)


def has_phenomenon(phenomena_list, ph):
    return isinstance(phenomena_list, list) and ph in phenomena_list


# ---------- Tables ----------

def system_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang in LANGS:
        for ml in MODELS:
            sub = df[(df["lang"] == lang) & (df["model"] == ml)]
            row = {"lang": lang, "model": ml, "n": len(sub)}
            for m in METRICS:
                row[m] = sub[m].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def per_tag(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (lang, phenomenon, model): with-X mean, without-X mean, delta."""
    rows = []
    for lang in LANGS:
        for ph in PHENOMENA[lang]:
            for ml in MODELS:
                base = df[(df["lang"] == lang) & (df["model"] == ml)]
                sub_with    = base[base["phenomena"].apply(lambda xs: has_phenomenon(xs, ph))]
                sub_without = base[~base["phenomena"].apply(lambda xs: has_phenomenon(xs, ph))]
                with_mean    = sub_with[metric].mean()
                without_mean = sub_without[metric].mean()
                rows.append({
                    "lang":           lang,
                    "phenomenon":     ph,
                    "model":          ml,
                    "n_with":         len(sub_with),
                    "n_without":      len(sub_without),
                    f"{metric}_with":    with_mean,
                    f"{metric}_without": without_mean,
                    f"{metric}_delta":   with_mean - without_mean,
                })
    return pd.DataFrame(rows)


# ---------- Plots ----------

def plot_system_level(df: pd.DataFrame) -> None:
    summary = system_level(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, lang in zip(axes, LANGS):
        sub = summary[summary["lang"] == lang].set_index("model").reindex(MODELS)
        x = np.arange(len(MODELS))
        w = 0.2
        ax.bar(x - 1.5 * w, sub["chrf"],          w, label="chrF (0-100)")
        ax.bar(x - 0.5 * w, sub["comet"]    * 100, w, label="COMET (×100)")
        ax.bar(x + 0.5 * w, sub["adequacy"] * 20,  w, label="adequacy (×20)")
        ax.bar(x + 1.5 * w, sub["fluency"]  * 20,  w, label="fluency (×20)")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], rotation=15, fontsize=9)
        ax.set_title(f"{LANG_NAME[lang]} — system-level MT quality (n={int(sub['n'].iloc[0])})")
        ax.set_ylabel("Score (rescaled to ~100)")
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle("MT system-level scores across 4 metrics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_system_level.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_tag(df: pd.DataFrame, metric: str, fname: str) -> None:
    """Per (lang, phenomenon): with-X bars colored by model, with delta vs without-X annotated."""
    breakdown = per_tag(df, metric)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, lang in zip(axes, LANGS):
        phenomena = PHENOMENA[lang]
        x = np.arange(len(phenomena))
        w = 0.2

        for i, ml in enumerate(MODELS):
            with_scores = []
            deltas = []
            ns = []
            for ph in phenomena:
                row = breakdown[
                    (breakdown["lang"] == lang) & (breakdown["phenomenon"] == ph) & (breakdown["model"] == ml)
                ].iloc[0]
                with_scores.append(row[f"{metric}_with"])
                deltas.append(row[f"{metric}_delta"])
                ns.append(row["n_with"])
            bars = ax.bar(x + (i - 1.5) * w, with_scores, w,
                          label=MODEL_DISPLAY[ml], color=MODEL_COLOR[ml])
            # Delta annotation above each bar
            for bar, delta in zip(bars, deltas):
                if not np.isnan(delta):
                    color = "#C0392B" if delta < -3 else ("#27AE60" if delta > 3 else "#666666")
                    ax.annotate(f"{delta:+.1f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                xytext=(0, 1), textcoords="offset points",
                                ha="center", fontsize=6.5, color=color, fontweight="bold")

        # n annotation under each phenomenon group
        n_per_ph = breakdown[(breakdown["lang"] == lang) & (breakdown["model"] == MODELS[0])].set_index("phenomenon")
        for i_ph, ph in enumerate(phenomena):
            n = n_per_ph.loc[ph, "n_with"]
            ax.text(i_ph, -0.05 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0,
                    f"n={int(n)}", ha="center", fontsize=7, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(phenomena, rotation=10, fontsize=9)
        ax.set_title(f"{LANG_NAME[lang]} — avg {metric} on items WITH phenomenon\n(annotation: Δ vs items WITHOUT)")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle(f"MT {metric} per linguistic phenomenon (with-X score and Δ vs without-X)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_task(df_mt: pd.DataFrame) -> None:
    """QA open-book accuracy vs MT chrF per (model, phenomenon) on native QA benchmarks."""
    qa_items = load_jsonl(QA_RESULTS)
    if not qa_items:
        print("Skipping cross-task: QA results not found")
        return

    rows = []
    for lang in LANGS:
        for ph in PHENOMENA[lang]:
            for ml in MODELS:
                # QA accuracy on items WITH phenomenon, openbook, native benchmark
                qa_correct = qa_total = 0
                for it in qa_items:
                    if it.get("lang") != lang or it.get("dataset") != QA_NATIVE_DATASET[lang]:
                        continue
                    phenomena = it.get("tags", {}).get("linguistic_phenomena", []) or []
                    if ph not in phenomena:
                        continue
                    pred = it.get("predictions", {}).get(ml, {}).get("openbook", {})
                    if "correct" not in pred:
                        continue
                    qa_total += 1
                    if pred["correct"]:
                        qa_correct += 1

                # MT chrF on items WITH phenomenon
                mt_sub = df_mt[(df_mt["lang"] == lang) & (df_mt["model"] == ml)
                               & (df_mt["phenomena"].apply(lambda xs: has_phenomenon(xs, ph)))]
                mt_chrf = mt_sub["chrf"].mean()

                if qa_total >= 5 and not np.isnan(mt_chrf):
                    rows.append({
                        "lang": lang, "phenomenon": ph, "model": ml,
                        "qa_acc": 100.0 * qa_correct / qa_total,
                        "mt_chrf": mt_chrf,
                        "n_qa": qa_total, "n_mt": len(mt_sub),
                    })

    if not rows:
        print("Cross-task: no eligible overlap")
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {"id": "o", "sw": "^"}
    for ml in MODELS:
        for lang in LANGS:
            sub = df[(df["model"] == ml) & (df["lang"] == lang)]
            if sub.empty:
                continue
            ax.scatter(sub["qa_acc"], sub["mt_chrf"],
                       c=MODEL_COLOR[ml], marker=markers[lang], s=140, alpha=0.75,
                       edgecolors="black", linewidth=0.6,
                       label=f"{MODEL_DISPLAY[ml]} ({lang})")
            for _, row in sub.iterrows():
                ax.annotate(row["phenomenon"], (row["qa_acc"], row["mt_chrf"]),
                            xytext=(6, 5), textcoords="offset points", fontsize=7)

    # Pearson r
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(df["qa_acc"], df["mt_chrf"])
        title = f"Cross-task consistency: QA open-book accuracy vs MT chrF per phenomenon\nPearson r = {r:.2f}, p = {p:.3f}"
    except ImportError:
        r = df[["qa_acc", "mt_chrf"]].corr().iloc[0, 1]
        title = f"Cross-task: QA accuracy vs MT chrF per phenomenon (r = {r:.2f})"

    ax.set_title(title)
    ax.set_xlabel("QA open-book accuracy on items WITH phenomenon (%)")
    ax.set_ylabel("MT chrF on items WITH phenomenon")
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_cross_task.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_agreement(df: pd.DataFrame) -> None:
    sub = df[METRICS].dropna()
    if sub.empty or len(sub) < 5:
        print("Skipping metric agreement: insufficient data")
        return
    corr = sub.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRICS)))
    ax.set_yticks(range(len(METRICS)))
    ax.set_xticklabels(METRICS)
    ax.set_yticklabels(METRICS)
    for i in range(len(METRICS)):
        for j in range(len(METRICS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if corr.values[i, j] < 0.5 else "black",
                    fontsize=10, fontweight="bold")
    ax.set_title(f"Pairwise metric agreement (Pearson, n={len(sub)})")
    fig.colorbar(im, label="Correlation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_metric_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- Text summary ----------

def write_summary(df: pd.DataFrame) -> None:
    lines = ["=" * 76, "MT PILOT — SUMMARY", "=" * 76, ""]

    lines.append("SYSTEM LEVEL (mean per language × model across all items)")
    lines.append("-" * 76)
    sl = system_level(df).copy()
    sl["chrf"]     = sl["chrf"].round(2)
    sl["comet"]    = sl["comet"].round(3)
    sl["adequacy"] = sl["adequacy"].round(2)
    sl["fluency"]  = sl["fluency"].round(2)
    lines.append(sl.to_string(index=False))

    for metric in METRICS:
        lines.append("")
        lines.append("=" * 76)
        lines.append(f"PER PHENOMENON — {metric}")
        lines.append("=" * 76)
        pt = per_tag(df, metric)
        pivot_with    = pt.pivot_table(index=["lang", "phenomenon"], columns="model",
                                       values=f"{metric}_with").round(2)
        pivot_without = pt.pivot_table(index=["lang", "phenomenon"], columns="model",
                                       values=f"{metric}_without").round(2)
        pivot_delta   = pt.pivot_table(index=["lang", "phenomenon"], columns="model",
                                       values=f"{metric}_delta").round(2)
        lines.append("\nItems WITH phenomenon:")
        lines.append(pivot_with.to_string())
        lines.append("\nItems WITHOUT phenomenon:")
        lines.append(pivot_without.to_string())
        lines.append("\nΔ (with − without):")
        lines.append(pivot_delta.to_string())

    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "mt_summary.txt").write_text(text, encoding="utf-8")


# ---------- Main ----------

def main() -> None:
    items = load_jsonl(MT_RESULTS)
    if not items:
        print(f"{MT_RESULTS} not found. Run `python run_mt.py` first.")
        return

    df = flatten_mt(items)
    print(f"Loaded {len(items)} items → {len(df)} (item, model) rows\n")

    write_summary(df)
    plot_system_level(df)
    plot_per_tag(df, "chrf",     "02_per_tag_chrf.png")
    plot_per_tag(df, "comet",    "03_per_tag_comet.png")
    plot_per_tag(df, "adequacy", "04_per_tag_adequacy.png")
    plot_per_tag(df, "fluency",  "05_per_tag_fluency.png")
    plot_cross_task(df)
    plot_metric_agreement(df)

    print(f"\nWrote plots + summary to {OUT_DIR}/")


if __name__ == "__main__":
    main()
