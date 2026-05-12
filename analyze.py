"""
Analysis + plots for the multilingual diagnostic pilot.

Reads data/results.jsonl. Outputs per language:
  outputs/{lang}/coverage_report.txt
  outputs/{lang}/accuracy_table.txt
  outputs/{lang}/01_coverage.png
  outputs/{lang}/02_accuracy_aggregate.png
  outputs/{lang}/03_accuracy_sliced.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("data/results.jsonl")
OUT_ROOT = Path("outputs")
OUT_ROOT.mkdir(exist_ok=True)

# Per-language phenomenon vocabularies + domain values
LANG_CONFIG = {
    "ur": {
        "name": "Urdu",
        "native_dataset": "uquad",
        "phenomena": ["ergative_ne", "izafat", "light_verb", "complex_NP"],
        "domains": ["islamic_history", "pakistan_geography", "general"],
        "native_label": "UQuAD (native)",
        "translated_label": "Belebele-ur (translated)",
    },
    "id": {
        "name": "Indonesian",
        "native_dataset": "indoqa",
        "phenomena": ["voice_meN", "voice_di", "reduplication", "complex_NP_yang"],
        "domains": ["indonesian_history", "indonesian_geography", "general"],
        "native_label": "IndoQA (native)",
        "translated_label": "Belebele-id (translated)",
    },
}

PARAPHRASE = ["literal_match", "paraphrase", "requires_inference"]
MODELS = ["claude_opus_4_7", "qwen_25_7b", "llama_31_8b"]


def load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------- Coverage --------------------------------------------------------

def coverage_table(items: list[dict], lang: str) -> pd.DataFrame:
    cfg = LANG_CONFIG[lang]
    rows = []
    for ds_label, ds_key in [(cfg["native_label"], cfg["native_dataset"]),
                              (cfg["translated_label"], "belebele")]:
        sub = [it for it in items if it["lang"] == lang and it["dataset"] == ds_key and "tags" in it]
        n = len(sub)
        if n == 0:
            continue
        rec = {"dataset": ds_label, "n": n}
        for p in cfg["phenomena"]:
            rec[p] = sum(1 for it in sub if p in it["tags"].get("linguistic_phenomena", []))
        rec["none"] = sum(1 for it in sub if not it["tags"].get("linguistic_phenomena"))
        for v in PARAPHRASE:
            rec[v] = sum(1 for it in sub if it["tags"].get("paraphrase_distance") == v)
        for v in cfg["domains"]:
            rec[v] = sum(1 for it in sub if it["tags"].get("domain") == v)
        rows.append(rec)
    return pd.DataFrame(rows)


def write_coverage_report(df: pd.DataFrame, lang: str, out_dir: Path) -> None:
    cfg = LANG_CONFIG[lang]
    if df.empty:
        return
    lines = ["=" * 64, f"COVERAGE REPORT — {cfg['name']}", "=" * 64]
    for _, r in df.iterrows():
        n = r["n"]
        lines.append(f"\n{r['dataset']}  (n={n})")
        lines.append("-" * 40)
        lines.append("Linguistic phenomena (% of items):")
        for p in cfg["phenomena"] + ["none"]:
            lines.append(f"  {p:<22} {r[p]:>4}  ({100.0 * r[p] / n:5.1f}%)")
        lines.append("Paraphrase distance:")
        for v in PARAPHRASE:
            lines.append(f"  {v:<22} {r[v]:>4}  ({100.0 * r[v] / n:5.1f}%)")
        lines.append("Domain:")
        for v in cfg["domains"]:
            lines.append(f"  {v:<22} {r[v]:>4}  ({100.0 * r[v] / n:5.1f}%)")
    text = "\n".join(lines)
    print(text)
    (out_dir / "coverage_report.txt").write_text(text, encoding="utf-8")


def plot_coverage(df: pd.DataFrame, lang: str, out_dir: Path) -> None:
    cfg = LANG_CONFIG[lang]
    if df.empty or len(df) < 2:
        return
    native_pct = [100.0 * df.iloc[0][p] / df.iloc[0]["n"] for p in cfg["phenomena"]]
    trans_pct  = [100.0 * df.iloc[1][p] / df.iloc[1]["n"] for p in cfg["phenomena"]]
    x = range(len(cfg["phenomena"])); w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - w/2 for i in x], native_pct, width=w, label=df.iloc[0]["dataset"])
    ax.bar([i + w/2 for i in x], trans_pct,  width=w, label=df.iloc[1]["dataset"])
    ax.set_xticks(list(x)); ax.set_xticklabels(cfg["phenomena"], rotation=15)
    ax.set_ylabel("% of items containing phenomenon")
    ax.set_title(f"Linguistic-phenomenon coverage: native vs translated ({cfg['name']})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "01_coverage.png", dpi=150); plt.close(fig)


# ---------- Accuracy --------------------------------------------------------

def flatten(items: list[dict], lang: str) -> pd.DataFrame:
    rows = []
    for it in items:
        if it["lang"] != lang:
            continue
        tags = it.get("tags", {})
        for ml in MODELS:
            for cond, entry in it.get("predictions", {}).get(ml, {}).items():
                if "correct" not in entry:
                    continue
                rows.append({
                    "id": it["id"],
                    "dataset": it["dataset"],
                    "model": ml,
                    "condition": cond,
                    "correct": int(bool(entry["correct"])),
                    "phenomena": tags.get("linguistic_phenomena", []),
                    "paraphrase_distance": tags.get("paraphrase_distance"),
                    "domain": tags.get("domain"),
                })
    return pd.DataFrame(rows)


def write_accuracy_table(df: pd.DataFrame, lang: str, out_dir: Path) -> None:
    cfg = LANG_CONFIG[lang]
    if df.empty:
        print(f"({cfg['name']}: no scored predictions yet)"); return
    lines = ["=" * 64, f"ACCURACY (aggregate) — {cfg['name']}", "=" * 64, ""]
    agg = df.groupby(["dataset", "condition", "model"])["correct"].agg(["mean", "count"]).reset_index()
    agg["accuracy_%"] = (agg["mean"] * 100).round(1)
    agg = agg.rename(columns={"count": "n"}).drop(columns=["mean"])
    lines.append(agg.to_string(index=False))

    sub = df[(df["dataset"] == cfg["native_dataset"]) & (df["condition"] == "openbook")].copy()
    if not sub.empty:
        lines += ["", "=" * 64, f"Sliced by phenomenon — {cfg['name']} native open-book", "=" * 64, ""]
        for p in cfg["phenomena"]:
            sub[f"has_{p}"] = sub["phenomena"].apply(lambda xs: p in xs)
            slc = sub.groupby(["model", f"has_{p}"])["correct"].mean().unstack().mul(100).round(1)
            if True in slc.columns and False in slc.columns:
                slc.columns = [f"no_{p}", f"has_{p}"]
                slc[f"delta_{p}"] = (slc[f"has_{p}"] - slc[f"no_{p}"]).round(1)
                lines.append(slc.to_string()); lines.append("")

        lines += ["=" * 64, f"Sliced by paraphrase_distance — {cfg['name']} native open-book", "=" * 64, ""]
        lines.append(sub.groupby(["model", "paraphrase_distance"])["correct"].mean().unstack().mul(100).round(1).to_string())

        lines += ["", "=" * 64, f"Sliced by domain — {cfg['name']} native open-book", "=" * 64, ""]
        lines.append(sub.groupby(["model", "domain"])["correct"].mean().unstack().mul(100).round(1).to_string())

    text = "\n".join(lines)
    print(text)
    (out_dir / "accuracy_table.txt").write_text(text, encoding="utf-8")


def plot_accuracy_aggregate(df: pd.DataFrame, lang: str, out_dir: Path) -> None:
    cfg = LANG_CONFIG[lang]
    if df.empty: return
    d = df.copy(); d["panel"] = d["dataset"] + " / " + d["condition"]
    pivot = d.groupby(["panel", "model"])["correct"].mean().unstack().mul(100)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuracy (%)"); ax.set_xlabel("")
    ax.set_title(f"Accuracy: model × condition × dataset ({cfg['name']})")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=12)
    ax.grid(axis="y", alpha=0.3); ax.legend(title="model")
    fig.tight_layout(); fig.savefig(out_dir / "02_accuracy_aggregate.png", dpi=150); plt.close(fig)


def plot_accuracy_sliced(df: pd.DataFrame, lang: str, out_dir: Path) -> None:
    cfg = LANG_CONFIG[lang]
    sub = df[(df["dataset"] == cfg["native_dataset"]) & (df["condition"] == "openbook")].copy()
    if sub.empty: return
    rows = []
    for p in cfg["phenomena"]:
        sub[f"has_{p}"] = sub["phenomena"].apply(lambda xs: p in xs)
        for has, grp in sub.groupby(f"has_{p}"):
            for model, g2 in grp.groupby("model"):
                rows.append({"phenomenon": p, "slice": "with" if has else "without",
                             "model": model, "accuracy": 100.0 * g2["correct"].mean(), "n": len(g2)})
    a = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(cfg["phenomena"]), figsize=(14, 4.5), sharey=True)
    for ax, p in zip(axes, cfg["phenomena"]):
        d = a[a["phenomenon"] == p]
        if d.empty:
            ax.set_title(f"{p}\n(no data)"); continue
        pv = d.pivot(index="slice", columns="model", values="accuracy").reindex(["without", "with"])
        pv.plot(kind="bar", ax=ax, legend=(ax is axes[-1]))
        ns = d.groupby("slice")["n"].first().reindex(["without", "with"])
        ax.set_title(f"{p}\n(n_without={ns.get('without', 0)}, n_with={ns.get('with', 0)})")
        ax.set_xlabel(""); ax.set_ylabel("Accuracy (%)" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3); ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.suptitle(f"{cfg['name']} native open-book accuracy sliced by linguistic phenomenon", y=1.02)
    fig.tight_layout(); fig.savefig(out_dir / "03_accuracy_sliced.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_combined_disentanglement(items: list[dict]) -> None:
    """Side-by-side: open-book vs closed-book accuracy for Urdu + Indonesian native sets, all models.
    The killer chart for the motivation letter."""
    data: dict[tuple[str, str, str], float] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for lang, cfg in LANG_CONFIG.items():
        for ml in MODELS:
            for cond in ("openbook", "closedbook"):
                vals = []
                for it in items:
                    if it["lang"] != lang or it["dataset"] != cfg["native_dataset"]:
                        continue
                    entry = it.get("predictions", {}).get(ml, {}).get(cond, {})
                    if "correct" in entry:
                        vals.append(int(bool(entry["correct"])))
                if vals:
                    data[(lang, ml, cond)] = 100.0 * sum(vals) / len(vals)
                    counts[(lang, ml, cond)] = len(vals)

    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    bar_width = 0.35
    x = list(range(len(MODELS)))
    open_color, closed_color = "#3B7DD8", "#E5793A"

    for ax, lang in zip(axes, ["ur", "id"]):
        cfg = LANG_CONFIG[lang]
        opens   = [data.get((lang, ml, "openbook"), 0) for ml in MODELS]
        closeds = [data.get((lang, ml, "closedbook"), 0) for ml in MODELS]
        n       = counts.get((lang, MODELS[0], "openbook"), 0)

        b1 = ax.bar([i - bar_width/2 for i in x], opens,   bar_width, label="open-book",   color=open_color)
        b2 = ax.bar([i + bar_width/2 for i in x], closeds, bar_width, label="closed-book", color=closed_color)

        # Value labels on bars
        for bar, v in list(zip(b1, opens)) + list(zip(b2, closeds)):
            ax.annotate(f"{v:.0f}", xy=(bar.get_x() + bar.get_width()/2, v),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)

        # Delta annotation above each model pair
        for i, (o, c) in enumerate(zip(opens, closeds)):
            delta = c - o
            ax.annotate(f"Δ {delta:+.0f}pp",
                        xy=(i, max(o, c) + 8), ha="center",
                        fontweight="bold", fontsize=11,
                        color="#C0392B" if delta <= -30 else "#555555")

        ax.set_xticks(x)
        ax.set_xticklabels([ml.replace("_", " ") for ml in MODELS], fontsize=9)
        ax.set_title(f"{cfg['name']} — {cfg['native_label']} (n={n})", fontsize=11)
        if lang == "ur":
            ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_axisbelow(True)

    fig.suptitle("Disentangling language ability (open-book) from world knowledge (closed-book)",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "00_combined_disentanglement.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_combined_coverage(items: list[dict]) -> None:
    """Side-by-side coverage: native vs translated, both languages, paraphrase_distance + linguistic_phenomena summary."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, lang in zip(axes, ["ur", "id"]):
        cfg = LANG_CONFIG[lang]
        native = [it for it in items if it["lang"] == lang and it["dataset"] == cfg["native_dataset"] and "tags" in it]
        trans  = [it for it in items if it["lang"] == lang and it["dataset"] == "belebele"           and "tags" in it]
        if not native or not trans:
            continue
        cats = PARAPHRASE
        n_n, n_t = len(native), len(trans)
        native_pct = [100.0 * sum(1 for it in native if it["tags"].get("paraphrase_distance") == c) / n_n for c in cats]
        trans_pct  = [100.0 * sum(1 for it in trans  if it["tags"].get("paraphrase_distance") == c) / n_t for c in cats]
        x = list(range(len(cats))); w = 0.4
        ax.bar([i - w/2 for i in x], native_pct, width=w, label=f"native (n={n_n})",      color="#3B7DD8")
        ax.bar([i + w/2 for i in x], trans_pct,  width=w, label=f"translated (n={n_t})",  color="#E5793A")
        ax.set_xticks(x); ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=9)
        ax.set_ylabel("% of items" if lang == "ur" else "")
        ax.set_title(f"{cfg['name']}: paraphrase distance — native vs translated", fontsize=11)
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(0, 100)
    fig.suptitle("Native benchmarks are extractive; translated benchmarks demand paraphrase + inference",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "00_combined_paraphrase_distance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    items = load_results()
    if not items:
        print("No data/results.jsonl yet — run `python run_pipeline.py` first.")
        return

    # Cross-language headline charts (top-level outputs/)
    plot_combined_disentanglement(items)
    plot_combined_coverage(items)
    print(f"Wrote combined charts to {OUT_ROOT}/00_*.png")

    for lang in LANG_CONFIG:
        if not any(it["lang"] == lang for it in items):
            continue
        out_dir = OUT_ROOT / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n\n##### {LANG_CONFIG[lang]['name']} #####\n")
        cov = coverage_table(items, lang)
        write_coverage_report(cov, lang, out_dir)
        plot_coverage(cov, lang, out_dir)

        df = flatten(items, lang)
        write_accuracy_table(df, lang, out_dir)
        plot_accuracy_aggregate(df, lang, out_dir)
        plot_accuracy_sliced(df, lang, out_dir)
        print(f"\nWrote {lang} outputs to {out_dir}/")

if __name__ == "__main__":
    main()
