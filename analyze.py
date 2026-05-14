"""
Analysis + plots for the multilingual diagnostic pilot (v2: Indonesian + Swahili).

Reads data/results.jsonl.

Top-level (all four benchmarks — native + translated, ID + SW):
  outputs/00_all_benchmarks_accuracy.txt
  outputs/00_accuracy_all_four_native_translated.png
  outputs/00_combined_disentanglement.png
  outputs/00_combined_paraphrase_distance.png

Per language (outputs/{lang}/):
  coverage_report.txt, accuracy_table.txt,
  01_coverage.png, 02_accuracy_aggregate.png,
  03_accuracy_sliced_native_openbook.png, 04_accuracy_sliced_translated_openbook.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("data/results.jsonl")
OUT_ROOT = Path("outputs")
OUT_ROOT.mkdir(exist_ok=True)

# Per-language phenomenon vocabularies + domain values (pilot v2: id + sw only)
LANG_CONFIG = {
    "id": {
        "name": "Indonesian",
        "native_dataset": "indoqa",
        "translated_dataset": "squad_id",
        "phenomena": ["voice_meN", "voice_di", "reduplication", "complex_NP_yang"],
        "domains": ["indonesian_history", "indonesian_geography", "general"],
        "native_label": "IndoQA (native)",
        "translated_label": "SQuAD-ID (Gemini translated)",
    },
    "sw": {
        "name": "Swahili",
        "native_dataset": "tydiqa_sw",
        "translated_dataset": "squad_sw",
        "phenomena": ["noun_class_concord", "applicative", "passive", "locative_ni"],
        "domains": ["east_african_history", "east_african_geography", "general"],
        "native_label": "TyDi QA Swahili (native)",
        "translated_label": "SQuAD-SW (Gemini translated)",
    },
}

PARAPHRASE = ["literal_match", "paraphrase", "requires_inference"]
MODELS = ["gemini_3_pro", "deepseek_v4_pro", "gemma_4_31b_it", "llama_31_8b"]

# Canonical order for “all four” summaries (ID native/trans, SW native/trans)
ALL_BENCHMARKS: list[tuple[str, str, str]] = [
    ("id", "indoqa", "IndoQA (ID native)"),
    ("id", "squad_id", "SQuAD-ID (ID translated)"),
    ("sw", "tydiqa_sw", "TyDi QA (SW native)"),
    ("sw", "squad_sw", "SQuAD-SW (SW translated)"),
]


def load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------- Coverage --------------------------------------------------------

def coverage_table(items: list[dict], lang: str) -> pd.DataFrame:
    cfg = LANG_CONFIG[lang]
    rows = []
    trans_key = cfg["translated_dataset"]
    for ds_label, ds_key in [
        (cfg["native_label"], cfg["native_dataset"]),
        (cfg["translated_label"], trans_key),
    ]:
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


def flatten_all_scored(items: list[dict]) -> pd.DataFrame:
    """All languages: one row per (item, model, condition) with a judge score."""
    parts = [flatten(items, lang) for lang in LANG_CONFIG]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _append_openbook_accuracy_slices(
    lines: list[str],
    df: pd.DataFrame,
    cfg: dict,
    dataset_id: str,
    section_title: str,
) -> None:
    """Append phenomenon / paraphrase / domain tables for one benchmark (open-book only)."""
    sub = df[(df["dataset"] == dataset_id) & (df["condition"] == "openbook")].copy()
    if sub.empty:
        return
    lines += ["", "=" * 64, section_title, "=" * 64, ""]
    for p in cfg["phenomena"]:
        sub[f"has_{p}"] = sub["phenomena"].apply(lambda xs, pp=p: pp in xs)
        slc = sub.groupby(["model", f"has_{p}"])["correct"].mean().unstack().mul(100).round(1)
        if True in slc.columns and False in slc.columns:
            slc.columns = [f"no_{p}", f"has_{p}"]
            slc[f"delta_{p}"] = (slc[f"has_{p}"] - slc[f"no_{p}"]).round(1)
            lines.append(slc.to_string())
            lines.append("")

    lines += [
        "=" * 64,
        section_title.replace("Sliced by phenomenon", "Sliced by paraphrase_distance"),
        "=" * 64,
        "",
    ]
    lines.append(
        sub.groupby(["model", "paraphrase_distance"])["correct"].mean().unstack().mul(100).round(1).to_string()
    )

    lines += [
        "",
        "=" * 64,
        section_title.replace("Sliced by phenomenon", "Sliced by domain"),
        "=" * 64,
        "",
    ]
    lines.append(sub.groupby(["model", "domain"])["correct"].mean().unstack().mul(100).round(1).to_string())


def write_all_benchmarks_summary(items: list[dict]) -> None:
    """Print + save one table covering all four benchmarks × conditions × models."""
    df = flatten_all_scored(items)
    lines = [
        "=" * 72,
        "ALL BENCHMARKS — Indonesian + Swahili (native + translated)",
        "=" * 72,
        "",
    ]
    if df.empty:
        lines.append("(No scored predictions yet — run pipeline through judge.)")
        text = "\n".join(lines)
        print(text)
        (OUT_ROOT / "00_all_benchmarks_accuracy.txt").write_text(text, encoding="utf-8")
        return

    order = [ds for _, ds, _ in ALL_BENCHMARKS]
    ds_order = {ds: i for i, ds in enumerate(order)}
    agg = df.groupby(["dataset", "condition", "model"])["correct"].agg(["mean", "count"]).reset_index()
    agg["accuracy_%"] = (agg["mean"] * 100).round(1)
    agg = agg.rename(columns={"count": "n"}).drop(columns=["mean"])
    agg["_ord"] = agg["dataset"].map(ds_order)
    agg = agg.sort_values(["_ord", "condition", "model"]).drop(columns=["_ord"])
    lines.append(agg.to_string(index=False))

    lines += ["", "-" * 72, "Tagged items per dataset:", ""]
    for lang, ds, label in ALL_BENCHMARKS:
        n = sum(1 for it in items if it["lang"] == lang and it["dataset"] == ds and "tags" in it)
        lines.append(f"  {label:<42} n_tagged={n}")

    text = "\n".join(lines)
    print(text)
    (OUT_ROOT / "00_all_benchmarks_accuracy.txt").write_text(text, encoding="utf-8")


def plot_all_four_accuracy(items: list[dict]) -> None:
    """2×2 panels: ID native, ID translated, SW native, SW translated — open vs closed per model."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    panels = [
        (0, 0, "id", "indoqa", "Indonesian — native (IndoQA)"),
        (0, 1, "id", "squad_id", "Indonesian — translated (SQuAD-ID)"),
        (1, 0, "sw", "tydiqa_sw", "Swahili — native (TyDi QA)"),
        (1, 1, "sw", "squad_sw", "Swahili — translated (SQuAD-SW)"),
    ]
    bar_w = 0.35
    x = list(range(len(MODELS)))
    c_open, c_closed = "#3B7DD8", "#E5793A"

    for r, c, lang, ds, title in panels:
        ax = axes[r][c]
        opens, closeds = [], []
        for ml in MODELS:
            vals_o, vals_c = [], []
            for it in items:
                if it["lang"] != lang or it["dataset"] != ds:
                    continue
                eo = it.get("predictions", {}).get(ml, {}).get("openbook", {})
                ec = it.get("predictions", {}).get(ml, {}).get("closedbook", {})
                if "correct" in eo:
                    vals_o.append(int(bool(eo["correct"])))
                if "correct" in ec:
                    vals_c.append(int(bool(ec["correct"])))
            opens.append(100.0 * sum(vals_o) / len(vals_o) if vals_o else 0.0)
            closeds.append(100.0 * sum(vals_c) / len(vals_c) if vals_c else 0.0)

        b1 = ax.bar([i - bar_w / 2 for i in x], opens, bar_w, label="open-book", color=c_open)
        b2 = ax.bar([i + bar_w / 2 for i in x], closeds, bar_w, label="closed-book", color=c_closed)
        for bar, v in list(zip(b1, opens)) + list(zip(b2, closeds)):
            ax.annotate(
                f"{v:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ") for m in MODELS], fontsize=8, rotation=15)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
        if r == 1:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("Accuracy (%)")
        if r == 0 and c == 1:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "All four benchmarks: native vs translated × Indonesian vs Swahili (open vs closed)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "00_accuracy_all_four_native_translated.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_accuracy_table(df: pd.DataFrame, lang: str, out_dir: Path) -> None:
    cfg = LANG_CONFIG[lang]
    if df.empty:
        print(f"({cfg['name']}: no scored predictions yet)"); return
    lines = ["=" * 64, f"ACCURACY (aggregate) — {cfg['name']}", "=" * 64, ""]
    agg = df.groupby(["dataset", "condition", "model"])["correct"].agg(["mean", "count"]).reset_index()
    agg["accuracy_%"] = (agg["mean"] * 100).round(1)
    agg = agg.rename(columns={"count": "n"}).drop(columns=["mean"])
    ds_rank = {cfg["native_dataset"]: 0, cfg["translated_dataset"]: 1}
    agg["_d"] = agg["dataset"].map(ds_rank)
    agg = agg.sort_values(["_d", "condition", "model"]).drop(columns=["_d"])
    lines.append(agg.to_string(index=False))

    nat = cfg["native_dataset"]
    trn = cfg["translated_dataset"]
    _append_openbook_accuracy_slices(
        lines, df, cfg, nat, f"Sliced by phenomenon — {cfg['name']} native open-book ({nat})"
    )
    _append_openbook_accuracy_slices(
        lines, df, cfg, trn, f"Sliced by phenomenon — {cfg['name']} translated open-book ({trn})"
    )

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


def plot_accuracy_sliced(
    df: pd.DataFrame,
    lang: str,
    out_dir: Path,
    *,
    dataset_id: str,
    outfile: str,
    bench_short: str,
) -> None:
    cfg = LANG_CONFIG[lang]
    sub = df[(df["dataset"] == dataset_id) & (df["condition"] == "openbook")].copy()
    if sub.empty:
        return
    rows = []
    for p in cfg["phenomena"]:
        sub[f"has_{p}"] = sub["phenomena"].apply(lambda xs, pp=p: pp in xs)
        for has, grp in sub.groupby(f"has_{p}"):
            for model, g2 in grp.groupby("model"):
                rows.append(
                    {
                        "phenomenon": p,
                        "slice": "with" if has else "without",
                        "model": model,
                        "accuracy": 100.0 * g2["correct"].mean(),
                        "n": len(g2),
                    }
                )
    a = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(cfg["phenomena"]), figsize=(14, 4.5), sharey=True)
    for ax, p in zip(axes, cfg["phenomena"]):
        d = a[a["phenomenon"] == p]
        if d.empty:
            ax.set_title(f"{p}\n(no data)")
            continue
        pv = d.pivot(index="slice", columns="model", values="accuracy").reindex(["without", "with"])
        pv.plot(kind="bar", ax=ax, legend=(ax is axes[-1]))
        ns = d.groupby("slice")["n"].first().reindex(["without", "with"])
        ax.set_title(f"{p}\n(n_without={ns.get('without', 0)}, n_with={ns.get('with', 0)})")
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy (%)" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.suptitle(
        f"{cfg['name']} — {bench_short} ({dataset_id}) open-book: accuracy by phenomenon",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_disentanglement(items: list[dict]) -> None:
    """Side-by-side: open-book vs closed-book accuracy for each language's native set, all models."""
    langs = [L for L in LANG_CONFIG if any(it["lang"] == L for it in items)]
    if not langs:
        return

    data: dict[tuple[str, str, str], float] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for lang, cfg in LANG_CONFIG.items():
        if lang not in langs:
            continue
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

    fig, axes = plt.subplots(1, len(langs), figsize=(6.5 * len(langs), 5.5), sharey=True)
    if len(langs) == 1:
        axes = [axes]
    bar_width = 0.35
    x = list(range(len(MODELS)))
    open_color, closed_color = "#3B7DD8", "#E5793A"

    for ax, lang in zip(axes, langs):
        cfg = LANG_CONFIG[lang]
        opens = [data.get((lang, ml, "openbook"), 0) for ml in MODELS]
        closeds = [data.get((lang, ml, "closedbook"), 0) for ml in MODELS]
        n = counts.get((lang, MODELS[0], "openbook"), 0)

        b1 = ax.bar([i - bar_width / 2 for i in x], opens, bar_width, label="open-book", color=open_color)
        b2 = ax.bar([i + bar_width / 2 for i in x], closeds, bar_width, label="closed-book", color=closed_color)

        for bar, v in list(zip(b1, opens)) + list(zip(b2, closeds)):
            ax.annotate(
                f"{v:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        for i, (o, c) in enumerate(zip(opens, closeds)):
            delta = c - o
            ax.annotate(
                f"Δ {delta:+.0f}pp",
                xy=(i, max(o, c) + 8),
                ha="center",
                fontweight="bold",
                fontsize=11,
                color="#C0392B" if delta <= -30 else "#555555",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([ml.replace("_", " ") for ml in MODELS], fontsize=9)
        ax.set_title(f"{cfg['name']} — {cfg['native_label']} (n={n})", fontsize=11)
        if ax is axes[0]:
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
    """Side-by-side coverage: native vs translated paraphrase_distance, one panel per language in LANG_CONFIG."""
    langs = [L for L in LANG_CONFIG if any(it["lang"] == L for it in items)]
    if not langs:
        return
    fig, axes = plt.subplots(1, len(langs), figsize=(6.5 * len(langs), 4.5))
    if len(langs) == 1:
        axes = [axes]
    for ax, lang in zip(axes, langs):
        cfg = LANG_CONFIG[lang]
        native = [
            it
            for it in items
            if it["lang"] == lang and it["dataset"] == cfg["native_dataset"] and "tags" in it
        ]
        trans = [
            it
            for it in items
            if it["lang"] == lang and it["dataset"] == cfg["translated_dataset"] and "tags" in it
        ]
        if not native or not trans:
            continue
        cats = PARAPHRASE
        n_n, n_t = len(native), len(trans)
        native_pct = [
            100.0 * sum(1 for it in native if it["tags"].get("paraphrase_distance") == c) / n_n for c in cats
        ]
        trans_pct = [
            100.0 * sum(1 for it in trans if it["tags"].get("paraphrase_distance") == c) / n_t for c in cats
        ]
        x = list(range(len(cats)))
        w = 0.4
        ax.bar([i - w / 2 for i in x], native_pct, width=w, label=f"native (n={n_n})", color="#3B7DD8")
        ax.bar([i + w / 2 for i in x], trans_pct, width=w, label=f"translated (n={n_t})", color="#E5793A")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=9)
        ax.set_ylabel("% of items" if ax is axes[0] else "")
        ax.set_title(f"{cfg['name']}: paraphrase distance — native vs translated", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 100)
    fig.suptitle("Native vs translated (extractive): paraphrase distance mix",
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
    write_all_benchmarks_summary(items)
    print()
    plot_all_four_accuracy(items)
    plot_combined_disentanglement(items)
    plot_combined_coverage(items)
    print(f"Wrote combined charts + summary to {OUT_ROOT}/00_*.png and 00_all_benchmarks_accuracy.txt")

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
        cfg = LANG_CONFIG[lang]
        plot_accuracy_sliced(
            df,
            lang,
            out_dir,
            dataset_id=cfg["native_dataset"],
            outfile="03_accuracy_sliced_native_openbook.png",
            bench_short="native",
        )
        plot_accuracy_sliced(
            df,
            lang,
            out_dir,
            dataset_id=cfg["translated_dataset"],
            outfile="04_accuracy_sliced_translated_openbook.png",
            bench_short="translated",
        )
        print(f"\nWrote {lang} outputs to {out_dir}/")

if __name__ == "__main__":
    main()
