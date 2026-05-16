# Fine-Grained Multilingual Evaluation: Beyond Aggregate Benchmarks

> Standard NLP evaluations treat benchmarks as black boxes, reporting a single macro-averaged score. This repository demonstrates that aggregate scores mask systemic model failures. True model capability is only revealed when evaluating at the grain size of structural linguistics (Machine Translation Pilot) and semantic complexity (Question Answering Pilot).

---

## TL;DR

This project consists of two pilot studies, Question Answering (QA) and Machine Translation (MT), to prove that aggregate evaluation scores are misleading. By breaking down the data into specific categories, we find the exact weak points of Large Language Models.

### Pilot 1: Question Answering (QA) Insights
- **The Knowledge vs. Extraction Gap**: We break down the evaluation into Open-Book (with context), Closed-Book (without context), Native, and Translated datasets. Under Closed-Book conditions, all models show a massive drop in accuracy. This proves that while models are good at extracting answers when given a text, they lack deep internal knowledge of mid-to-low resource languages.
- **The Logic Breakdown**: When we analyze the data by paraphrase distance, we find that every model fails on sentences that require inference (sentences that need combining facts, counting, or logical steps). In Open-Book settings, models look perfect (~95% accuracy) on simple text-matching, but drop to a low 50% when they actually have to think more.
- **The Morphosyntactic Passive Illusion**: The data shows how translated datasets hide real grammar challenges. In the translated dataset, all models handle passive sentences easily, with Gemini reaching 100% accuracy and Llama showing a 16.2% jump in performance. However, in the native dataset, passive structures cause a real drop in accuracy. This drop is small for frontier models like Gemini (7.3%) and Gemma (4.4%), but it remains a severe problem for DeepSeek (21.3%) and Llama (18.7%) due to the complex grammar chains of native Swahili. 
- **Local Context is Erased in Translations**: Nearly half (47.8%) of the native Indonesian dataset (IndoQA) covers local history and geography, while the translated dataset (SQuAD-ID) drops to just 0.8%. Translated benchmarks only measure generic world knowledge, not language-specific competence. This shows it is really important to build and use native benchmarks for each language, rather than relying on unified but translated datasets.

### Pilot 2: Machine Translation (MT) Insights
- **The Resource-Tier Collapse for Small Models**: We tested 4 models (Gemini 3.1 Pro, DeepSeek V4 Pro, Gemma 4 31B, Llama 3.1 8B) on French (high-resource), Indonesian (mid-resource), and Swahili (low-resource). Llama achieves chrF 64 on French and 62 on Indonesian, but drops to 41 on Swahili (judge-adequacy 1.56/5, basically unintelligible). Frontier models like Gemini stay within a 7-point chrF band across all three tiers. So "multilingual capability" averaged across languages hides massive resource-tier gaps. The same small model can be functional in one language and completely broken in another, depending only on which language it has to translate.
- **Each Language Has Its Own Linguistic Bottleneck**: When we slice MT scores by linguistic phenomenon, each language reveals a different weakness. Swahili noun class agreement causes 6 to 13 chrF drops across all four models, including frontier ones (Gemini -13.2, Gemma -13.3). French pre-verbal clitics cause a consistent 5 to 6 chrF drop across all models. Indonesian shows no phenomenon-specific weakness, with voice morphology and relative clauses handled equally well across the board. So the same diagnostic method surfaces different problems depending on the language. This shows that linguistic evaluation needs to be tailored per language, not used as one universal framework.
- **Bantu Agreement Morphology Breaks Even Frontier Models**: The Swahili noun_class_concord drop is the most striking finding. Gemini 3.1 Pro drops from 78.4 to 65.2 chrF on items containing concord. Gemma 4 31B drops 75.9 to 62.6, and Llama's COMET score crashes by 0.20 points. This is not a small-model problem. It scales across all model capability tiers. Aggregate Swahili MT scores would only say "models struggle on Swahili" without explaining why. The diagnostic shows exactly which morphological feature breaks the translation.
- **Same Language, Different Bottleneck per Task**: In the QA pilot, Swahili passive constructions caused the biggest accuracy drop (-26 to -38pp). In the MT pilot, noun class concord dominates (-7 to -13 chrF), while passive impact is much smaller (-2 to -4 chrF). Same language, same tagging method, but the hardest feature depends entirely on the task. So linguistic difficulty cannot be reduced to one universal taxonomy. Diagnostic methods need to be built specifically for each (language, task) pair.

> **For complete data tables, all sliced findings, and confound controls, see [ANALYSIS.md](ANALYSIS.md).**

## Motivation

Current multilingual benchmarks report a single accuracy number, but that number conflates multiple distinct capabilities: linguistic competence, world knowledge stored in the model, and artifacts of benchmark construction. This pilot demonstrates a diagnostic methodology that systematically separates these confounds across languages and tasks.

## Setup

### Languages (3 resource tiers, 3 families)

| Language | Family | Resource tier | Morphology | Script |
|---|---|---|---|---|
| French | Romance | High | Moderate (clitics, agreement, subjunctive) | Latin |
| Indonesian | Austronesian | Mid | Voice morphology (meN-, di-), reduplication | Latin |
| Swahili | Bantu | Low | Rich agglutinative (noun classes, verb extensions, locative suffix) | Latin |

### Datasets

| Task | Language | Native source | Translated/parallel |
|---|---|---|---|
| QA | Indonesian | IndoQA (n=500, natively authored) | Gemini-translated SQuAD (n=499, same English subset) |
| QA | Swahili | TyDi QA Swahili (n=500, natively authored) | Gemini-translated SQuAD (n=498, same English subset) |
| MT | French | FLORES-200 Plus `fra_Latn` devtest, filtered (n=365) | EN→FR via 4 models |
| MT | Indonesian | FLORES-200 Plus `ind_Latn` devtest, filtered (n=103) | EN→ID via 4 models |
| MT | Swahili | FLORES-200 Plus `swh_Latn` devtest, filtered (n=134) | EN→SW via 4 models |

Translated QA benchmarks use the **same English SQuAD subset** translated to both target languages via the **same engine (Gemini 3 Flash)** to control for translation pipeline as a confound. MT items are sampled from FLORES devtest by iterative tag-and-filter until each linguistic-phenomenon bucket has ≥50 items.

### Models tested (4)

| Label | Model ID | Origin | Class |
|---|---|---|---|
| `gemini_3_pro` | google/gemini-3.1-pro-preview | Western (Google) | Frontier |
| `deepseek_v4_pro` | deepseek/deepseek-v4-pro | Chinese | Mid-frontier |
| `gemma_4_31b_it` | google/gemma-4-31b-it | Western open (Google) | Mid (31B) |
| `llama_31_8b` | meta-llama/llama-3.1-8b-instruct | US (Meta) | Small (8B) |

### Tagger and judge

- **Linguistic tagger**: Claude Sonnet 4.6 via OpenRouter. Tags each item with language-specific morphosyntactic phenomena, paraphrase distance, and domain (QA only).
- **QA judge**: GPT-4o-mini, binary correct/incorrect on extractive QA against gold answer.
- **MT judge**: GPT-5.4-mini (OpenAI direct), 1–5 scales for adequacy and fluency, alongside chrF (sacrebleu) and COMET-22 (Unbabel/wmt22-comet-da).

### Linguistic phenomena tagged per language

**Indonesian** (4): `voice_meN`, `voice_di`, `complex_NP_yang`, `reduplication` (bonus).
**Swahili** (4): `noun_class_concord`, `passive`, `locative_ni`, `applicative` (bonus).
**French** (4): `clitic_pronoun`, `subjunctive`, `complex_NP_relative`, `past_participle_agreement` (bonus).

### Conditions

- **Open-book** (QA): passage given. Tests reading comprehension and language ability.
- **Closed-book** (QA): no passage. Tests world knowledge encoded in the model in that language.
- **MT**: English source → target language reference, scored against reference.


## Reproduce

```bash
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=sk-or-..."   > .env
echo "GOOGLE_API_KEY=..."             >> .env
echo "OPENAI_API_KEY=sk-..."          >> .env

# QA pipeline (Indonesian + Swahili, ~30–45 min, ~$25–30)
python run_pipeline.py              # prepare → tag → infer → judge
python analyze.py                   # coverage + accuracy + charts

# MT pipeline (Indonesian + Swahili + French, ~30–45 min, ~$10–15)
python run_mt.py                    # filter+tag → translate → chrF + COMET + LLM judge
python analyze_mt.py                # per-tag breakdown + cross-task + metric agreement
```

Each stage is idempotent and resumable. Re-runs only process missing fields. Smoke test: `python run_mt.py --per-bucket 5 --skip-comet` (≈ $1, ≈ 2 min).

---

## File layout

```
.
├── run_pipeline.py             QA: prepare + tag + infer + judge
├── analyze.py                  QA: coverage + accuracy + charts
├── run_mt.py                   MT: filter+tag + translate + chrF/COMET/judge
├── analyze_mt.py               MT: per-tag breakdown + cross-task correlation
├── ANALYSIS.md                 Detailed analysis report: complete data tables and slicing
├── requirements.txt
├── data/
│   ├── results.jsonl           QA: one record per item, all stages combined
│   ├── mt_results.jsonl        MT: one record per item, translations + scores
│   ├── indoqa_val.json         IndoQA cache
│   ├── squad_id_gemini.jsonl   Gemini-translated SQuAD → Indonesian
│   └── squad_sw_gemini.jsonl   Gemini-translated SQuAD → Swahili
├── outputs/
│   ├── 00_combined_disentanglement.png       QA: open vs closed-book per model per lang
│   ├── 00_combined_paraphrase_distance.png   QA: paraphrase distance native vs translated
│   ├── 00_all_benchmarks_accuracy.txt        QA: full accuracy table
│   ├── id/  (per-language QA: coverage + accuracy + sliced)
│   ├── sw/  (per-language QA: same)
│   └── mt/
│       ├── 01_system_level.png         MT: 4 metrics per model per lang
│       ├── 02_per_tag_chrf.png         MT: chrF per phenomenon (with Δ vs without)
│       ├── 03_per_tag_comet.png        MT: COMET per phenomenon
│       ├── 04_per_tag_adequacy.png     MT: adequacy per phenomenon
│       ├── 05_per_tag_fluency.png      MT: fluency per phenomenon
│       ├── 06_cross_task.png           QA-MT correlation per phenomenon (ID+SW only)
│       ├── 07_metric_agreement.png     Pearson correlation across 4 MT metrics
│       └── mt_summary.txt              MT: full summary tables
└── README.md                   ← this file
```
