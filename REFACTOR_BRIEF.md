# Refactor Brief: Multilingual Evaluation Pilot v2

## Context

This repo currently runs a diagnostic pilot comparing native vs translated benchmarks for **Urdu** and **Indonesian**, then evaluates 3 models in open-book / closed-book / MCQ conditions. Current pipeline: `run_pipeline.py` (prepare → tag → infer → judge) + `analyze.py` (coverage + accuracy + charts).

**Two methodological issues require a v2:**

1. **UQuAD is not native Urdu QA** — turned out to be adapted/translated. The Urdu "native vs translated" comparison is invalid.
2. **Native QA (extractive) vs Belebele (MCQ) introduces a format confound** — accuracy differences can't be cleanly attributed to native vs translated when task format also differs.

**v2 goal:** Clean, format-matched, methodology-pure pilot:
- **Native extractive QA** vs **translated extractive QA** (same format both sides)
- **Same translation engine** for translated benchmarks across both languages (eliminate translation pipeline confound)
- **Two genuinely native baselines** with verified provenance

## What changes

### Languages
- ❌ Drop **Urdu** (no reliable native QA available)
- ✅ Keep **Indonesian**
- ➕ Add **Swahili** (Bantu, low-resource African language, typologically distant from Indonesian)

### Datasets

| Language | Native (extractive QA) | Translated (extractive QA) |
|---|---|---|
| Indonesian | **IndoQA** (existing, keep) | **Gemini-translated SQuAD** (new — translate fresh from English SQuAD) |
| Swahili | **TyDi QA Swahili** (new, HF: `google-research-datasets/tydiqa`, secondary task) | **Gemini-translated SQuAD** (new — same English SQuAD subset, translated to Swahili) |

**Crucial:** Both translated benchmarks must come from the **same English SQuAD subset**, translated via the **same engine (Gemini 3 Flash)**. This isolates target-language properties from translation-pipeline artifacts.

### Sample sizes
- 500 items per benchmark (4 benchmarks × 500 = 2000 items total)
- Source for translation: 500-item subset of SQuAD v1.1 or v2.0 English train (deterministic, seeded)

### Translation engine
- **Gemini 3 Flash** via Google API (user has pricing: $0.50/M input, $3/M output)
- Estimated cost: ~$0.30 for full 500 items × 2 languages

### Drop entirely
- ❌ All Urdu datasets (`uquad`, `belebele urd_Arab`)
- ❌ All Belebele MCQ data (`belebele ind_Latn`)
- ❌ MCQ condition / `belebele` dataset branch in pipeline
- ❌ `BELEBELE_TMPL`, `BELEBELE_CONFIG`, MCQ-related judging
- ❌ `urdu_uquad.jsonl` raw file

## New pipeline flow

```
Stage 0 (NEW): translate
   Input:  SQuAD English subset (500 items, deterministic seed)
   Engine: Gemini 3 Flash via google-genai SDK
   Output: data/squad_id_gemini.jsonl, data/squad_sw_gemini.jsonl
   IMPORTANT: must preserve answer-span extractability — answer text must appear
   verbatim in translated context. Two strategies:
     (a) translate context + question, then extract span via LLM
     (b) translate context, then translate Q+A as a tuple, ensuring A appears in C
   Recommend (a): translate context first, then use Gemini to locate answer span
   in translated context given English Q+A as reference. Items where span can't
   be located → drop or flag.

Stage 1: prepare
   Loads 4 benchmarks into unified format:
     - indoqa (native, id, extractive)
     - squad_id (Gemini-translated, id, extractive)
     - tydiqa_sw (native, sw, extractive) — use secondary_task split, minimal-answer
     - squad_sw (Gemini-translated, sw, extractive)
   500 items each, deterministic sampling.

Stage 2: tag  (rewrite per-language)
   Indonesian: keep existing tag rules (voice_meN, voice_di, reduplication, complex_NP_yang)
   Swahili: NEW tag rules (see below)
   Tagger: keep deepseek-v4-pro via OpenRouter (or upgrade to Claude if cheaper for tagging)

Stage 3: infer
   Conditions: openbook + closedbook ONLY (no mcq)
   Models: keep current 3 (claude-opus-4.7, qwen-2.5-7b, llama-3.1-8b)
   System prompts: add Swahili variant

Stage 4: judge
   Extractive QA only, LLM-judge with gpt-4o-mini (keep existing logic)
```

## Swahili tag rules (NEW — add to TAG_USER_TMPL)

Tag the QUESTION sentence only, same as other languages.

```
1) linguistic_phenomena — list of phenomena present in the QUESTION sentence:
   - "noun_class_concord": noun class agreement marker on adjective, verb, or
     possessive (M-Wa class: m-/wa- prefix; Ki-Vi class: ki-/vi-; N class: n-;
     Ji-Ma class: ji-/ma-; U class: u-/n-; etc.). Examples: "watoto wadogo"
     (small children, M-Wa concord), "vitabu vyangu" (my books, Ki-Vi concord).
   - "applicative": verb with applicative extension -i-/-e- adding a benefactive
     or directional argument. Examples: andikia (write to/for), pikia (cook for).
   - "passive": verb with passive extension -w-. Examples: andikwa (be written),
     pikwa (be cooked). Do NOT confuse with question word "wapi" (where).
   - "locative_ni": noun + locative suffix -ni indicating location/direction.
     Examples: nyumbani (at home), shuleni (at school), mjini (in town).
     Bare "ni" as copula ("ni mtoto", "is a child") does NOT count.

2) paraphrase_distance — same options as other languages:
   - "literal_match" / "paraphrase" / "requires_inference"

3) domain — Swahili-specific:
   - "east_african_history": Tanzania/Kenya/Uganda historical figures, events
   - "east_african_geography": Tanzania/Kenya/Uganda/Swahili-coast geography
   - "general": anything else
```

## Letter alignment

The motivation letter (separate doc) commits to:
- Two typologically distant low-resource languages → **Indonesian + Swahili**
- Pilot on native vs translated benchmarks → must hold up methodologically
- Reviewer feedback flagged need for MT extension → see "Stretch goal" below

## File changes

### Modify
- `run_pipeline.py`:
  - Remove Urdu, Belebele, MCQ paths
  - Add Swahili (sw) paths throughout
  - Add Stage 0 (`translate.py` or `stage_translate` function) for Gemini translation
  - Add `google-genai` to imports + requirements
  - Update `CONDITIONS` to drop "mcq"
  - Update `TAG_USER_TMPL` with Swahili variant
  - Update `INFER_SYSTEM`, `OPENBOOK_TMPL`, `CLOSEDBOOK_TMPL` with Swahili variants
  - Update `LANG_NAME` to include Swahili
- `analyze.py`:
  - Update `LANG_CONFIG` to replace `ur` with `sw`, add Swahili phenomena + domains
  - Update `plot_combined_disentanglement` and `plot_combined_coverage` for `sw` / `id`
- `requirements.txt`:
  - Add `google-genai>=0.3.0`
- `README.md`:
  - Rewrite for Indonesian + Swahili setup
  - Add methodology rationale (why format-matched, why same engine)
  - Add Limitations section noting Gemini translation as variable
  - Add results once pilot complete

### Create
- `translate_squad.py` (new): translates SQuAD English subset → Indonesian + Swahili via Gemini
  - Deterministic sampling (seed 42)
  - Answer-span preservation via two-step prompt (translate context, then locate translated span)
  - Output: `data/squad_id_gemini.jsonl`, `data/squad_sw_gemini.jsonl`
  - Each record: `{id, src_question_en, src_answer_en, question, context, gold, span_found: bool}`

### Delete
- `data/urdu_uquad.jsonl`
- `data/results.jsonl` (regenerate from scratch)
- `outputs/ur/` directory

### Env vars
- Add `GOOGLE_API_KEY` for Gemini
- Keep `OPENROUTER_API_KEY` for tagger / models / judge

## Stretch goal (if time permits): MT extension

After core pilot complete, add MT-side coverage analysis to address reviewer feedback that pilot is QA-only:

- Pull **FLORES-200** Indonesian (`ind_Latn`) and Swahili (`swh_Latn`) test sets
- Pull natural-language baseline: 500-sentence sample from **OSCAR-23.01** Indonesian + Swahili
- Apply same linguistic tag rules (Indonesian + Swahili) to both source sets
- Compute coverage distribution: FLORES (translated benchmark) vs OSCAR (natural language)
- Output: `outputs/mt_extension/coverage_flores_vs_natural.png` + writeup section in README

This is the **separate, format-clean MT pilot** that demonstrates the diagnostic methodology generalizes beyond QA — directly addressing reviewer feedback about MT alignment.

## Key constraints

1. **Reproducibility**: All sampling seeded (seed=42 for SQuAD subset, existing seeds for native datasets).
2. **Modular preservation**: Keep the current 4-stage pipeline pattern. Each stage idempotent + resumable via field-presence checks (existing design).
3. **No hand-editing of translated data**: Translation pipeline must be reproducible from scratch given API keys.
4. **Honest limitations**: README must call out that Gemini translation quality varies, that we sample 50 items for sanity check, that translation pipeline itself is a study variable.
5. **Sample size flexibility**: If Gemini translation produces >10% items with broken answer-span extractability, fall back to 300/benchmark rather than padding with bad items.

## Validation steps (build into pipeline)

After Stage 0 translation:
1. Spot-check 20 random translated items per language for fluency (manual review)
2. Verify answer-span extractability rate (target >85%)
3. Print summary: `Indonesian: 487/500 valid (97.4%) | Swahili: 451/500 valid (90.2%)`

After Stage 2 tagging:
4. Print tag distribution per benchmark (4 benchmarks × top tags)
5. Flag if any tag has <5 instances in any benchmark (rare-event warning)

## Deliverables

When refactor complete:
1. Working `python run_pipeline.py` end-to-end (translate → prepare → tag → infer → judge)
2. Working `python analyze.py` producing coverage + accuracy charts for Indonesian + Swahili
3. Updated README.md with v2 setup, methodology, findings, limitations
4. `data/results.jsonl` populated with full 2000-item pilot
5. (Optional, if time) MT extension analysis + chart

## Time budget

Saturday 2026-05-16, ~10-12 hours:
- 1h: Setup, env, Gemini API access verification
- 2h: Stage 0 translation (Indonesian + Swahili)
- 2h: Spot-check + Swahili tag rule refinement
- 2h: Stages 1-4 pipeline run (parallelized via existing ThreadPoolExecutor)
- 1h: Analysis + chart regeneration
- 2h: README rewrite
- 2h: (Stretch) MT extension

Deadline: Sunday 2026-05-17, 1pm CEST (Inria submission).
