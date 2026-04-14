# Semantic-Aware LLM-Guided Initialization for Genetic Algorithms

Code for the paper *"LLM-Guided Initialization for Genetic Algorithms Under Strict Feasibility Constraints"* (PPSN 2026) — **semantic-aware variant**.

This script extends the base LLM-guided GA initialization with **aggregate semantic prompting**: the LLM receives category-level dependency, conflict, and synergy matrices derived from the problem structure, enabling it to reason about relational item interactions without ever accessing individual item attributes. This is the configuration that surpasses the handcrafted FFGR baseline on the semantic knapsack dataset (Table 6 in the paper).

---

## Overview

Standard knapsack formulations treat items independently. The semantic knapsack introduces three types of relational constraints:

- **Dependencies** — selecting item A requires item B
- **Synergies** — jointly selecting related items yields bonus value
- **Conflicts** — incompatible items incur penalties when selected together

The key innovation of this script is `build_semantic_aggregate_summary()`, which computes category-level interaction matrices (dependency, conflict, synergy) from the problem structure and passes them to the LLM as part of the initialization prompt. This allows the LLM to reason holistically about which *combinations* of categories are beneficial — something item-level greedy heuristics like FFGR cannot do.

Prompt version: `v1.2-semantic-aggregates-2026-03-23`

---

## Requirements

```bash
pip install anthropic numpy pandas matplotlib
```

Python 3.9+ recommended.

---

## API Key

```bash
export ANTHROPIC_API_KEY=your-key-here
```

Or pass it as a positional argument (see Usage below).

---

## Datasets

| `--dataset` value | Description | Required file |
|---|---|---|
| `synthetic` | Procedurally generated knapsack (easy + hard) | none |
| `hard_json` | Realistic 200-item constrained knapsack | `hard_knapsack_200.json` |
| `semantic_json` | Expedition planning with relational constraints | `semantic_expedition.json` |

Place `.json` dataset files in the working directory before running.

---

## Usage

### Quick test (2 runs, fast smoke test)

```bash
python semantic_aware_llm_init_scriptSEEME.py --quick --dataset semantic_json
```

```bash
python semantic_aware_llm_init_scriptSEEME.py --quick --dataset hard_json --easy
```

```bash
python semantic_aware_llm_init_scriptSEEME.py --quick --dataset synthetic
```

### Full camera-ready experiments (reproduces paper Table 6)

```bash
python semantic_aware_llm_init_scriptSEEME.py --camera --dataset semantic_json --runs 30 --seed0 12345
```

```bash
python semantic_aware_llm_init_scriptSEEME.py --camera --dataset hard_json --runs 30 --outdir results_hard
```

---

## CLI Reference

```
positional:
  api_key               Anthropic API key (overrides ANTHROPIC_API_KEY env var)

mode (required, mutually exclusive):
  --quick               Run a 2-run smoke test
  --camera              Run full 30-run camera-ready experiments

options:
  --dataset             synthetic | hard_json | semantic_json  (default: hard_json)
  --easy                Also run the easy (synthetic) problem variant
  --runs N              Number of independent runs for --camera  (default: 30)
  --seed0 N             Base random seed  (default: 12345)
  --outdir DIR          Output directory  (default: camera_ready_results)
  --cache-k N           LLM guidances to pre-generate and rotate across runs  (default: 5)
  --cache-file PATH     JSON cache file for LLM guidance pool  (default: llm_guidance_cache.json)
```

---

## LLM Guidance Caching

To avoid redundant API calls, LLM guidances are generated once and cached. The cache key includes the prompt version, model name, dataset identity, and a SHA-256 hash of the dataset file — so the cache is automatically invalidated when any of these change.

To force regeneration:

```bash
rm llm_guidance_cache.json
```

---

## Outputs

Written to `--outdir` (default: `camera_ready_results/`):

- Per-method summary statistics (mean ± std best feasible fitness, feasibility rate, wall time)
- Convergence curves: best feasible fitness by generation
- Raw per-run data as CSV

---

## Code Structure

```
semantic_aware_llm_init_scriptSEEME.py
│
├── Data classes
│   ├── Item / SimpleItem          # Standard knapsack items
│   └── SemanticItem               # Items with requires / synergies / conflicts / enables
│
├── Problem classes
│   ├── SimpleProblem              # Easy synthetic (50 items, 1 constraint)
│   ├── HardKnapsackProblem        # Hard synthetic (200 items, 4 constraint types)
│   └── SemanticKnapsackProblem    # Expedition planning with relational constraints
│
├── build_semantic_aggregate_summary()
│   └── Builds category-level dependency / conflict / synergy matrices
│       for LLM prompting — no item-level identities exposed
│
├── LLMGuidance
│   ├── _get_simple_guidance()     # Prompt for easy instances
│   ├── _get_hard_guidance()       # Prompt for hard constrained instances
│   └── _get_semantic_guidance()   # Semantic-aware prompt with relational matrices
│
├── GeneticAlgorithm
│   ├── initialize_population_random()
│   ├── initialize_population_ffgr()         # Handcrafted greedy baseline
│   └── initialize_population_llm_guided()   # LLM-guided, semantic-aware
│
├── get_guidance_pool()            # Cache-aware LLM guidance pre-generation
├── quick_test()                   # Fast smoke test
└── camera_ready_experiments()     # Full reproducible evaluation (30 runs)
```

---

## Key Design Choices

- **Single LLM query per run.** The LLM is never called inside the evolutionary loop.
- **Aggregate-only information.** The LLM receives category-level statistics and interaction matrices — never individual item identifiers, values, or solution candidates.
- **Semantic live multiplier.** During LLM-guided initialization, a `semantic_live_multiplier()` function re-scores items at construction time using the aggregate synergy/conflict matrices, allowing initialization to respect relational structure dynamically.
- **Strict feasibility.** All reported metrics count only strictly feasible solutions. No penalized fitness is reported anywhere.
- **Deterministic safety clamps.** All LLM scalar outputs are clamped to valid ranges before use, ensuring robustness to model variance.

---

## Results Summary (from paper, Table 6 — Semantic Dataset, 30 runs)

| Method | Best Feasible Fitness | Std | Time (s) |
|---|---|---|---|
| Pure GA | 4825.0 | 41.7 | 0.79 |
| FFGR Init | 4864.9 | 33.3 | 0.94 |
| **LLM Init** | **4893.7** | **24.1** | 1.08 |
| **LLM Init + Adaptive** | **4904.5** | **21.9** | 3.43 |

LLM Init outperforms the handcrafted FFGR baseline in both mean fitness and variance (p = 0.0782 vs FFGR; LLM Init + Adaptive vs FFGR: p = 0.0146).

---

## Citation

```bibtex
@inproceedings{llm-ga-ppsn2026,
  title     = {LLM-Guided Initialization for Genetic Algorithms Under Strict Feasibility Constraints},
  booktitle = {Parallel Problem Solving from Nature (PPSN 2026)},
  year      = {2026}
}
```
