# LLM-Guided Initialization for Genetic Algorithms Under Strict Feasibility Constraints

Code for the paper *"LLM-Guided Initialization for Genetic Algorithms Under Strict Feasibility Constraints"* (PPSN 2026).

## Overview

This repository provides a controlled empirical study of LLM-guided population initialization for Genetic Algorithms (GAs) on constrained combinatorial optimization problems. Feasibility is enforced as a **hard constraint** throughout evaluation — no soft penalties, no relaxed fitness.

The core idea is simple: query an LLM **once per run** using only aggregate dataset statistics (never individual items) to derive initialization biases for a constraint-aware greedy constructor. The GA itself is unchanged.

Four configurations are compared:

| Method | Description |
|---|---|
| `Pure GA` | Standard GA with random initialization |
| `FFGR Init` | Handcrafted feasibility-first greedy initializer (oracle baseline) |
| `LLM Init` | LLM-guided initialization from aggregate statistics |
| `LLM Init + Adaptive` | LLM-guided init with sparse adaptive mutation control |

## Requirements

```bash
pip install anthropic numpy pandas matplotlib
```

Python 3.9+ is recommended.

## API Key

The script uses the [Anthropic API](https://www.anthropic.com/). Provide your key either as a positional argument or via environment variable:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Datasets

Three dataset modes are supported:

| `--dataset` value | Description |
|---|---|
| `synthetic` | Procedurally generated knapsack (easy + hard variants) |
| `hard_json` | Realistic 200-item constrained knapsack (`hard_knapsack_200.json`) |
| `semantic_json` | Expedition planning with dependencies, synergies, and conflicts (`semantic_expedition.json`) |

For `hard_json` and `semantic_json`, place the corresponding `.json` files in the working directory before running.

## Usage

### Quick test (2 runs, fast)

```bash
python corrected_ffgr_llm_init_script___SEE_ME.py --quick --dataset synthetic
```

```bash
python corrected_ffgr_llm_init_script___SEE_ME.py --quick --dataset hard_json --easy
```

```bash
python corrected_ffgr_llm_init_script___SEE_ME.py --quick --dataset semantic_json
```

### Full camera-ready experiments (30 runs, reproduces paper results)

```bash
python corrected_ffgr_llm_init_script___SEE_ME.py --camera --dataset hard_json --runs 30 --seed0 12345
```

```bash
python corrected_ffgr_llm_init_script___SEE_ME.py --camera --dataset semantic_json --runs 30 --outdir results_semantic
```

### All CLI options

```
positional:
  api_key               Anthropic API key (overrides env var)

mode (required, mutually exclusive):
  --quick               Run a quick 2-run test
  --camera              Run full 30-run camera-ready experiments

options:
  --dataset             Dataset: synthetic | hard_json | semantic_json (default: hard_json)
  --easy                Also run the easy (synthetic) problem variant
  --runs N              Number of independent runs for --camera (default: 30)
  --seed0 N             Base random seed (default: 12345)
  --outdir DIR          Output directory for results and plots (default: camera_ready_results)
  --cache-k N           Number of distinct LLM guidances to generate and rotate across runs (default: 5)
  --cache-file PATH     Path to JSON cache file for LLM guidance pool (default: llm_guidance_cache.json)
```

## LLM Guidance Caching

To avoid redundant API calls across runs, the script caches LLM-generated initialization guidances. On the first run, `--cache-k` distinct guidances are generated and stored in `--cache-file`. Subsequent runs reuse the cache if the dataset and prompt version match.

To force regeneration, delete the cache file:

```bash
rm llm_guidance_cache.json
```

## Outputs

Results are written to `--outdir` (default: `camera_ready_results/`) and include:

- Per-method summary statistics (mean ± std best feasible fitness, feasibility rate, runtime)
- Convergence curves (best feasible fitness by generation, one plot per problem)
- Raw per-run data as CSV

## Code Structure

```
corrected_ffgr_llm_init_script___SEE_ME.py
│
├── Problem definitions
│   ├── SimpleProblem          # Easy synthetic knapsack (50 items, 1 constraint)
│   ├── HardKnapsackProblem    # Hard synthetic knapsack (200 items, 4 constraint types)
│   └── SemanticKnapsackProblem  # Expedition knapsack with relational constraints
│
├── Initialization strategies
│   ├── initialize_population_random()       # Pure GA baseline
│   ├── initialize_population_ffgr()         # Handcrafted greedy (oracle baseline)
│   └── initialize_population_llm_guided()   # LLM-derived heuristic biases
│
├── LLMGuidance                # Single-query LLM interface with guidance caching
├── GeneticAlgorithm           # Standard GA (tournament selection, 2-pt crossover, bit-flip mutation)
│
├── run_experiment_on_problem()  # Single-problem experiment runner
├── quick_test()                 # Fast smoke test
└── camera_ready_experiments()   # Full reproducible evaluation
```

## Key Design Choices

- **Single LLM query per run.** The LLM is never called inside the evolutionary loop.
- **Aggregate-only information.** The LLM never sees individual items, solution candidates, or fitness values — only category-level statistics.
- **Strict feasibility.** All reported metrics (best fitness, feasibility rate, convergence curves) count only strictly feasible solutions. No penalized fitness is reported.
- **Reproducibility.** Fixed seeds, identical GA hyperparameters across all methods, and deterministic safety clamps on all LLM outputs.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{llm-ga-ppsn2026,
  title     = {LLM-Guided Initialization for Genetic Algorithms Under Strict Feasibility Constraints},
  booktitle = {Parallel Problem Solving from Nature (PPSN 2026)},
  year      = {2026}
}
```

