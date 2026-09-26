# Adaptive splitting for stochastic samplers

Code for the paper's experiments: a two-phase procedure that learns how many
branches to split into at each point of a sampler's trajectory, compared
against fixed-N and uniform-c splitting.

## Setup

```bash
uv sync
```

Python 3.12. On Linux x86_64 this installs CUDA 12.8 builds of PyTorch. Every
script also runs on CPU (`--device cpu`), slowly.

## Benchmarks

| Paper benchmark | Runner | Config | Needs |
|---|---|---|---|
| Ornstein–Uhlenbeck | `simple_ou` | `default` | – |
| Coupled double-well Langevin | `coupled_double_well_langevin` | `default` | – |
| EDM, 2D Gaussian mixture | `edm_gmm2d` | `default` | `./train.sh` |
| DDPM, CIFAR-10 | `ddpm_cifar10_hf` | `mmd` | downloads `google/ddpm-cifar10-32` |
| Latent diffusion, FFHQ | `ldm_ffhq` | `mmd` | downloads `asparius/ldm-ffhq-256` |
| OU minimax oracle | `ou_oracle` | `default` | – |

`default` configs score samples with KS; `mmd` configs use MMD.

## Running

To reproduce everything:

```bash
./train.sh                # only needed for edm_gmm2d
./compare.sh              # all experiments; or e.g. ./compare.sh simple_ou
```

`compare.sh` reads `DEVICE` (default `cuda:0`), `BASE_DIR` (default
`outputs_paper_final`) and `DEBUG=1` (verbose logging) from the environment. For one experiment, it runs:

```bash
uv run python src/ensure_samples.py --runner simple_ou --config default --batch_size 1000000
uv run python src/compare.py --runner simple_ou --config default \
    --output_dir outputs_paper_final/simple_ou --n_runs 2500 --n_parallel 100
uv run python src/plots.py outputs_paper_final/simple_ou/compare_results_<splits>.csv --ci 0.95
```

1. `ensure_samples.py` builds the reference samples the metrics compare
   against. They are cached under `checkpoints/<runner>/`.
2. `compare.py` runs the sweep. Each completed run is cached in
   `<output_dir>/runs/`, so an interrupted sweep resumes where it stopped.
   It writes one `compare_results_<splits>.csv` per split schedule, plus
   `sweep.json` (the resolved config) and `compare_outputs.json` (a list of the
   CSVs).
3. `plots.py` draws per-experiment diagnostics from a CSV.

Once all six experiments are complete, the paper's figures and tables are
rebuilt with:

```bash
uv run python paper/paper_plots.py     # -> plots/
uv run python paper/paper_tables.py    # -> tables/
```

## Configs

Each runner lives in `src/runners/<family>/<case>/` as a `runner.py` (the
sampler) and a `configs.py`. The `configs.py` exports a dict
`CONFIGS = {"default": {...}, "mmd": {...}}`, and `--config` picks an entry.
Shared defaults are in `src/runners/common_configs.py` and, for the SDE cases,
`src/runners/sde/configs.py`. To change an experiment, edit these dicts or add
a new named entry.

| Key | Meaning |
|---|---|
| `B_list` | Total budgets B to sweep; every method at a given B spends the same compute. |
| `B1_list` | Phase-1 (pilot) budget for each B. An int is an absolute budget; a float in (0, 1) is a fraction of B; a string `"C,alpha"` means `floor(C * B**alpha)`. Entries with B1 ≥ B are skipped. |
| `split_percentages_list` | Split points, as fractions of the remaining trajectory. `split_schedules()` builds evenly spaced schedules for each count in `SPLIT_COUNTS` (9, 19, 39). |
| `optimization_modes` | Allocation rule: `monotone` is the learned allocation, `learned_c` is the Learned-c baseline. |
| `baselines` | `fixed_N` (no splitting), `uniform_c` (the same factor c at every split; values from `UNIFORM_C_BY_SPLIT_COUNT`), or a named solver baseline. |
| `sampling_configs`, `step_schedules` | Sampler and its parameters; step count per budget (a `{B: steps}` map). |
| `metrics`, `primary_metric` | `ks` and/or `mmd`. |
| `phase1` | Set by `mmd_sibling_config(...)` for MMD configs; KS configs use CrossFit-Q, tuned by the `--crossfit_q_*` flags. |
| `n_runs`, `n_parallel` | Repetitions per setting and how many run at once (both overridable on the command line). |

Example: a quick OU sweep with a pilot budget of 10% of B.

```python
# src/runners/sde/simple_ou/configs.py
CONFIGS["quick"] = {**CONFIGS["default"], "B_list": [100_000], "B1_list": [0.1], "n_runs": 50}
```

```bash
uv run python src/ensure_samples.py --runner simple_ou --config quick --batch_size 1000000
uv run python src/compare.py --runner simple_ou --config quick --output_dir outputs/quick
```
