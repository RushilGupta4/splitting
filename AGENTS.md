# AGENTS.md

Guidance for coding agents working in this repository. Read `README.md` first
for setup, the benchmark list and the config keys.

## Commands

Run everything from the repository root with `uv run`; scripts import from
`src/` by path, so run them as `python src/<script>.py`.

```bash
uv sync
./train.sh                                   # edm_gmm2d checkpoint
./compare.sh [experiment ...]                # full pipeline, see EXPERIMENTS in the script
uv run python src/ensure_samples.py --runner R --config C --batch_size N
uv run python src/compare.py --runner R --config C --output_dir DIR
uv run python src/plots.py DIR/compare_results_<splits>.csv --ci 0.95
uv run python paper/paper_plots.py           # outputs_paper_final/ -> plots/
uv run python paper/paper_tables.py          # outputs_paper_final/ -> tables/
```

There is no test suite. To check a config change without running a sweep,
load the runner, call `compare._validate_config` and `compare._build_trial_specs`
on the config, and confirm the resolved B1 values and trial specs.

## Layout

- `src/compare.py` is the entry point. It resolves the config, builds one trial
  spec per (B, B1, optimization mode, baseline, step schedule), runs the trials,
  caches each run in `<output_dir>/runs/<config_id>/runs.jsonl`, and writes the
  summary CSVs.
- `src/adaptive.py`: the two-phase method. Phase 1 uses pilot paths and the
  CrossFit-Q variance estimator (KS); the minimax allocation problem is solved
  here too. `src/phase1_mmd.py` has the sibling-pilot Phase 1 for MMD.
  `src/uniform_c.py` implements the Uniform-c and Learned-c restrictions.
- `src/runners/splitting.py` simulates the splitting tree; `src/runners/trees.py`
  rounds a relaxed allocation to a mixture of exact integer (dyadic) trees.
- `src/baselines.py` (fixed-N and solver baselines), `src/trials.py` (seeding,
  chunking, parallel runs), `src/metrics/` (KS and random-Fourier MMD),
  `src/reference_cache.py` (reference samples under `checkpoints/<runner>/`).
- `src/runners/<family>/` holds one sampler family (`sde`, `edm`, `ddpm`); each
  case below it has a `runner.py` and a `configs.py` exporting `CONFIGS`.
- `paper/` rebuilds the paper's figures and tables from `outputs_paper_final/`.

## Adding a runner or config

- New config: add a named entry to the runner's `CONFIGS` dict. Build on the
  shared helpers in `runners/common_configs.py` (`split_schedules`,
  `crossfit_q_config`, `mmd_sibling_config`) rather than copying values.
- New runner: subclass the family runner (or `runners.base.BaseRunner`), set
  `runner_name` and `config_module`, and add it to the family's
  `*_RUNNER_CLASSES` in `runners/<family>/__init__.py`; `runners/registry.py`
  collects those.
- SDE cases: set `terminal_time` only through `sde_case_configs` in
  `runners/sde/configs.py`, which writes it to all three places that read it.
- `step_schedules` must have an entry for every value in `B_list`.

## Rules

- Do not weaken validation in `compare.py` or in `paper/`. The paper scripts
  check that run counts and settings match the paper; fix the data, not the check.
- Results are cached by config identity. Changing a config value creates new
  runs rather than overwriting old ones; delete an output directory only on
  purpose.
- Keep seeding deterministic (`trials.py` seed offsets); results must be
  reproducible from the config alone.
- Match the surrounding code: plain functions, explicit `ValueError`s with the
  offending value, no new dependencies without need.
- This repository is anonymized for review. Do not add names, emails, user or
  absolute paths, cluster or scheduler details, or links to non-anonymous
  repositories, in code, comments, configs or docs.
