# Paper workflow

This directory contains the scripts that turn completed experiment results in the repository-root `outputs_paper_final/` into paper figures and table data.

From `code/`, run:

```bash
uv run python paper/paper_plots.py --outputs-root ../outputs_test --output-dir ../plots
uv run python paper/paper_tables.py --outputs-root ../outputs_test --output-dir ../tables
```

The plot script writes publication PNGs to the repository-root `plots/`. The table script writes CSVs to the repository-root `tables/`. Normal runs validate that the expected results and run counts are complete; do not weaken validation or use `--debug` for final paper artifacts.

## Updating `main.tex`

`main.tex` is at the repository root, one directory above `code/`.

The existing `\includegraphics` commands already use `plots/*.png`, so regenerating plots in the default location updates the manuscript figures. Compile LaTeX from the repository root so these paths resolve correctly.

Tables are not imported automatically. Treat the generated CSVs as the source of truth and copy their values into the matching table environments in `main.tex`:

- `complete_ou.csv` -> `tab:complete-ou`
- `complete_langevin.csv` -> `tab:complete-langevin`
- `complete_edm.csv` -> `tab:complete-edm`
- `complete_ddpm.csv` -> `tab:complete-ddpm`
- `ou_oracle_reductions.csv` -> `tab:ou-oracle-reductions`
- `numerical_schedules.csv` and `reference_samples.csv` -> their corresponding setup tables

For complete-result tables, copy the mean reduction and its two-sided 90% confidence interval into each cell. Arrange split counts as rows and budgets as columns. Use the largest available budget whenever the manuscript displays only one budget. Keep the existing rounding style unless precision matters to a claim.

After updating a table or figure, also update its caption and nearby prose. Search `main.tex` for stale numbers, `TO BE UPDATED`, old budget claims, and descriptions that no longer match the CSV semantics—especially whether OU-oracle values are measured KS reductions or variance-proxy reductions. Preserve existing labels unless references are intentionally changed, then compile and visually inspect the PDF.
