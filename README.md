# MDE Pipeline CLI Guide

This repository exposes a Typer-based CLI entrypoint named `mde` (`src.mde_pipeline.cli:app`).

## 1) Setup and invocation

## Install dependencies

```bash
pip install -e .
```

If your environment does not install script entrypoints, you can also run:

```bash
python -m src.mde_pipeline.cli <command> [options]
```

## Inspect commands

```bash
mde --help
mde <command> --help
```

> Notes
> - The CLI is defined in `src/mde_pipeline/cli.py`.
> - Default output paths are centralized in `src/mde_pipeline/utils/paths.py`.

---

## 2) End-to-end pipeline order

Typical order:

1. `preprocess` — read raw maps and standardize/smooth/mask them into processed HDF5.
2. `regions` — construct region masks from processed maps.
3. `simulate` — generate simulated sky maps (optionally with gains/noise).
4. `fit` — run Fisher and/or MCMC per region.
5. `fisher` — canonical Fisher forecasts (gain-marginalized, prior-aware, dataset-set comparisons).
6. Optional utilities:
   - `combine-cbass-spass`
   - `smooth-cmb`
   - `figures` (stub)
   - `run` (recipe/orchestration mode)

---

## 3) Command-by-command usage

## `preprocess`

```bash
mde preprocess \
  --instruments configs/preprocessing/instruments.yaml \
  --config configs/preprocessing/preprocessing.yaml \
  --tag v001
```

What it does:
- Loads instrument metadata (`instruments.yaml`) keyed by `map_id`.
- Loads per-map operation lists from preprocessing YAML.
- Reads each map via the `read_class` reader in the instrument metadata.
- Applies ops in sequence (supported ops include `fix_pol_convention`, `unit_convert`, `smooth_maps`, `dec_mask`, and factory op `subtract_cmb`).
- Writes processed maps to `products/processed_maps/<tag>/processed_maps.h5` unless `--out` is passed.

Useful flags:
- `--only <map_id>` (repeatable) to process selected maps.
- `--dry-run` to print actions without writing.
- `--overwrite` to overwrite output.

How to modify behavior:
- Add/remove instruments in `configs/preprocessing/instruments.yaml`.
- Edit operation chain per map in `configs/preprocessing/preprocessing.yaml`.

## `regions`

```bash
mde regions \
  --config configs/regions/gal_plus_north_south.yaml \
  --tag v001
```

What it does:
- Loads region definitions under top-level `regions`.
- Builds masks using registered mask functions (`percentile_threshold`, `regions_minus_masks`).
- Writes a region map and metadata to `products/regions/<tag>/regions.h5` unless `--out` is passed.
- Optionally writes QC region plots.

Useful flags:
- `--processed` to point at a specific processed HDF5.
- `--dry-run` for plan-only execution.

How to modify behavior:
- Edit/add region config files under `configs/regions/`.
- Adjust `group_name`, `masks.*.type`, and `masks.*.kwargs`.

## `simulate`

```bash
mde simulate --config configs/simulations/simulations.yaml
```

What it does:
- Loads `simulations` config block.
- Reads targets from `processed_h5`.
- Loads templates and evaluates configured emission components per target frequency.
- Optionally applies gain and/or noise.
- Writes outputs to `out_h5` in the simulation YAML.
- Writes `gains.json` and a spectrum plot in the simulations output folder.

How to modify behavior:
- Edit `configs/simulations/simulations.yaml`:
  - `targets`
  - `templates`
  - `components`
  - `gain.enabled` / `noise.enabled`
  - `processed_h5` / `out_h5`

## `fit`

```bash
mde fit \
  --config configs/fitting/fitter.yaml \
  --data configs/fitting/data.yaml \
  --templates configs/templates/templates.yaml \
  --tag v001 \
  --run-name paper_main
```

What it does:
- Loads fit model config (`fitter` block): targets, components, priors, outputs, mode flags.
- Loads regions from `regions_h5` (`products/regions/<tag>/regions.h5` by default).
- Loads map targets from processed/simulation HDF5.
- Builds region-sliced datasets and templates.
- Depending on mode flags, runs:
  - Fisher matrix estimation,
  - MCMC (`emcee`) sampling,
  - post-processing from saved samples.
- Writes region-wise results under `products/fits/<tag>/<run-name>/` unless overridden.

Useful flags:
- `--region` (repeatable) to run only selected region IDs.
- `--out-dir` for custom fit root output directory.

How to modify behavior:
- Edit `configs/fitting/fitter.yaml`:
  - `targets`
  - `components` and their `params_map`/`fixed_params`/`init`/`priors`
  - optional fit-Stokes selection via `fitter.stokes_fit` (e.g. `["Q", "U"]` to exclude intensity)
  - optional mode block: `fitter.modes.run_fisher`, `run_mcmc`, `run_postprocess`
  - optional curved thermal dust component `cls_name: ThermalDustCurved` with params `beta_d`, `T_d`, and curvature `c`

## `fisher` (standalone)

```bash
mde fisher \
  --config configs/fitting/fitter.yaml \
  --data configs/fitting/data.yaml \
  --templates configs/templates/templates.yaml \
  --tag v001 \
  --run-name paper_fisher
```

What it does:
- Uses the same canonical Fisher engine as fitting (`src/mde_pipeline/fitting/fisher.py`), including gain-marginalized Fisher + Gaussian prior contributions.
- Supports config-level `fitter.dataset_sets`, where each set defines a target list (e.g. `baseline`, `baseline_plus_litebird`).
- Runs all configured dataset sets in one command and writes side-by-side outputs for each region:
  - `sigma_1d` (per-parameter),
  - full covariance,
  - magnetic-dust SNR metric (`|theta_fid| / sigma`) for params with names like `A_md` / `phi`.
- Writes per-set files under `.../fisher/<dataset_set>/<region>/` and a consolidated comparison file at `.../fisher/dataset_set_comparison.json`.

When to use:
- Use `mde fit` for integrated fit workflows.
- Use `mde fisher` for canonical Fisher-only comparisons across dataset sets.

## `combine-cbass-spass`

```bash
mde combine-cbass-spass \
  --config configs/combine_cbass_spass/combine.yaml \
  --instruments configs/preprocessing/instruments.yaml \
  --tag v001
```

What it does:
- Reads C-BASS and S-PASS raw maps.
- Combines them into one map using config parameters (`sync_beta`, mask settings, target beam).
- Writes FITS to `products/combined/<tag>/combined_cbass_spass.fits` by default.

## `smooth-cmb`

```bash
mde smooth-cmb \
  --target-fwhm-arcmin 60 \
  --instruments configs/preprocessing/instruments.yaml \
  --tag v001
```

What it does:
- Reads `cmb_commander` from instruments config.
- Smooths it to target FWHM.
- Writes FITS to `products/cmb/<tag>/smoothed_cmb_map_fwhmXX.Xarcmin.fits`.

## `figures`

`figures` is currently a stub (`NotImplementedError`) in the CLI and needs implementation hookup.

## `run` recipe mode

```bash
mde run --config path/to/recipe.yaml
```

Runs multiple steps from one YAML recipe (`run_id`, `configs`, `paths`, and `steps`).

---

## 4) YAML files and when to use each

Core YAMLs:
- `configs/preprocessing/instruments.yaml` — map catalog and reader metadata.
- `configs/preprocessing/preprocessing.yaml` — operation chain per `map_id`.
- `configs/regions/*.yaml` — region/mask definitions.
- `configs/simulations/simulations*.yaml` — simulation model, targets, templates, output paths.
- `configs/templates/templates.yaml` — template aliases for fitting.
- `configs/fitting/fitter*.yaml` — fitting model definition, priors, targets, output behavior.
- `configs/fitting/data.yaml` — map source metadata used by fitting.
- `configs/combine_cbass_spass/combine.yaml` — C-BASS/S-PASS merge parameters.

Variant config sets you can choose from:
- `configs/simulations/simulations.yaml`
- `configs/simulations/simulations_sytdffsp.yaml`
- `configs/simulations/simulations_sytdffspmd.yaml`
- `configs/simulations/simulations_sytdffspmd_litebird.yaml`
- `configs/fitting/fitter.yaml`
- `configs/fitting/fitter_sytdffsp.yaml`
- `configs/fitting/fitter_sytdffspmd.yaml`
- `configs/fitting/fitter_sytdffspmd_litebird.yaml`

---

## 5) Safe customization workflow

1. Copy an existing YAML variant (e.g., `fitter.yaml` -> `fitter_mytest.yaml`).
2. Change only one section at a time (targets, then components, then priors).
3. Run command with `--dry-run` first.
4. Execute a small subset (`--only` for preprocess, `--region` for fit).
5. Scale to full target list once outputs look correct.

---

## 6) Practical caveats

- The `figures` command is intentionally not implemented yet.
- Some CLI defaults reference filenames that may not exist in this repo; pass explicit `--config` paths to avoid ambiguity.
- Use a consistent `--tag` across preprocess/regions/simulate/fit so default paths line up.



## 7) Minimal command sequence: simulation grid → Fisher dataset-set comparison

```bash
# 1) Build a simulation grid (example variant with magnetic dust + LiteBIRD)
mde simulate --config configs/simulations/simulations_sytdffspmd_litebird.yaml

# 2) Run canonical Fisher once; it executes all dataset_sets from fitter config
mde fisher \
  --config configs/fitting/fitter_sytdffspmd_litebird.yaml \
  --data configs/fitting/data.yaml \
  --templates configs/templates/templates.yaml \
  --tag v001 \
  --run-name fisher_dataset_compare

# 3) Inspect combined summary
cat products/fits/v001_sytdffspmd03/fisher_dataset_compare/fisher/dataset_set_comparison.json
```


## 8) Quick test case: Fisher then MCMC (single region)

Use this as a smoke test after generating one simulation realization.

```bash
# Fisher-only quick run
python -m src.mde_pipeline.cli fit   --config configs/fitting/fitter_sytdffspmd.yaml   --data configs/fitting/data.yaml   --templates configs/templates/templates.yaml   --tag v001   --run-name smoke_fisher   --region <region_name>   --overwrite
```

Set in your fitter YAML:

```yaml
fitter:
  modes:
    run_fisher: true
    run_mcmc: false
```

Then run MCMC on the same setup:

```bash
python -m src.mde_pipeline.cli fit   --config configs/fitting/fitter_sytdffspmd.yaml   --data configs/fitting/data.yaml   --templates configs/templates/templates.yaml   --tag v001   --run-name smoke_mcmc   --region <region_name>   --overwrite
```

with:

```yaml
fitter:
  modes:
    run_fisher: false
    run_mcmc: true
```

## 9) Diagnostics and per-run tabular outputs

The pipeline writes diagnostics per region automatically:

- MCMC diagnostics: `corner.png`, `trace.png`, `resid_hist.png`, spectrum plot, and residual map products in each region folder.
- Fisher diagnostics: `regions/<region>/fisher/fisher.npz`, `regions/<region>/fisher/corner.png`, and `regions/<region>/fisher/summary.json`.

In addition, each fit run now writes a run-level CSV:

- `products/fits/<tag>/<run_name>/run_summary.csv`

This CSV includes, per region/mode row:

- `run_id`, `region`, `mode`
- quality-of-fit stats (`chi2`, `red_chi2`, `AIC`, `BIC`) for MCMC rows
- Fisher 1-sigma values (`sigma_<param>`)
- fiducial parameter values (`fid_<param>`) and best-fit values (`best_<param>`)

This makes it straightforward to merge baseline and baseline+LiteBIRD runs and compare fit quality and MDE detectability in one table.
