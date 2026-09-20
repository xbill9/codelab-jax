---
name: new-notebook
description: Scaffold a new notebook in this repo - creates the py:percent source in src/, registers it in notebooks.json with an accelerator, and builds it. Use when adding a notebook to the JAX-on-TPU series.
disable-model-invocation: true
---

Scaffold a new notebook. `$ARGUMENTS` is the notebook name (e.g. `03_sharding_on_tpu`).

These three steps are coupled — a source with no `notebooks.json` entry is silently skipped by the
build, and an entry with no source fails it. Do all three.

## 1. Pick the number and name

Read `notebooks.json` for the existing entries and continue the numbering. The stem must be
`NN_snake_case`, matching the existing files. If `$ARGUMENTS` is empty or has no number, ask.

## 2. Create `src/<stem>.py`

Start from the header block that `src/01_jax_tpu_mechanics.py` uses:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---
```

Then a `# %% [markdown]` title cell and the section skeleton. Follow the house style of the existing
sources: each section states what it teaches, names the source file in `~/tpu-jax` it draws from, and
lists the beats to hit. Leave `# TODO` in the code cells rather than inventing content.

Do **not** add an Apache-2.0 header or a Colab badge — `tools/build_notebooks.py` injects both.

## 3. Register it in `notebooks.json`

Add to the `notebooks` object:

```json
"<stem>.py": { "title": "...", "accelerator": "TPU", "tpu": "v6e1" }
```

Use `"tpu": "v5e1"` if the notebook is specifically about the 16 GB chip's constraints. Set
`"accelerator": "None"` only if the notebook genuinely does not need an accelerator.

## 4. Build and confirm

Run `make build`, then `make check` to confirm the build is reproducible. Report the new notebook's
cell count and its Colab URL:

`https://colab.research.google.com/github/<repo>/blob/main/notebooks/<stem>.ipynb`

Do not run `make verify` — a skeleton of TODO cells has nothing to verify.
