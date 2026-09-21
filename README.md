# colab-jax

Colab notebooks for the AI GDE **Marathon: JAX on TPU Tutorial**. Apache 2.0.

Two notebooks:

| Notebook | What it teaches | Needs a token? |
| --- | --- | --- |
| [`01_jax_tpu_mechanics`](notebooks/01_jax_tpu_mechanics.ipynb) | Buffer donation, `jit` retracing and static shapes, cached decode, int8 KV — measured on synthetic arrays | No |
| [`02_serving_gemma4_on_tpu`](notebooks/02_serving_gemma4_on_tpu.ipynb) | The same mechanics spent on a real `gemma-4-E2B-it-qat-w4a16-ct` checkpoint, loaded without PyTorch | Yes (gated Gemma) |

The measurements come from single-chip TPU rigs in `~/tpu-jax` and
`~/gemma4-dev`. Colab's TPU runtimes are **v5e-1 and v6e-1** — the same silicon
those rigs used — so the findings are re-measured live in the notebook rather
than quoted at the reader. These notebooks target **v5e-1** (one `TPU v5 lite`,
16.91 GB HBM, JAX 0.7.2 as of 2026-09-20); the source rigs measured on v6e-1,
and citations say so. See [`docs/OUTLINE.md`](docs/OUTLINE.md) for the
section-by-section source map.

## Authoring model

**Notebooks are build artifacts. Edit `src/*.py`, never `notebooks/*.ipynb`.**

Sources are [jupytext](https://jupytext.readthedocs.io) `py:percent` files, so
a prose change is a one-line diff instead of a re-encoded JSON blob. The build
prepends the Apache-2.0 cell and the "Open in Colab" badge, and sets
`metadata.accelerator = "TPU"` so the reader lands on a TPU runtime instead of
silently benchmarking a CPU.

The `.ipynb` files *are* committed, because that is what Colab serves from
GitHub. `make check` fails if they have drifted from their sources.

```bash
make setup     # jupytext, nbformat, google-colab-cli
make build     # src/*.py -> notebooks/*.ipynb
make check     # fail if committed notebooks are stale (CI runs this)
make verify    # run every notebook on a real Colab TPU; fail on any error
```

## Verification

`make verify` is the gate. Rendered markdown will not tell you that cell 14
raises, so every notebook is executed on the accelerator it claims before it
ships, via the [Colab CLI](https://github.com/googlecolab/google-colab-cli):

```
colab --auth adc new -s verify-01 --tpu v5e1
colab --auth adc exec -s verify-01 -f notebooks/01_jax_tpu_mechanics.ipynb
colab --auth adc stop -s verify-01
```

`colab exec` writes the executed copy back as `*_output.ipynb`;
`tools/verify_on_tpu.py` scans it for `output_type == "error"` and unexpected
stderr, then files it under `runs/<date>/` as evidence.

`--auth adc` reuses your gcloud Application Default Credentials. The CLI's own
default is `oauth2`, which blocks on a pasted browser code and cannot be
scripted; ADC must be a user credential, since a Colab runtime belongs to a
Google user and bills against that user's compute units.

`make verify-v6e1` targets the 32 GB chip, but v6e-1 needs a Colab tier this
account does not have — it returns `Backend rejected accelerator 'V6E1'`.

These runs provision real hardware and spend compute units. `make sessions`
lists anything still up; `colab stop -s <name>` releases it.

## Two Colab tools, two jobs

- **[`google-colab-cli`](https://github.com/googlecolab/google-colab-cli)** —
  headless. Provisions `--tpu v5e1|v6e1`, executes a local `.ipynb`, tears
  down. This is the build and CI half, and the only one used by `make verify`.
- **[`colab-mcp`](https://github.com/googlecolab/colab-mcp)** — *not* a
  headless builder. It proxies over a websocket to a notebook already open in
  your browser, exposing one tool (`open_colab_browser_connection`) until the
  page hands over the editing tools. Good for live iteration on one notebook
  against a real TPU; wrong tool for building two deterministically.

Both want `uv`, which is not installed on this machine:

```bash
pip install uv && uv tool install google-colab-cli
```

## Layout

```
src/         py:percent sources — the thing you edit
notebooks/   generated .ipynb — committed, served to Colab from GitHub
tools/       build_notebooks.py, verify_on_tpu.py
docs/        OUTLINE.md: section-by-section source map and known gaps
runs/        executed notebooks kept as evidence that a version ran clean
```
