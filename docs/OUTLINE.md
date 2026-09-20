# Outline and source map

Two notebooks for the AI GDE **Marathon: JAX on TPU Tutorial** (deadline
2027-01-31, Apache 2.0, up to 3 collaborators per notebook).

Notebook 01 teaches the JAX/TPU mechanics on synthetic arrays with no
checkpoint and no token. Notebook 02 spends those mechanics on a real Gemma 4
checkpoint end to end. They are separable — a reader can do 01 alone — but 02
assumes 01.

## Why this material and not a generic JAX tutorial

Everything below was measured on v5e-1 and v6e-1 rigs in `~/tpu-jax*` and
`~/gemma4-dev`. **Colab's TPU runtimes are `v5e1` and `v6e1`** — the same
silicon — so unlike most ported benchmark prose, these numbers are things a
reader can re-measure in the notebook rather than take on faith.

**Target chip: v5e-1** (one `TPU v5 lite`, 16.91 GB HBM, JAX 0.7.2, verified
2026-09-20). v6e-1 is rejected on this account, so the numbers cited below —
taken on v6e-1's 32 GB — are the *source* results, not what a reader will see.
Ratios should carry; absolute capacity figures will not.

## Notebook 01 — JAX on TPU: donation, static shapes, and cached state

| § | Topic | Source | Headline number |
| --- | --- | --- | --- |
| 1 | `jit` retracing, 128-aligned buckets | `jax_e_model.py`, `tests/test_chunked_prefill.py` | padding does not change output |
| 2 | **Buffer donation** | `tests/test_donation.py`, kv-quant REPORT §1 | **1.60–1.62×**, ~2× resident tokens |
| 3 | Cached decode + parity test | `tests/test_kv_cache_parity.py` | matches re-forward to float32 tol |
| 4 | Decode is bandwidth-bound | kv-quant REPORT §1, jax-e2b REPORT §2c | budget constant "to 0.0%" |
| 5 | int8 KV cache | `tests/test_quantized_kv.py`, kv-quant REPORT §2 | **1.17–1.19×**, **1.82–1.98×** capacity |
| 6 | Sharding (emulated, short) | — none; see gap below | n/a |

§2 is the spine. It is the largest measured win in the source work, it is
under-covered in existing JAX tutorials, and it is cheap to demonstrate.

## Notebook 02 — Serving Gemma 4 E2B on a single TPU with pure JAX

| § | Topic | Source |
| --- | --- | --- |
| 0 | Chip + gated-token check | v6e-1 32 GB vs v5e-1 16 GB; w4a16 is 8.32 GB |
| 1 | Four checkpoint quirks that change your code | `docs/gemma4-quirks.md` (quirks 12, 8, 2/3, 10) |
| 2 | safetensors → JAX pytree, no PyTorch | `ports/gemma4/jax_e_loader.py` (221 lines) |
| 3 | One forward pass | `ports/gemma4/jax_e_model.py` (1,570 lines — reduce, do not paste) |
| 4 | Generation loop: donation, buckets, int8 KV, chunked prefill | `jax_engine.py` (481 lines) |
| 5 | **Kernel speed is not serving speed** | real-http REPORT |
| 6 | Reading a correction history | kv-quant REPORT "Corrections to earlier reporting" |

§5 is the differentiator: **2,888 tok/s** as a static decode kernel vs
**~139–141 tok/s** for the same weights on the same chip through HTTP, because
the server runs independent B=1 executions and concurrent requests never form a
device batch. A ~20× gap with a structural explanation is a better lesson than
another "look how fast TPUs are" notebook.

## Known gaps — decide before writing

- **No sharding anywhere.** `grep -rn 'jax.sharding|NamedSharding|pmap|shard_map|Mesh('`
  over `~/tpu-jax` returns nothing; every rig is single-chip. Colab gives one
  chip, so it cannot be measured there either. Either teach the API on
  `--xla_force_host_platform_device_count=8` and label it emulation, or let a
  Marathon teammate own sharding as a separate notebook. Do not fake a
  multi-chip result.
- **Pallas is deliberately out of scope** — being handled separately with
  Rubens. Notebook 01 §2–5 must not depend on it.
- **Gated checkpoint.** Notebook 02 needs an HF token via Colab secrets.
  Notebook 01 deliberately needs none, so it works for every reader.
- **v5e-1 headroom.** Every capacity claim should be read off the reader's own
  chip, not hardcoded — the target chip has half the HBM the source rigs did.

## Source repos

`~/tpu-jax` is the substantive one. `~/tpu-jax-{v5e1-2b,v6e1-2b,4b,12b,26b,31b,inf2}`
are near-identical forks — do not treat them as separate sources. Rig naming
follows `~/gemma4-dev/NAMING.md`.
