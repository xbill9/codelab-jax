# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Serving Gemma 4 E2B on a single TPU with pure JAX
#
# Notebook 01 measured three TPU mechanics on synthetic arrays. This one spends
# them on a real checkpoint: load `gemma-4-E2B-it-qat-w4a16-ct` from safetensors
# with **no PyTorch in the path**, run it in JAX on one chip, and generate text.
#
# The honest framing, which belongs near the top rather than buried at the end:
# this is an inspectable reference implementation, not a vLLM replacement. It
# exists because the tested vLLM TPU stack could not load this QAT export, and
# building the missing path is a good way to see what a serving stack actually
# does.
#
# Prerequisites: notebook 01, a TPU runtime, and a Hugging Face token with
# access to the gated Gemma repo.

# %% [markdown]
# ## 0. Runtime and access
#
# Two things to check before anything expensive happens.
#
# **The chip.** These notebooks target **v5e-1** — one `TPU v5 lite`, 16.91 GB
# HBM. v6e-1 has 32 GB and is what the source rigs measured on, so a reader who
# has one will see more headroom than the text assumes. The w4a16 checkpoint is
# 8.32 GB on disk, so it fits either, but the KV budget differs sharply and
# every capacity number below should be read off the reader's own chip.
#
# **The token.** Gemma is gated. Use Colab's secrets panel
# (`google.colab.userdata`), never a pasted literal — this notebook ships to
# GitHub and a pasted token ships with it.

# %%
# TODO: detect device_kind, print HBM from memory_stats(); read HF_TOKEN from
# userdata.get('HF_TOKEN') with a clear error pointing at the key icon.

# %% [markdown]
# ## 1. What is actually in the checkpoint
#
# SOURCE: `~/tpu-jax/docs/gemma4-quirks.md` (15 findings, each verified against
# the reference implementation).
#
# Do not dump all fifteen. Pick the four that change the code you are about to
# write, and show each one as a live assertion against the downloaded tensors
# rather than as prose:
#
# - **`attention_k_eq_v`** (quirk 12): V *is* K on full-attention layers, and
#   the checkpoint ships no `v_proj` at all. If you assume it exists you get a
#   `KeyError` and no idea why.
# - **Per-Layer Embeddings** (quirk 8): 16% of the model by parameter count,
#   and a gather rather than a matmul — which is why quantizing it is nearly
#   free and buys almost nothing in throughput.
# - **Sandwich norms** (quirk 2) and **`layer_scalar`** (quirk 3), which scales
#   the whole residual stream.
# - **The tokenizer does not add BOS** (quirk 10, flagged as a trap).
#
# Then link the full quirks doc for the other eleven.

# %%
# TODO

# %% [markdown]
# ## 2. safetensors to a JAX pytree, without PyTorch
#
# SOURCE: `~/tpu-jax/ports/gemma4/jax_e_loader.py` (221 lines — small enough to
# walk through almost in full).
#
# Beats to hit:
# - `huggingface_hub.snapshot_download` with an allow-list, so the reader is not
#   pulling the vision and audio towers they will not use.
# - Read tensors with `safetensors.numpy`, not `safetensors.torch`. This is the
#   whole "no PyTorch" claim and it is one import.
# - Assemble a nested dict and treat it as a pytree; show
#   `jax.tree_util.tree_map` over it to check dtypes and total bytes.
# - Dequantize the w4a16 weights. Show the packing layout explicitly.
#
# This is the pytree chapter of a JAX course with a 10 GB payload instead of a
# dictionary of toy arrays.

# %%
# TODO

# %% [markdown]
# ## 3. One forward pass
#
# SOURCE: `~/tpu-jax/ports/gemma4/jax_e_model.py`. That file is 1,570 lines and
# must NOT be pasted in. Reduce it to the minimum that produces correct logits
# for a single prompt, and link the full file for the rest.
#
# Beats to hit:
# - RMSNorm as `x_normed * weight` (quirk 4 — no `+ 1`, which is where a port
#   from another Gemma generation silently goes wrong).
# - RoPE with the concatenated frequency layout and partial rotary by masking
#   (quirk 6).
# - Attention with no score softcap and `scaling = 1.0` (quirk 5).
# - KV sharing keyed by layer *type* (quirk 7).
# - Sanity check: greedy-decode a known prompt, eyeball that it is English.

# %%
# TODO

# %% [markdown]
# ## 4. Making it a generation loop
#
# Now apply notebook 01 directly. Same three mechanics, real weights:
#
# - Bucket the prefill to 128, cite the token-exactness test.
# - `donate_argnums` on the cache. Measure. Expect the ~1.6x shape again.
# - int8 KV. Measure the capacity change on the reader's own chip.
#
# SOURCE: `~/tpu-jax/jax_engine.py` (481 lines) for the stateful engine shape.
#
# Also worth showing, because it is short and it is the technique that lets a
# 7,679-token prompt fit: **chunked prefill**, verified token-exact against
# one-shot prefill in `~/tpu-jax/tests/test_chunked_prefill.py`.

# %%
# TODO

# %% [markdown]
# ## 5. What this measures, and what it does not
#
# SOURCE: `~/tpu-jax/benchmarks/runs/2026-07-29-real-http-v6e1/REPORT.md`.
#
# This is the most valuable section in the notebook and the one most tutorials
# do not have. The gap, measured on v6e-1:
#
# | | Result |
# | --- | --- |
# | Best static decode kernel | 2,888 aggregate tok/s (B=32, context 8,192) |
# | Real checkpoint over HTTP | ~139-141 aggregate tok/s |
#
# A 20x gap between a kernel number and a serving number, on the same chip with
# the same weights. The reason is structural, not a tuning problem: the server
# runs independent B=1 executions, so eight concurrent requests never form a
# device batch. At concurrency 2/4/8 aggregate throughput was
# 128.7/133.6/143.3 tok/s while median latency rose 497/952/1,775 ms — every
# request succeeded, none of them batched.
#
# Closing it needs a request batcher, batched KV ownership, continuous
# admission, and prefix reuse. That is what vLLM is, and saying so is a better
# lesson than pretending the kernel number is a serving number.
#
# Teach the reader to ask, of any benchmark: was this a kernel or a server?

# %%
# TODO: reproduce the single-stream context-scaling row on the reader's chip
# (506 / 2,045 / 7,679 prompt tokens), which is cheap, and *cite* the
# concurrency table rather than running an HTTP server inside Colab.

# %% [markdown]
# ## 6. Where to go next
#
# TODO: the correction history in
# `benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md` is worth linking as an
# example of how these numbers got walked back once (an artificial
# power-of-two capacity invariant, an undonated cache copy, a bad roofline
# accounting of the PLE gather). Retracted results are a teaching artifact.
