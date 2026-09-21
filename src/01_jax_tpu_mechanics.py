# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # JAX on TPU: donation, static shapes, and cached state
#
# Three things decide whether a JAX program is fast on a TPU, and none of them
# are the model. This notebook measures all three on the TPU runtime you are
# connected to right now.
#
# By the end you will be able to explain:
#
# 1. **Why `jit` recompiles** when your input shape changes by one token, and
#    what padding to a bucket costs and does not cost.
# 2. **What `donate_argnums` does to a KV cache** — the single largest speedup
#    in the work this notebook is drawn from, and why the number you measure
#    depends as much on your benchmark as on your chip.
# 3. **How to carry mutable state through a pure function** with
#    `lax.dynamic_update_slice`, and how to prove the cached path agrees with a
#    full re-forward.
#
# Everything runs on one chip with synthetic arrays. No checkpoint, no gated
# model, no Hugging Face token. Notebook 02 does the real thing.

# %% [markdown]
# ## Setup
#
# SOURCE: adapt the device/version preamble from `~/tpu-jax/ports/gemma4/
# jax_e_smoke_test.py`. Assert we are actually on a TPU and print the chip
# generation, because every number below is meaningless on CPU.

# %%
import jax
import jax.numpy as jnp

print("JAX", jax.__version__)

devices = jax.devices()
dev = devices[0]
print(f"{len(devices)} device(s): {dev.device_kind} ({dev.platform})")

if dev.platform != "tpu":
    raise RuntimeError(
        "This notebook needs a TPU runtime. In Colab: Runtime > Change runtime type > Hardware accelerator: TPU"
    )

stats = dev.memory_stats() or {}
print(f"HBM: {stats.get('bytes_limit', 0) / 1e9:.2f} GB")

# %% [markdown]
# ## 1. Why your jitted function keeps recompiling
#
# SOURCE: the 128-aligned bucket padding in `~/tpu-jax/ports/gemma4/
# jax_e_model.py`, and the "padding to 128-aligned TPU buckets does not change
# model output" claim verified in `~/tpu-jax/tests/test_chunked_prefill.py`.
#
# Beats to hit:
# - Trace a trivial function over sequence lengths 5, 6, 7 -> three compiles.
# - Show the compile cost with `jax.block_until_ready` and a timer.
# - Introduce bucketing to 128. Three lengths, one compile.
# - Then the part tutorials skip: show the padded result is *bitwise identical*
#   to the unpadded one, so bucketing is free correctness-wise and only costs
#   wasted FLOPs.

# %%
# TODO

# %% [markdown]
# ## 2. Buffer donation
#
# SOURCE: `~/tpu-jax/tests/test_donation.py` and section 1 of
# `~/tpu-jax/benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md`.
#
# `lax.dynamic_update_slice` does not write into an array. It **produces a new
# one**. So a decode step that appends a single token to the KV cache reads the
# whole cache, writes a whole copy, and reads that to attend — every step, for
# one token.
#
# `donate_argnums` tells `jit` that an argument is dead after the call, so XLA
# may reuse its buffer for the output instead of allocating a second one. The
# copy disappears.
#
# Measured in the source engine on a v6e-1 at ctx 8192, B=32: **1.62x on a bf16
# cache**, 1.22x on int8. The cells below will report a much larger number, and
# the section after them explains why — the answer is about how the benchmark is
# built, not about your chip.
#
# There is a cost. A donated buffer is **invalidated** by the call that
# consumes it. Keeping a reference to a cache you passed in and reading it
# afterwards is a bug, and the third cell shows what that looks like.

# %%
import statistics
import time

# A KV cache shaped like a small model's: (layers, batch, heads, seq, head_dim).
# The model is irrelevant here -- what matters is the size of the buffer being
# copied. 512 MiB per cache is large enough that the copy costs milliseconds and
# small enough to leave headroom on a 16 GB chip.
L, B, H, S, Dh = 8, 8, 8, 4096, 128
DTYPE = jnp.bfloat16

cache_bytes = L * B * H * S * Dh * jnp.dtype(DTYPE).itemsize
print(f"cache {L}x{B}x{H}x{S}x{Dh} {jnp.dtype(DTYPE).name}")
print(f"  {cache_bytes / 2**20:.0f} MiB each, {2 * cache_bytes / 2**20:.0f} MiB for k+v")

# The single token appended each step.
new_k = jnp.ones((L, B, H, 1, Dh), dtype=DTYPE)
new_v = jnp.ones((L, B, H, 1, Dh), dtype=DTYPE)


def fresh_cache():
    zeros = jnp.zeros((L, B, H, S, Dh), dtype=DTYPE)
    return zeros, zeros + 1


def decode_step(cache_k, cache_v, k_tok, v_tok, pos):
    """Append one token to the cache, then read it the way attention would."""
    cache_k = jax.lax.dynamic_update_slice(cache_k, k_tok, (0, 0, 0, pos, 0))
    cache_v = jax.lax.dynamic_update_slice(cache_v, v_tok, (0, 0, 0, pos, 0))
    out = (cache_k[..., :1, :] * cache_v[..., :1, :]).sum(dtype=jnp.float32)
    return cache_k, cache_v, out


def timed(step, n=20):
    """Median step time in ms, plus every step's output for the parity check."""
    cache_k, cache_v = fresh_cache()
    # First call compiles; do not time it.
    cache_k, cache_v, out = step(cache_k, cache_v, new_k, new_v, jnp.int32(0))
    jax.block_until_ready((cache_k, cache_v, out))

    times, outs = [], []
    for i in range(n):
        start = time.perf_counter()
        cache_k, cache_v, out = step(cache_k, cache_v, new_k, new_v, jnp.int32(i + 1))
        jax.block_until_ready((cache_k, cache_v, out))
        times.append(time.perf_counter() - start)
        outs.append(out)
    return statistics.median(times) * 1e3, jnp.stack(outs)


plain_step = jax.jit(decode_step)
donated_step = jax.jit(decode_step, donate_argnums=(0, 1))

# %% [markdown]
# `peak_bytes_in_use` is a high-water mark and never goes down, so run the
# donated path **first**. Any rise afterwards belongs to the non-donated run.


# %%
def peak_mib():
    return (dev.memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**20


peak_start = peak_mib()
donated_ms, donated_outs = timed(donated_step)
peak_after_donated = peak_mib()
plain_ms, plain_outs = timed(plain_step)
peak_after_plain = peak_mib()

print(f"donated      {donated_ms:7.3f} ms/step")
print(f"not donated  {plain_ms:7.3f} ms/step")
print(f"speedup      {plain_ms / donated_ms:7.2f}x")
print()
print(f"peak HBM before either run    {peak_start:8.0f} MiB")
print(f"peak HBM after donated run    {peak_after_donated:8.0f} MiB")
print(f"peak HBM after plain run      {peak_after_plain:8.0f} MiB")
print(f"  the non-donated path added  {peak_after_plain - peak_after_donated:8.0f} MiB")
print(f"  one cache is               {cache_bytes / 2**20:8.0f} MiB")

# Donation is a scheduling change. If it alters a single value, it is a bug.
assert jnp.array_equal(donated_outs, plain_outs), "donation changed the result"
print("\noutputs identical across all steps: True")

# %% [markdown]
# ### That number is too good, and the reason matters
#
# The engine this came from measured **1.62x**. You just measured something far
# larger. The chip is not the reason — the benchmark is.
#
# `decode_step` reads a single token back out of the cache. So the copy is
# essentially the whole step, and removing it removes essentially the whole
# step. Real attention reads the **entire** cache every step. Against that, the
# copy is one cost among several rather than the only one.
#
# Change one line — read the whole cache instead of one token — and watch the
# ratio fall:


# %%
def decode_step_full_read(cache_k, cache_v, k_tok, v_tok, pos):
    """Same append, but attention-shaped: reads the whole cache back."""
    cache_k = jax.lax.dynamic_update_slice(cache_k, k_tok, (0, 0, 0, pos, 0))
    cache_v = jax.lax.dynamic_update_slice(cache_v, v_tok, (0, 0, 0, pos, 0))
    out = (cache_k * cache_v).sum(dtype=jnp.float32)
    return cache_k, cache_v, out


donated_full_ms, _ = timed(jax.jit(decode_step_full_read, donate_argnums=(0, 1)))
plain_full_ms, _ = timed(jax.jit(decode_step_full_read))

print(f"one-token read   {plain_ms / donated_ms:6.2f}x   (copy is the whole step)")
print(f"full-cache read  {plain_full_ms / donated_full_ms:6.2f}x   (copy competes with attention)")
print("engine, measured   1.62x   (copy also competes with every weight matmul)")

# %% [markdown]
# Three numbers, same optimization, one real answer. A microbenchmark that
# isolates a cost will always overstate what removing it buys, and the more
# completely it isolates it, the more it overstates.
#
# This is worth carrying into the rest of the notebook: when a cell reports a
# speedup far above what the source measured, suspect the benchmark before
# celebrating the result.

# %% [markdown]
# ### The hazard
#
# The buffer you donate is gone. JAX deletes it, so a stale reference raises
# rather than quietly reading a recycled buffer — which is the good outcome, and
# the reason donation is safe to use at all.

# %%
cache_k, cache_v = fresh_cache()
stale = cache_k  # same buffer we are about to hand over

cache_k, cache_v, _ = donated_step(cache_k, cache_v, new_k, new_v, jnp.int32(0))
jax.block_until_ready(cache_k)

try:
    jax.block_until_ready(stale + 0)
    print("UNEXPECTED: the donated buffer was still readable")
except Exception as exc:
    print(f"reading the donated buffer raises, as it must:\n  {type(exc).__name__}: {exc}")

# %% [markdown]
# One consequence worth knowing, from the source repo's own correction history:
# every benchmark in that engine built its `jit` without donation, so each
# step-time ratio measured before 2026-07-29 was taken on the copying path. When
# donation was turned on, the "int8 KV is 1.2-1.8x faster" claim shrank to about
# 1.18x — int8 had been getting credit for halving the bytes of a copy that
# should never have existed.
#
# Two optimizations, one measurement, and the wrong one was being paid.

# %% [markdown]
# ## 3. Cached decode, and proving it correct
#
# SOURCE: `~/tpu-jax/tests/test_kv_cache_parity.py` — "cached decode matches
# full-sequence re-forward within float32 tolerance".
#
# Beats to hit:
# - A toy attention block, run two ways: full re-forward over the whole prefix
#   each step, vs. append-to-cache with `lax.dynamic_update_slice`.
# - Assert `allclose`. This is the test that catches an off-by-one in the write
#   index, which is the most common way a hand-rolled cache goes wrong.
# - Plot tokens/s for both as sequence length grows: the re-forward curve is
#   quadratic, the cached one flat.

# %%
# TODO

# %% [markdown]
# ## 4. The decode budget is a constant
#
# SOURCE: section 1 of `2026-07-29-kv-quant-v6e1/REPORT.md` ("The decode budget
# is a constant, to 0.0%") and section 2c of `2026-07-28-jax-e2b-v6e1/REPORT.md`
# ("we are memory-limited, not latency-limited").
#
# The point: single-token decode moves the whole KV cache through memory every
# step, so it is bandwidth-bound, not compute-bound. Derive bytes/token, divide
# by the chip's HBM bandwidth, compare to the measured step time. When those
# two agree you have understood the machine.
#
# This is also the honest framing for why the int8 KV cache in the next section
# helps: it is a bandwidth cut, not a math cut.

# %%
# TODO

# %% [markdown]
# ## 5. An int8 KV cache
#
# SOURCE: `~/tpu-jax/tests/test_quantized_kv.py` and section 2 of
# `2026-07-29-kv-quant-v6e1/REPORT.md` — 1.17-1.19x faster than donated BF16,
# and 1.82-1.98x the capacity.
#
# Beats to hit:
# - Per-head scale, quantize on write, dequantize on read.
# - Verify against a dequantize-first reference (that is the real test).
# - Measure both axes: the speedup *and* the capacity, since capacity is the
#   bigger win and the one people forget to measure.
# - Quality: the source work measured 28.41 vs 28.73 perplexity and 97.08%
#   greedy-token agreement over 583 steps. We cannot redo that here without a
#   checkpoint -- state the number, cite notebook 02, do not re-derive it.

# %%
# TODO

# %% [markdown]
# ## 6. A note on multiple chips
#
# Colab gives you one TPU chip (v5e-1 or v6e-1), so `jax.sharding` and
# `shard_map` cannot be *measured* here. Rather than hand-wave, show the API on
# emulated devices and label it as emulation:
#
#     XLA_FLAGS=--xla_force_host_platform_device_count=8
#
# Build a `Mesh`, shard an array, show the partition spec, and say plainly that
# the numbers are CPU and the point is the API surface. Then point at the real
# multi-chip path (v6e-8 on GCE or GKE).
#
# DECISION: keep this section short, or drop it if a Marathon teammate takes
# sharding as their own notebook. Do not fake a multi-chip result.

# %%
# TODO

# %% [markdown]
# ## What to take away
#
# TODO: restate the three mechanics, and hand off to notebook 02, which applies
# all of them to a real Gemma 4 checkpoint.
