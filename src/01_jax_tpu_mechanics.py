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
# SOURCE: `TPUv6eHardwareProfile` and `pad_to_tpu_v6e_bucket` in
# `~/tpu-jax/ports/gemma4/jax_e_model.py`, and the pad-slot warning in the same
# file's windowed-mask docstring.
#
# `jit` compiles per *shape*, not per function. A prompt one token longer is a
# new shape, so it is a new compile — and in a server that means the first
# request at every length pays for a compile nobody asked for.
#
# The fix is to round the sequence length up to one of a few fixed buckets. The
# source engine uses `(64, 128, 256, 512, 1024, 2048, 4096, 8192)`, all
# 128-aligned because the MXU is a 128x128 systolic array.
#
# The part that is easy to get wrong is what padding costs you in *correctness*,
# and the answer is: nothing, but only if you carry the mask.

# %%
import time

VOCAB, EMBED = 1024, 8
emb_table = jax.random.normal(jax.random.PRNGKey(0), (VOCAB, EMBED), dtype=jnp.float32)

traces = 0


@jax.jit
def prefill(emb, tokens, mask):
    """Embed a prompt and reduce it, ignoring padded positions."""
    global traces
    # A side effect inside a jitted function runs at TRACE time, not call time.
    # That is exactly what makes it a compile counter.
    traces += 1
    x = emb[tokens]
    x = jnp.where(mask[..., None], x, 0.0)
    return x.sum(axis=1)


def run(n):
    tokens = jnp.ones((1, n), dtype=jnp.int32)
    mask = jnp.ones((1, n), dtype=jnp.bool_)
    start = time.perf_counter()
    jax.block_until_ready(prefill(emb_table, tokens, mask))
    return (time.perf_counter() - start) * 1e3


lengths = [100, 101, 102, 250]

traces = 0
print("first time at each length")
for n in lengths:
    ms = run(n)
    print(f"  len {n:4d}  {ms:8.1f} ms   traces: {traces}")

print("\nsame lengths again")
for n in lengths:
    ms = run(n)
    print(f"  len {n:4d}  {ms:8.1f} ms   traces: {traces}")

# %% [markdown]
# Four lengths, four compiles, and the second pass is free because those four
# shapes are now in the cache. A server sees far more than four lengths.
#
# Now round every length up to a bucket. This is `pad_to_tpu_v6e_bucket` from
# the source engine, reduced to its two moving parts — and note that it returns
# **two** things.

# %%
BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


def nearest_bucket(seq_len):
    for b in BUCKETS:
        if b >= seq_len:
            return b
    return (seq_len + 127) // 128 * 128


def pad_to_bucket(tokens, pad_token_id=0):
    """Right-pad to a bucket. Returns (padded_tokens, mask) -- never just tokens."""
    B, S = tokens.shape
    bucket = nearest_bucket(S)
    if bucket == S:
        return tokens, jnp.ones((B, S), dtype=jnp.bool_)
    pad_len = bucket - S
    padded = jnp.pad(tokens, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
    mask = jnp.concatenate(
        [jnp.ones((B, S), dtype=jnp.bool_), jnp.zeros((B, pad_len), dtype=jnp.bool_)],
        axis=1,
    )
    return padded, mask


traces = 0
print("bucketed")
for n in lengths:
    tokens, mask = pad_to_bucket(jnp.ones((1, n), dtype=jnp.int32))
    start = time.perf_counter()
    jax.block_until_ready(prefill(emb_table, tokens, mask))
    ms = (time.perf_counter() - start) * 1e3
    print(f"  len {n:4d} -> {tokens.shape[1]:4d}  {ms:8.1f} ms   traces: {traces}")

# %% [markdown]
# Four lengths, two compiles — 100/101/102 all became 128. Bucketing trades
# wasted FLOPs on pad positions for a bounded number of compiled programs, and
# on a serving path that is a good trade.
#
# ### What the padding costs, and the way it silently doesn't
#
# Padding is free *correctness-wise* only because the mask zeroes the pad
# positions before they reach the reduction. Drop the mask and nothing raises —
# you simply get a different answer.

# %%
prompt = jax.random.randint(jax.random.PRNGKey(1), (1, 100), 1, VOCAB)
exact = prefill(emb_table, prompt, jnp.ones((1, 100), dtype=jnp.bool_))

padded, mask = pad_to_bucket(prompt)
with_mask = prefill(emb_table, padded, mask)

# The same padded input, with every position claimed to be real.
all_true = jnp.ones(padded.shape, dtype=jnp.bool_)
without_mask = prefill(emb_table, padded, all_true)

print(f"shape {prompt.shape[1]} -> {padded.shape[1]}")
print(f"  padded + mask   max |diff| = {jnp.abs(with_mask - exact).max():.3e}")
print(f"  padded, no mask max |diff| = {jnp.abs(without_mask - exact).max():.3e}")

assert jnp.allclose(with_mask, exact, rtol=1e-6, atol=1e-6), "masked padding changed the result"
print("\nmasked padding agrees to float32 tolerance; unmasked padding does not")

# %% [markdown]
# The two differences are not the same kind of thing. Masked padding came out at
# exactly `0.000e+00` — adding zeros cannot change a sum, so the padded program
# returned the identical float32. That is not guaranteed in general: a different
# length can make XLA group the reduction differently and move the last bits,
# which is why the assertion above uses a tolerance rather than `array_equal`.
# Unmasked padding is wrong by a wide margin, and raised nothing.
#
# The source engine documents a nastier version of this trap. Its KV cache is
# **not** filled contiguously: the server pads each prompt to a bucket, then
# decodes at `bucket + step` while the logical position tracks the real length.
# So the pad slots sit *inside* the filled range, and the obvious shortcut —
# "everything below the write cursor is real" — attends to pad K/V. Quoting the
# source: it "corrupts the output with no error at all."
#
# Padding is cheap. Forgetting what you padded is not.

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
# SOURCE: `~/tpu-jax/tests/test_kv_cache_parity.py`, including its `LOGIT_TOL`
# and the top-2 gap rule in `assert_parity`.
#
# A KV cache is a claim: *attending to a stored key is the same as recomputing
# it.* Nothing enforces that claim. An off-by-one in the write index produces
# fluent, confident, wrong text, and no exception anywhere.
#
# So you write both paths and compare them. Below, the same toy attention runs
# as a full re-forward over the whole prefix each step, and as a cached decode
# that appends one key/value and attends to the stored ones.

# %%
import functools

B, T_PROMPT, D_MODEL, VOCAB3 = 2, 7, 64, 256
SCALE = D_MODEL**-0.5


def make_params(seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    scale = 0.1
    return {
        "emb": jax.random.normal(keys[0], (VOCAB3, D_MODEL)) * scale,
        "Wq": jax.random.normal(keys[1], (D_MODEL, D_MODEL)) * scale,
        "Wk": jax.random.normal(keys[2], (D_MODEL, D_MODEL)) * scale,
        "Wv": jax.random.normal(keys[3], (D_MODEL, D_MODEL)) * scale,
        "Wo": jax.random.normal(keys[4], (D_MODEL, VOCAB3)) * scale,
    }


@jax.jit
def forward_full(params, tokens):
    """Reference: re-run attention over the entire sequence, every step."""
    x = params["emb"][tokens]
    q, k, v = x @ params["Wq"], x @ params["Wk"], x @ params["Wv"]
    scores = jnp.einsum("btd,bsd->bts", q, k) * SCALE
    t = tokens.shape[1]
    scores = jnp.where(jnp.tril(jnp.ones((t, t), dtype=bool)), scores, -jnp.inf)
    h = jnp.einsum("bts,bsd->btd", jax.nn.softmax(scores, axis=-1), v)
    return h[:, -1] @ params["Wo"]


@functools.partial(jax.jit, static_argnames=("offset",))
def decode_one(params, cache_k, cache_v, token, pos, offset=0):
    """Cached: append one key/value, attend to everything stored so far.

    `offset` exists only so the next cell can introduce an off-by-one on
    purpose. Real code would not have it.
    """
    x = params["emb"][token]
    q, k, v = x @ params["Wq"], x @ params["Wk"], x @ params["Wv"]
    write_at = pos + offset
    cache_k = jax.lax.dynamic_update_slice(cache_k, k, (0, write_at, 0))
    cache_v = jax.lax.dynamic_update_slice(cache_v, v, (0, write_at, 0))
    scores = jnp.einsum("bqd,bsd->bqs", q, cache_k) * SCALE
    live = jnp.arange(cache_k.shape[1]) <= pos
    scores = jnp.where(live, scores, -jnp.inf)
    h = jnp.einsum("bqs,bsd->bqd", jax.nn.softmax(scores, axis=-1), cache_v)
    return cache_k, cache_v, h[:, 0] @ params["Wo"]


def greedy_reference(params, prompt, n_new):
    """Grow the sequence and re-forward the whole thing each step."""
    tokens, logits = prompt, []
    for _ in range(n_new):
        lg = forward_full(params, tokens)
        logits.append(lg)
        tokens = jnp.concatenate([tokens, jnp.argmax(lg, -1)[:, None]], axis=1)
    return tokens[:, prompt.shape[1] :], logits


def greedy_cached(params, prompt, n_new, capacity, offset=0):
    """Prefill once, then one cached step per new token."""
    cache_k = jnp.zeros((prompt.shape[0], capacity, D_MODEL), jnp.float32)
    cache_v = jnp.zeros_like(cache_k)

    # Prefill: write every prompt key/value, read logits at the last position.
    x = params["emb"][prompt]
    cache_k = cache_k.at[:, : prompt.shape[1]].set(x @ params["Wk"])
    cache_v = cache_v.at[:, : prompt.shape[1]].set(x @ params["Wv"])
    lg = forward_full(params, prompt)

    out, logits = [], []
    for step in range(n_new):
        logits.append(lg)
        token = jnp.argmax(lg, -1)[:, None]
        out.append(token)
        cache_k, cache_v, lg = decode_one(params, cache_k, cache_v, token, prompt.shape[1] + step, offset)
    return jnp.concatenate(out, axis=1), logits


params3 = make_params()
prompt3 = jax.random.randint(jax.random.PRNGKey(42), (B, T_PROMPT), 1, VOCAB3)
N_NEW = 8

ref_tokens, ref_logits = greedy_reference(params3, prompt3, N_NEW)
cac_tokens, cac_logits = greedy_cached(params3, prompt3, N_NEW, T_PROMPT + N_NEW)

worst = max(float(jnp.abs(r - c).max()) for r, c in zip(ref_logits, cac_logits))
print(f"worst logit difference over {N_NEW} steps: {worst:.3e}")
print(f"tokens identical: {bool(jnp.array_equal(ref_tokens, cac_tokens))}")

# %% [markdown]
# The logits differ a little. They are supposed to. The two paths sum the same
# attention in a different order, and float32 addition is not associative — the
# source test allows `1e-4` and reports a measured worst case around `1e-6`.
#
# That gap has a consequence most people discover the hard way. If two tokens
# are nearly tied for the argmax, a difference of `1e-6` is enough to swap
# them, and your parity test fails on a decode that is completely correct.
#
# So the source test does not assert tokens are equal. It asserts they are equal
# **wherever the decision was not a near-tie**:

# %%
for i, (r, c) in enumerate(zip(ref_logits, cac_logits)):
    ordered = jnp.sort(r, axis=-1)
    gap = ordered[:, -1] - ordered[:, -2]  # top-2 margin, per row
    delta = jnp.abs(r - c).max(axis=-1)
    for row in range(r.shape[0]):
        decisive = float(gap[row]) > float(delta[row])
        same = int(ref_tokens[row, i]) == int(cac_tokens[row, i])
        if decisive:
            assert same, f"step {i} row {row}: differs on a decisive margin"
    print(f"step {i}: top-2 margins {[round(float(g), 4) for g in gap]}, max |dlogit| {float(delta.max()):.2e}")

print("\nevery decisive step agreed")

# %% [markdown]
# ### Does the test actually catch anything?
#
# A test that has never failed is a rumour. The most common way a hand-rolled
# cache breaks is an off-by-one in the write index — the key lands one slot
# late, the query attends to a stale zero, and the output is confidently wrong.
#
# `decode_one` takes an `offset` argument for exactly this. Set it to 1:

# %%
bad_tokens, bad_logits = greedy_cached(params3, prompt3, N_NEW, T_PROMPT + N_NEW + 1, offset=1)

bad_worst = max(float(jnp.abs(r - c).max()) for r, c in zip(ref_logits, bad_logits))
print(f"off-by-one worst logit difference: {bad_worst:.3e}   (correct path: {worst:.3e})")
print(f"off-by-one tokens identical:       {bool(jnp.array_equal(ref_tokens, bad_tokens))}")
print(f"\nreference tokens: {ref_tokens.tolist()}")
print(f"off-by-one tokens: {bad_tokens.tolist()}")
print("\nno exception was raised by the broken path.")

# %% [markdown]
# ### Why bother with the cache at all
#
# Re-forwarding is simpler and always correct. It is also quadratic: step *t*
# recomputes every key and value for tokens *0..t* that it already computed at
# step *t-1*, and builds a `[T, T]` score matrix to do it. The cached step
# computes one key, one value, and a `[1, T]` score row.
#
# Watch where that starts to matter. It is further out than you might guess.

# %%
print(f"{'context':>8}  {'re-forward':>12}  {'cached step':>12}  {'ratio':>7}")
for ctx in (256, 1024, 2048, 4096, 8192):
    toks = jax.random.randint(jax.random.PRNGKey(2), (1, ctx), 1, VOCAB3)
    ck = jnp.zeros((1, ctx + 1, D_MODEL), jnp.float32)
    cv = jnp.zeros_like(ck)
    one = toks[:, :1]

    jax.block_until_ready(forward_full(params3, toks))
    jax.block_until_ready(decode_one(params3, ck, cv, one, ctx - 1))

    start = time.perf_counter()
    for _ in range(10):
        jax.block_until_ready(forward_full(params3, toks))
    full_ms = (time.perf_counter() - start) * 1e2

    start = time.perf_counter()
    for _ in range(10):
        jax.block_until_ready(decode_one(params3, ck, cv, one, ctx - 1))
    cached_ms = (time.perf_counter() - start) * 1e2

    print(f"{ctx:>8}  {full_ms:>10.3f} ms  {cached_ms:>10.3f} ms  {full_ms / cached_ms:>6.1f}x")

# %% [markdown]
# Two things in that table, and the second is the more useful one.
#
# The cached column is roughly flat while the re-forward column climbs with
# context, which is the asymptotic story: `O(T^2)` against `O(T)`.
#
# But at the short end the ratio is **below 1** — the cache is *slower*. There
# is so little arithmetic in a 256-token attention at this width that both
# columns are measuring dispatch overhead, and the cached path has slightly more
# of it: an extra `dynamic_update_slice` and its own kernel launch. The cache
# does not begin paying for itself until the work it avoids is bigger than the
# work it adds.
#
# That is worth carrying around. "Cached decode is faster than re-forwarding" is
# a statement about large contexts. At small ones it is false, and a benchmark
# run only at small ones would tell you so confidently.
#
# The flatness of the cached column is what section 4 is about — and it is flat
# for a reason that has nothing to do with arithmetic.

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
