---
name: port-finding
description: Turn a measured finding from the ~/tpu-jax source rigs into a notebook section that re-measures on the reader's own chip. Use when writing notebook content from a REPORT.md result or a test in the source repo.
---

Port a finding from `~/tpu-jax` into a notebook section. `$ARGUMENTS` names the finding (e.g.
"buffer donation", "int8 KV capacity") or a source file.

The rule this exists to enforce: **the notebook re-measures, it does not quote.** A reader on a v5e-1
must see their own number, not a v6e-1 number asserted at them.

## 1. Find the primary source

Locate the finding in `~/tpu-jax`. Findings have two halves and you usually need both:

- **The test** under `tests/` — proves correctness, and is the honest basis for the notebook's
  assertion cell (e.g. `test_donation.py`, `test_kv_cache_parity.py`, `test_quantized_kv.py`).
- **The report** under `benchmarks/runs/*/REPORT.md` — carries the measured number, the conditions it
  was measured under, and sometimes a correction that walked an earlier number back.

Read the surrounding section, not just the line with the number. Check
`2026-07-29-kv-quant-v6e1/REPORT.md` under "Corrections to earlier reporting" before citing any
capacity figure — several were withdrawn.

Do not cite `~/tpu-jax-{v5e1-2b,v6e1-2b,4b,12b,26b,31b,inf2}`; they are near-identical forks.

## 2. Write the section into `src/<stem>.py`

Structure each ported finding as:

1. **A markdown cell** stating what is being measured and why it matters, citing the source
   (`SOURCE: tests/test_donation.py and kv-quant REPORT section 1`).
2. **A code cell that measures it** on the live runtime. Keep it small enough to read. Use
   `jax.block_until_ready` around timed regions and report a median over several runs, not one sample.
   For memory claims read `jax.local_devices()[0].memory_stats()`.
3. **An assertion or comparison** derived from the test half — the thing that would catch a wrong
   answer, not just a slow one.
4. **A markdown cell interpreting the reader's result**, which states the source measurement as
   context and explicitly says the reader's number will differ by chip.

Never write the source number as the output of a cell. If a claim genuinely cannot be re-measured in
Colab (the perplexity study, the HTTP concurrency table), cite it as a reported result with its
conditions and say why it is not reproduced here.

## 3. Build and verify

`make build`, then `make check`. Then `make verify` to run it on a real TPU — this is the step that
catches a cell that raises, which reading the markdown never will. Consider `make verify-v5e1` too if
the section makes any capacity or memory claim, since that is where they break.

Report the number the run actually produced against the number the source reported. If they disagree
by more than the chip difference explains, say so rather than smoothing it over.
