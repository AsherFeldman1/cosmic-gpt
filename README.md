# How many cosmic rays can GPT-2 survive?

Fault-injection experiment: flip uniformly random bits in GPT-2 (124M) weights and watch perplexity on a fixed WikiText-2 slice.

## Summary of findings

1. **GPT-2 dies after about 12 random bit flips** (median, fp16; fp32 22, bf16 16) out of 2–4 billion weight bits. 3 flips already kill 10–20% of runs, and 100 flips kill essentially every run in every dtype.
2. **It's basically one bit.** The exponent MSB (fp32 bit 30, fp16/bf16 bit 14) kills **82–92% of the time from a single flip**. Mantissa bits never killed it, even 100 at once.
3. **Per flip, all tensor groups are about equally deadly, so LayerNorm is ~1000× more fragile per bit.** It is 0.6 Mbit out of 1,991.
4. **Clamping each weight to its clean tensor's [min, max] raises median lethality from ~12 to ~1,400 flips** (~100×) on identical flips, with no perplexity cost below ~1k flips.
5. **Qwen2.5-0.5B is no tougher** (median ~8 flips). Per Gbit of weights it is ~6× more fragile. With placeholder physics (1000 FIT/Mbit, unsourced), unprotected fp16 GPT-2 lasts **~9 months at sea level and ~1 day at 35k ft.**

## Charts

### 1. Survival vs number of flips
![survival vs flips](charts/survival_vs_flips.png)

### 2. Per-bit-position lethality (100 flips at bit b)
![bit lethality](charts/bit_lethality.png)

### 3. Per-tensor-group lethality (fp16)
![group survival](charts/group_survival.png)

### 4. Mitigation: clamp to clean range (fp16)
![clamp](charts/clamp.png)

### 5. A modern model: Qwen2.5-0.5B (fp16)
![modern model](charts/modern_model.png)

## Cosmic-ray estimate

`uv run python -m cosmic_gpt.cosmic --dtype fp16` (default 1000 FIT/Mbit):

```
GPT-2 fp16: 1,991,036,928 bits (1,899 Mbit) at 1000 FIT/Mbit
p_lethal per random flip (sweep 2, mean over 16 bit positions): 5.17e-02  -> expected flips to death 1/p = 19
empirical N50 (sweep 1, median lethal flips): 12

environment             mult   flips/hour    death (1/p)    death (N50)
-----------------------------------------------------------------------
sea level                  1       0.0019         1.2 yr        273.4 d
Denver (1.6 km)          3.5      0.00665        121.2 d         78.1 d
airliner (35k ft)        300         0.57          1.4 d         21.9 h
ISS (LEO)               1000          1.9         10.2 h          6.6 h
```

The two lethality estimates agree. The `1/p` column is the *mean* flips to death. For a geometric process the median is ln2/p ≈ 13, which matches the empirical N50 of 12.

> **TODO (you): sources.** The default FIT/Mbit (`DEFAULT_FIT_PER_MBIT`) and the altitude multipliers (`ALTITUDES`) in `cosmic_gpt/cosmic.py` are round placeholder numbers, not cited values. Real soft-error rates vary by orders of magnitude with process node, memory type, and ECC. The ISS is a different radiation environment, and a single multiplier is a crude stand-in for it.

## Run it

```bash
uv sync
uv run pytest                              # injector unit tests (CPU, tiny model)
uv run python -m cosmic_gpt.sweeps all     # sweeps 1-6 -> results/sweep*.csv (resumable)
uv run python -m cosmic_gpt.plots          # -> charts/*.png
uv run python -m cosmic_gpt.cosmic --fit-per-mbit 1000 --dtype fp16
uv run python -m cosmic_gpt.sweeps --replay s1_fp16_n30_seed3   # re-run any logged trial bit-exactly
```

Measured compute time on an M1 Pro (16 GB, MPS): sweep 1 10 min, sweep 2 6.5 min, sweep 3 7 min, sweep 4 3.5 min, sweep 5 5 min, sweep 6 1.5 min. **About 34 min total.** (An earlier sweep 5 attempt was killed after swapping on a memory-loaded machine; it resumed from its CSV.)

## Layout

| file | what |
|---|---|
| `cosmic_gpt/model.py` | model loading, fixed eval tokens, perplexity |
| `cosmic_gpt/inject.py` | `Injector.flip_bits(n, rng, bit=, group=)`, replay, restore, clamp |
| `cosmic_gpt/sweeps.py` | sweeps 1–6, CSV output, flip logging, `--replay` |
| `cosmic_gpt/plots.py` | charts |
| `cosmic_gpt/cosmic.py` | flips/hour and time-to-death table |
| `results/sweepN.csv` | one row per trial (ppl, baseline, hard/soft death, seconds) |
| `results/flips/sweepN/<trial_id>.npz` | exact flip list per trial: `param`, `elem`, `bit` (+ param names) |

## Design choices and assumptions

- **Model / device:** HF `gpt2` (124M) on MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1`. If an int op fails on device, bit flips fall back to a CPU copy.
- **Eval set:** the first 4,096 GPT-2 tokens of `Salesforce/wikitext` `wikitext-2-raw-v1` test split (joined with `\n\n`), as 4 non-overlapping 1,024-token windows. Perplexity is computed in fp32 from the model's logits. Clean baselines: fp32 28.216, fp16 28.221, bf16 28.306, Qwen2.5-0.5B fp16 14.297.
- **What gets flipped:** every entry of `model.named_parameters()`: weights, biases, and LayerNorm gains. GPT-2's tied `wte`/`lm_head` counts once, which is how it sits in memory. Buffers such as the causal mask are excluded. Bits are drawn uniformly **without replacement** over all `numel × width` bits. Several flips landing in the same weight are XOR'd together.
- **Bit numbering:** 0 = mantissa LSB, width−1 = sign.
- **Groups:** `embed` = wte, wpe; `ln` = all LayerNorms; `attn` = c_attn and attn c_proj; `mlp` = c_fc and mlp c_proj. Sweep 3's grid stops at 1e5 flips.
- **Death:** *hard* = any NaN/Inf in the logits. *Soft* = perplexity > 2× the same dtype's clean baseline, where a NaN or Inf perplexity also counts as dead. Alive = neither. Of 1,716 deaths across all sweeps, 885 were hard and 348 were perplexity = ∞ with finite logits (some token's probability underflows to 0), which counts as soft.
- **Clean state:** a clean copy of every tensor is held on device. Every tensor touched by a trial is restored before the next trial, so no trial sees an earlier trial's flips.
- **Seeding:** global seeds fixed to 0. Each trial uses its own `np.random.default_rng([sweep, dtype, n, seed, extra])`. Sweep 4 deliberately reuses sweep 1's fp16 streams, so clamp vs no clamp is compared on **identical flips**.
- **Clamp mitigation:** after flipping, NaN → 0, then each touched tensor is clamped to its clean (min, max). This assumes those two numbers per tensor are stored somewhere safe.
- **Seeds:** 20 per point for sweeps 1, 3 and 4. 10 per bit for sweep 2. **10 per point for sweep 5** (Qwen is ~4× slower). 50 for sweep 6. Bars on the charts are 95% Wilson intervals.
- **Modern model:** `Qwen/Qwen2.5-0.5B` (tied embeddings, 494M params, 7.9 Gbit in fp16). It is evaluated one window at a time because its 151k vocabulary makes batched logits too large for a 16 GB machine.

## Deviations from the original spec

- **Added sweep 6 (single-flip lethality at saturated bits).** Sweep 2 flips 100 weights per trial, so once a bit position kills 100% of runs it can't tell you the *per-flip* lethality, which is what `cosmic.py` needs. Without sweep 6, the p-model said 429 flips to death while sweep 1 measured 12. Sweep 6 flips that bit once per trial (50 seeds) and measured 82–92% lethality. `cosmic.py` uses that measurement for those bits and the sweep 2 formula `1-(1-D_b)^(1/100)` for the others.
- **Flip logs:** every trial's exact flip list is written to disk. Lists with more than 1,000 flips (734 MB in total) go to `results/flips/big/`, which is **gitignored**. They can be regenerated bit-exactly from the seed with `--replay`. Lists with ≤1,000 flips are committed.
- `cosmic.py` treats 1 Mbit as 2²⁰ bits and counts only the model weights as exposed memory, not activations or the KV cache. Override with `--bits`.

## License

MIT, see [LICENSE](LICENSE).
