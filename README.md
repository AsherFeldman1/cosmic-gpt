# How many cosmic rays can GPT-2 survive?

Fault-injection experiment: flip uniformly random bits in GPT-2 (124M) weights and watch perplexity on a fixed WikiText-2 slice.

## Summary of findings

1. **GPT-2 dies after about 12 random bit flips** (median, fp16; fp32 22, bf16 16) out of 2–4 billion weight bits. 3 flips already kill 10–20% of runs, and 100 flips kill essentially every run in every dtype.
2. **It's basically one bit.** A single flip of the exponent MSB (fp32 bit 30, fp16/bf16 bit 14) kills it **82–92% of the time**. Mantissa bits never killed it, even 100 at once.
3. **Per flip, all tensor groups are about equally deadly, so LayerNorm is ~1000× more fragile per bit.** It is 0.6 Mbit out of 1,991.
4. **Clamping each weight to its clean tensor's [min, max] raises median lethality from ~12 to ~1,400 flips** (~100×) on identical flips, with no perplexity cost below ~1k flips.
5. **Qwen2.5-0.5B is no tougher** (median ~8 flips). Per Gbit of weights it is ~6× more fragile. In low Earth orbit, unprotected fp16 GPT-2 would last **about 10 minutes to 1.5 hours**. On the ground, the answer ranges from months to over a millennium depending on the memory's soft-error rate.

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

`uv run python -m cosmic_gpt.cosmic --dtype fp16`:

```
GPT-2 fp16: 1,991,036,928 weight bits (1,899 Mbit)
median flips to death (sweep 1): 12;  mean flips to death 1/p (sweeps 2+6): 19

environment        rate basis                            flips/hour  median death   mean death
----------------------------------------------------------------------------------------------
sea level          modern DRAM (0.56 FIT/Mbit) x1          1.06e-06    1,336.5 yr   2,073.8 yr
Denver             modern DRAM (0.56 FIT/Mbit) x3.5        3.72e-06      381.8 yr     592.5 yr
airliner, 40k ft   modern DRAM (0.56 FIT/Mbit) x300        0.000319        4.5 yr       6.9 yr
sea level          1990s DRAM (1000 FIT/Mbit) x1             0.0019       273.4 d       1.2 yr
Denver             1990s DRAM (1000 FIT/Mbit) x3.5          0.00665        78.1 d      121.2 d
airliner, 40k ft   1990s DRAM (1000 FIT/Mbit) x300             0.57        21.9 h        1.4 d
low Earth orbit    1e-07 upsets/bit/day                         8.3         1.5 h        2.3 h
low Earth orbit    1e-06 upsets/bit/day                          83         9 min       14 min
```

**How flips become deaths.** The *median* column uses sweep 1's measured median lethal flip count. The *mean* column uses a per-flip lethality p averaged over bit positions. For most bits p comes from sweep 2 (100 flips at bit b), as `p_b = 1-(1-D_b)^(1/100)`. For the exponent MSB, 100 flips always kill, so sweep 6 measures single-flip lethality directly (50 seeds per dtype). The two estimates agree: for a geometric process the median is ln2/p ≈ 13, against a measured 12.

**Rates and sources.** The ground-level rate is the biggest uncertainty, so the table shows both ends of the published range:

- **0.56 FIT/Mbit:** 99% upper bound on DRAM soft errors measured on production servers. Li, Shen, Huang & Chu, ["A Memory Soft Error Measurement on Production Systems"](https://sites.rochester.edu/michaelhuang/wp-content/uploads/2023/03/usenix07.pdf), USENIX ATC 2007.
- **1000 FIT/Mbit:** within the 200–5,000 FIT/Mbit range of earlier lab studies, which that paper summarizes. A Google fleet study found 25,000–70,000 FIT/Mbit but attributes most of it to hard errors, so it isn't used here. Schroeder, Pinheiro & Weber, ["DRAM Errors in the Wild"](https://www.cs.toronto.edu/~bianca/papers/sigmetrics09.pdf), SIGMETRICS 2009.
- **Denver ×3.5:** "more than three times higher in Denver than it is in New York." Actel, ["Effects of Neutrons on Programmable Logic"](https://ww1.microchip.com/downloads/aemDocuments/documents/FPGA/ProductDocuments/SupportingCollateral/serwp.pdf), 2002.
- **40k ft ×300:** "FL400 is 300X > neutron flux @ sea level." S. Wiley, ["NASA Aviation Activities and Space Weather"](https://ccmc.gsfc.nasa.gov/publicData/SWxWorkshops/2014/Wiley_aviationactivities.pdf), NASA GSFC Space Weather Workshop, 2014.
- **Low Earth orbit, 10⁻⁷–10⁻⁶ upsets/bit/day:** Wang et al., ["A Case for Application-Aware Space Radiation Tolerance in Orbital Computing"](https://arxiv.org/abs/2407.11853), 2024. The same paper reports 4.76×10⁻⁷ measured over 3 years on the JASON-2 satellite. In orbit the radiation is mostly trapped protons and heavy ions, so a per-bit rate is used directly instead of a neutron-flux multiplier.

Multipliers are relative to the JESD89A reference (New York City, sea level). The estimate counts only the model weights as exposed memory, not activations or the KV cache, and assumes no ECC. 1 Mbit = 2²⁰ bits. Override with `--bits` and `--fit-per-mbit`.

## Run it

```bash
uv sync
uv run pytest                              # injector unit tests (CPU, tiny model)
uv run python -m cosmic_gpt.sweeps all     # sweeps 1-6 -> results/sweep*.csv (resumable)
uv run python -m cosmic_gpt.plots          # -> charts/*.png
uv run python -m cosmic_gpt.cosmic --dtype fp16
uv run python -m cosmic_gpt.sweeps --replay s1_fp16_n30_seed3   # re-run any logged trial bit-exactly
```

All six sweeps take about 34 minutes on an M1 Pro (16 GB, MPS).

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

## Method

- **Sweeps:** (1) 1–10⁶ flips × fp32/fp16/bf16, 20 seeds per point. (2) 100 flips at each bit position, 10 seeds. (3) flips restricted to one tensor group, fp16, 20 seeds. (4) sweep 1's fp16 flips plus clamping, 20 seeds. (5) Qwen2.5-0.5B fp16, 10 seeds per point. (6) single flips at the bit positions that are always lethal in sweep 2, 50 seeds. Error bars on the charts are 95% Wilson intervals.
- **Model / device:** HF `gpt2` (124M) on MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1`. If an int op fails on device, bit flips fall back to a CPU copy.
- **Eval set:** the first 4,096 GPT-2 tokens of `Salesforce/wikitext` `wikitext-2-raw-v1` test split (joined with `\n\n`), as 4 non-overlapping 1,024-token windows. Perplexity is computed in fp32 from the model's logits. Clean baselines: fp32 28.216, fp16 28.221, bf16 28.306, Qwen2.5-0.5B fp16 14.297.
- **What gets flipped:** every entry of `model.named_parameters()`: weights, biases, and LayerNorm gains. GPT-2's tied `wte`/`lm_head` counts once, which is how it sits in memory. Buffers such as the causal mask are excluded. Bits are drawn uniformly **without replacement** over all `numel × width` bits. Several flips landing in the same weight are XOR'd together.
- **Bit numbering:** 0 = mantissa LSB, width−1 = sign.
- **Groups:** `embed` = wte, wpe; `ln` = all LayerNorms; `attn` = c_attn and attn c_proj; `mlp` = c_fc and mlp c_proj. Sweep 3's grid stops at 1e5 flips.
- **Death:** *hard* = any NaN/Inf in the logits. *Soft* = perplexity > 2× the same dtype's clean baseline, where a NaN or Inf perplexity also counts as dead. Alive = neither. Of 1,716 deaths across all sweeps, 885 were hard and 348 were perplexity = ∞ with finite logits (some token's probability underflows to 0), which counts as soft.
- **Clean state:** a clean copy of every tensor is held on device. Every tensor touched by a trial is restored before the next trial, so no trial sees an earlier trial's flips.
- **Seeding:** global seeds fixed to 0. Each trial uses its own `np.random.default_rng([sweep, dtype, n, seed, extra])`. Sweep 4 reuses sweep 1's fp16 streams, so clamp vs no clamp is compared on **identical flips**.
- **Clamp mitigation:** after flipping, NaN → 0, then each touched tensor is clamped to its clean (min, max). This assumes those two numbers per tensor are stored somewhere safe.
- **Modern model:** `Qwen/Qwen2.5-0.5B` (tied embeddings, 494M params, 7.9 Gbit in fp16). It is evaluated one window at a time to bound the memory of its 151k-token vocabulary's logits.
- **Flip logs:** every trial's exact flip list is saved. Lists with more than 1,000 flips (734 MB in total) are not in the repo; `--replay` regenerates them bit-exactly from the seed.

## License

MIT, see [LICENSE](LICENSE).
