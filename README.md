# How many cosmic rays can GPT-2 survive?

Computer memory gets hit by cosmic rays. Occasionally one flips a single bit, a 0 becomes a 1 or the reverse. An AI model is billions of numbers stored in memory, so what happens to it when bits start flipping?

I flipped random bits in GPT-2's weights (the 124-million numbers that make up the model) and checked whether it could still write normal text. In total that's 2,580 runs.

## What "broken" means

Here is GPT-2 finishing the sentence *"The Eiffel Tower is located in"*:

| | output |
|---|---|
| **Normal** | the heart of the city, and is the only place in the world where you can see the Eiffel Tower. |
| **After 3 random flips** | heated heated heated heated heated heated heated heated heated heated heated heated … |
| **After 3 random flips** (another run) | the heart 124 124 124 124 124 124 124 124 124 124 124 124 … |
| **After 10 random flips** | Bane Bane Bane Bane Baneacheacheache Bane Bane Bane Bane … |
| **After 10 random flips** (a lucky run) | the heart of the city, and is the only place in the world where you can see the Eiffel Tower. |

The model either keeps working as if nothing happened or turns into garbage; there is almost no in-between. To decide automatically, every run is scored on a few pages of Wikipedia text by how well the model predicts each next word. A run counts as **broken** if it predicts at least 2× worse than the normal model. In practice, 98.6% of runs were either within 10% of normal or more than 10× worse, so the exact cutoff barely matters.

## Results

### 1. About a dozen flips is enough

![A dozen random bit flips break GPT-2](charts/survival_vs_flips.png)

GPT-2 stored in 16 bits per number has about 2 billion bits. Flipping just **12 random ones breaks it half the time**, and 300 always do. Storing it in 32 bits per number barely helps (22 flips).

### 2. It's almost all one bit

![It's almost all one bit](charts/bit_lethality.png)

Each number is stored like scientific notation: a sign, an **exponent** (the "×10ⁿ" part), and a **fraction** (the digits). Flipping a fraction bit nudges a number slightly, and the model doesn't care. Flipping the top exponent bit turns a typical weight like **0.02 into 1,311**. In 32-bit storage it turns into 6.8 × 10³⁶. One such number is enough to drown out everything else, so a single flip of that bit breaks GPT-2 **82% of the time**. A random flip lands on it 1 time in 16, which is why a dozen random flips is the typical breaking point.

### 3. Small parts are just as deadly

![Small parts of the model are just as deadly](charts/group_survival.png)

LayerNorm (small scaling layers between the main blocks) is only 0.03% of the model, yet it breaks after about as many flips as the big parts. A flip that lands there is, bit for bit, about 1,000× more damaging.

### 4. A cheap fix makes it about 100× tougher

![A cheap fix makes it ~100x tougher](charts/clamp.png)

Since the damage comes from numbers becoming absurdly large, the fix is to remember each layer's smallest and largest weight and clamp anything outside that range back in. On exactly the same flips, it now takes **1,390 flips instead of 12** to break half the runs, and nearly every run survives 100 flips.

### 5. A newer, bigger model is no tougher

![A newer, 4x bigger model is no tougher](charts/modern_model.png)

Qwen2.5-0.5B, a 2024 model 4× the size of GPT-2, breaks after about 8 flips. Because it has 4× as many bits, random flips are also 4× more likely to happen to it.

## How long would it last?

How often bits flip depends on altitude, since the atmosphere shields us from cosmic rays, and on the memory chip. Published numbers for memory chips at sea level range from about 0.56 to 1,000 errors per billion hours per megabit. The low end is modern hardware and the high end is older. These estimates assume ordinary memory without error correction (ECC), the kind most laptops and phones use. Servers usually have ECC, which fixes single flipped bits.

Time until 16-bit GPT-2 has a 50% chance of being broken:

| where | modern memory | older memory |
|---|---|---|
| Sea level | ~1,300 years | ~9 months |
| Denver (3.5× more cosmic rays) | ~380 years | ~2.5 months |
| Airplane at 40,000 ft (300×) | ~4.5 years | ~1 day |
| Low Earth orbit | 1.5 hours to 9 minutes | (same) |

In orbit there's no atmosphere at all, and published rates are 0.1 to 1 flipped bit per million per day. At that rate GPT-2 breaks within hours.

Sources:
- Error rate of modern memory (0.56): Li, Shen, Huang & Chu, ["A Memory Soft Error Measurement on Production Systems"](https://sites.rochester.edu/michaelhuang/wp-content/uploads/2023/03/usenix07.pdf), USENIX ATC 2007. The 1,000 figure is within the 200–5,000 range of older lab studies that this paper summarizes.
- Denver: Actel, ["Effects of Neutrons on Programmable Logic"](https://ww1.microchip.com/downloads/aemDocuments/documents/FPGA/ProductDocuments/SupportingCollateral/serwp.pdf), 2002 ("more than three times higher in Denver than it is in New York").
- 40,000 ft: S. Wiley, ["NASA Aviation Activities and Space Weather"](https://ccmc.gsfc.nasa.gov/publicData/SWxWorkshops/2014/Wiley_aviationactivities.pdf), NASA GSFC, 2014 ("FL400 is 300X > neutron flux @ sea level").
- Orbit: Wang et al., ["A Case for Application-Aware Space Radiation Tolerance in Orbital Computing"](https://arxiv.org/abs/2407.11853), 2024.

Run `uv run python -m cosmic_gpt.cosmic` for the full table, including other number formats and your own error rates.

## Run it yourself

```bash
uv sync
uv run python -m cosmic_gpt.sweeps all   # all experiments, ~35 min on an M1 Pro; results/ has the outputs
uv run python -m cosmic_gpt.plots        # charts
uv run python -m cosmic_gpt.cosmic       # time-to-break table
uv run pytest                            # tests
```

<details>
<summary><b>Technical details</b></summary>

**Experiments** (`cosmic_gpt/sweeps.py`, one CSV per experiment in `results/`):

| # | what | runs |
|---|---|---|
| 1 | 1 to 1M random flips, in fp32, fp16 and bf16 | 20 per point |
| 2 | 100 flips all at the same bit position, for every position | 10 per bar |
| 3 | flips restricted to one part of the model (embeddings, attention, MLP, LayerNorm), fp16 | 20 per point |
| 4 | experiment 1's fp16 flips, then clamping | 20 per point |
| 5 | Qwen2.5-0.5B, fp16 | 10 per point |
| 6 | a single flip of the top exponent bit | 50 per format |

**Model and data.** HF `gpt2` on Apple MPS. The eval text is the first 4,096 GPT-2 tokens of WikiText-2-raw (test split), as four 1,024-token windows. "Predicts N× worse" means perplexity relative to the clean model: 28.2 for fp32/fp16, 28.3 for bf16, and 14.3 for Qwen2.5-0.5B. A run is broken if perplexity > 2× clean or the logits contain NaN/Inf.

**Flipping.** Bits are chosen uniformly, without replacement, across every parameter, including biases and LayerNorm gains. GPT-2's input and output embeddings share memory, so they count once. Each trial restores the clean weights first. Every trial has its own random seed and its exact flip list is saved in `results/flips/`. Lists over 1,000 flips are left out of the repo for size, and `uv run python -m cosmic_gpt.sweeps --replay <trial_id>` regenerates any trial exactly. Experiment 4 uses the same seeds as experiment 1, so both lines in chart 4 see identical flips.

**Clamping.** After the flips, NaNs become 0 and each tensor is clamped to its clean min and max. This assumes those two numbers per tensor are kept somewhere safe.

**Time to break.** flips/hour = errors per billion hours per Mbit × Mbit of weights × altitude multiplier ÷ 10⁹. Multipliers are relative to New York at sea level (the JESD89A reference). Only the weights count as exposed memory. The table above uses the measured median from experiment 1 (12 flips for fp16). `cosmic.py` also prints a mean estimate built from experiments 2 and 6 (19 flips), which agrees: for a random process with that per-flip risk, the median is about 13.

**Code.** `inject.py` (bit flipping, restore, clamp), `model.py` (loading, perplexity), `sweeps.py` (experiments), `plots.py` (charts), `cosmic.py` (time-to-break table), `tests/` (flip correctness).

</details>

## License

MIT, see [LICENSE](LICENSE).
