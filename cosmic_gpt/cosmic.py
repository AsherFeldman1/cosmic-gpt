"""Cosmic-ray time-to-death estimate for a model sitting in unprotected (non-ECC) memory.

Usage: python -m cosmic_gpt.cosmic [--dtype fp16] [--fit-per-mbit 0.56] [--orbit-rate 3e-7] [--bits N]

On the ground and in the air:
    flips/hour = FIT/Mbit * Mbit_in_memory * flux_multiplier / 1e9
(1 FIT = 1 failure per 1e9 device-hours; multipliers are relative to the
JESD89A reference, New York City at sea level.) In orbit the radiation
environment is different (trapped protons, heavy ions), so a measured
per-bit-per-day upset rate is used directly instead of a multiplier.

Lethality comes from the experiments:
  * N50: sweep 1's empirical median-lethal number of flips (log-interpolated).
  * p-model: per-flip lethality p = mean over bit positions b of p_b, where
    p_b = 1 - (1 - D_b)^(1/100) from sweep 2's death rate D_b at 100 flips,
    or the directly measured single-flip death rate (sweep 6) for bits where
    100 flips always kill. Mean flips to death = 1/p.
"""

import argparse

import numpy as np
import pandas as pd

from cosmic_gpt.model import ROOT

RESULTS = ROOT / "results"

# Sea-level soft-error rates for DRAM, FIT per Mbit.
#   0.56: 99% upper bound measured on production servers, excluding machines
#         with hard errors. Li, Shen, Huang & Chu, "A Memory Soft Error
#         Measurement on Production Systems", USENIX ATC 2007.
#   1000: within the 200-5000 FIT/Mbit range of earlier (1990s-era) lab
#         studies summarized in the same paper.
MODERN_FIT, OLDER_FIT = 0.56, 1000.0
FIT_PER_MBIT = {"modern DRAM": MODERN_FIT}   # pass --fit-per-mbit 1000 for 1990s DRAM

# Neutron flux relative to New York City at sea level.
#   Denver: "more than three times higher in Denver than it is in New York".
#           Actel, "Effects of Neutrons on Programmable Logic", 2002.
#   FL400:  "FL400 is 300X > neutron flux @ sea level". S. Wiley, "NASA Aviation
#           Activities and Space Weather", NASA GSFC Space Weather Workshop, 2014.
FLUX = {"sea level": 1.0, "Denver": 3.5, "airliner, 40k ft": 300.0}

# Low Earth orbit: 1e-7 to 1e-6 upsets per bit per day. Wang et al., "A Case for
# Application-Aware Space Radiation Tolerance in Orbital Computing", arXiv:2407.11853, 2024
# (which also reports 4.76e-7 /bit/day measured over 3 years on JASON-2).
# The default is the middle of that range on a log scale.
ORBIT_PER_BIT_DAY = 3e-7

GPT2_PARAMS = 124_439_808
WIDTH = {"fp32": 32, "fp16": 16, "bf16": 16}


def p_lethal_per_flip(dtype: str) -> float:
    s2 = pd.read_csv(RESULTS / "sweep2.csv")
    d = s2[s2.dtype == dtype]
    g = d.groupby("bit")["alive"].agg(["sum", "count"])
    n_flips = int(d.n_flips.iloc[0])
    dead = 1 - g["sum"] / g["count"]
    # All-dead bits would give p_b = 1; cap at the seed-count resolution
    # (sweep 6 replaces these with a direct measurement below).
    dead = dead.clip(upper=1 - 0.5 / g["count"])
    p_b = 1 - (1 - dead) ** (1 / n_flips)
    s6_path = RESULTS / "sweep6.csv"
    if s6_path.exists():
        s6 = pd.read_csv(s6_path)
        direct = 1 - s6[s6.dtype == dtype].groupby("bit")["alive"].mean()
        p_b.loc[direct.index] = direct
    return float(p_b.mean())


def n50(dtype: str) -> float:
    s1 = pd.read_csv(RESULTS / "sweep1.csv")
    g = s1[s1.dtype == dtype].groupby("n_flips")["alive"].mean()
    x, y = np.log10(g.index.values.astype(float)), g.values
    below = np.flatnonzero(y < 0.5)
    if below.size == 0:
        return float("inf")
    i = below[0]
    if i == 0:
        return float(g.index[0])
    t = (y[i - 1] - 0.5) / (y[i - 1] - y[i])
    return float(10 ** (x[i - 1] + t * (x[i] - x[i - 1])))


def human(hours: float) -> str:
    """Two significant figures, as in the write-up: 1,337 years prints as 1,300 yr."""
    if not np.isfinite(hours):
        return "never"
    sig = lambda x: f"{float(f'{x:.2g}'):,.0f}" if x >= 10 else f"{x:.1f}"
    for unit, h in (("yr", 8766), ("d", 24), ("h", 1)):
        if hours >= h:
            return f"{sig(hours / h)} {unit}"
    return f"{sig(hours * 60)} min"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", default="fp16", choices=list(WIDTH))
    ap.add_argument("--fit-per-mbit", type=float, nargs="+", default=None,
                    help="sea-level soft-error rate(s) in FIT per Mbit (default: 0.56, modern DRAM; 1000 for 1990s DRAM)")
    ap.add_argument("--orbit-rate", type=float, default=ORBIT_PER_BIT_DAY,
                    help="low Earth orbit upsets per bit per day (default: 3e-7, middle of the measured 1e-7 to 1e-6)")
    ap.add_argument("--bits", type=float, default=None,
                    help="bits exposed in memory (default: GPT-2 weights only, params x dtype width)")
    args = ap.parse_args()

    bits = args.bits or GPT2_PARAMS * WIDTH[args.dtype]
    mbit = bits / 2**20
    fits = {f"{f:g} FIT/Mbit": f for f in args.fit_per_mbit} if args.fit_per_mbit else FIT_PER_MBIT
    p = p_lethal_per_flip(args.dtype)
    med = n50(args.dtype)

    print(f"GPT-2 {args.dtype}: {bits:,.0f} weight bits ({mbit:,.0f} Mbit)")
    print(f"median flips to death (sweep 1): {med:,.0f};  mean flips to death 1/p (sweeps 2+6): {1 / p:,.0f}\n")

    rows = []
    for label, fit in fits.items():
        for env, mult in FLUX.items():
            rows.append((env, f"{label} ({fit:g} FIT/Mbit) x{mult:g}" if label in FIT_PER_MBIT
                         else f"{label} x{mult:g}", fit * mbit * mult / 1e9))
    rows.append(("low Earth orbit", f"{args.orbit_rate:g} upsets/bit/day", args.orbit_rate * bits / 24))

    hdr = f"{'environment':<18} {'rate basis':<36} {'flips/hour':>11} {'median death':>13} {'mean death':>12}"
    print(hdr)
    print("-" * len(hdr))
    for env, basis, rate in rows:
        print(f"{env:<18} {basis:<36} {rate:>11.3g} {human(med / rate):>13} {human(1 / p / rate):>12}")


if __name__ == "__main__":
    main()
