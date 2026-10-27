"""Cosmic-ray time-to-death estimate for a model sitting in (unprotected) memory.

Usage: python -m cosmic_gpt.cosmic [--fit-per-mbit 1000] [--dtype fp16] [--bits N]

flips/hour = FIT/Mbit * Mbit_in_memory * altitude_multiplier / 1e9
(1 FIT = 1 failure per 1e9 device-hours.)

Lethality comes from the experiments:
  * p-model: sweep 2 gives the death rate D_b when 100 weights are flipped at
    bit b. Assuming independent flips, per-flip lethality at bit b is
    p_b = 1 - (1 - D_b)^(1/100). For bits where 100 flips (nearly) always kill,
    that formula can't resolve p_b, so sweep 6 measures P(death | 1 flip at b)
    directly and replaces it. A random flip lands on a uniformly random bit,
    so p = mean_b p_b and expected flips-to-death = 1/p.
  * N50: sweep 1's empirical median-lethal number of flips (log-interpolated),
    a direct cross-check that doesn't assume independence.
"""

import argparse

import numpy as np
import pandas as pd

from cosmic_gpt.model import ROOT

RESULTS = ROOT / "results"

# TODO(you): cite a source for the default soft-error rate. Placeholder value only;
# published DRAM/SRAM figures span orders of magnitude by process node and ECC.
DEFAULT_FIT_PER_MBIT = 1000.0

# TODO(you): cite sources for these neutron/particle flux multipliers relative to sea level.
# Placeholder round numbers. The ISS case is a different radiation environment
# (trapped protons, GCR heavy ions, SAA passes), so a single multiplier is a crude stand-in.
ALTITUDES = {
    "sea level": 1.0,
    "Denver (1.6 km)": 3.5,
    "airliner (35k ft)": 300.0,
    "ISS (LEO)": 1000.0,
}

GPT2_PARAMS = 124_439_808
WIDTH = {"fp32": 32, "fp16": 16, "bf16": 16}


def p_lethal_per_flip(dtype: str) -> tuple[float, int]:
    s2 = pd.read_csv(RESULTS / "sweep2.csv")
    d = s2[s2.dtype == dtype]
    g = d.groupby("bit")["alive"].agg(["sum", "count"])
    n_flips = int(d.n_flips.iloc[0])
    dead = 1 - g["sum"] / g["count"]
    # All-dead bits would give p_b = 1, which over-states it; cap at the
    # resolution of the seed count (half a seed short of total death).
    dead = dead.clip(upper=1 - 0.5 / g["count"])
    p_b = 1 - (1 - dead) ** (1 / n_flips)
    # Saturated bits: replace with the directly measured single-flip death rate (sweep 6).
    s6_path = RESULTS / "sweep6.csv"
    if s6_path.exists():
        s6 = pd.read_csv(s6_path)
        direct = 1 - s6[s6.dtype == dtype].groupby("bit")["alive"].mean()
        p_b.loc[direct.index] = direct
    return float(p_b.mean()), len(g)


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
    if not np.isfinite(hours):
        return "never"
    for unit, h in (("yr", 8766), ("d", 24), ("h", 1)):
        if hours >= h:
            return f"{hours / h:,.1f} {unit}"
    return f"{hours * 60:,.1f} min"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-per-mbit", type=float, default=DEFAULT_FIT_PER_MBIT,
                    help=f"soft-error rate in FIT per Mbit (default {DEFAULT_FIT_PER_MBIT:g}, placeholder - see TODO)")
    ap.add_argument("--dtype", default="fp16", choices=list(WIDTH))
    ap.add_argument("--bits", type=float, default=None,
                    help="bits exposed in memory (default: GPT-2 weights only, params x dtype width)")
    args = ap.parse_args()

    bits = args.bits or GPT2_PARAMS * WIDTH[args.dtype]
    mbit = bits / 2**20
    p, nbits = p_lethal_per_flip(args.dtype)
    med = n50(args.dtype)

    print(f"GPT-2 {args.dtype}: {bits:,.0f} bits ({mbit:,.0f} Mbit) at {args.fit_per_mbit:g} FIT/Mbit")
    print(f"p_lethal per random flip (sweep 2, mean over {nbits} bit positions): {p:.2e}"
          f"  -> expected flips to death 1/p = {1 / p:,.0f}")
    print(f"empirical N50 (sweep 1, median lethal flips): {med:,.0f}\n")
    hdr = f"{'environment':<20} {'mult':>7} {'flips/hour':>12} {'death (1/p)':>14} {'death (N50)':>14}"
    print(hdr)
    print("-" * len(hdr))
    for name, mult in ALTITUDES.items():
        rate = args.fit_per_mbit * mbit * mult / 1e9
        print(f"{name:<20} {mult:>7g} {rate:>12.3g} {human(1 / p / rate):>14} {human(med / rate):>14}")


if __name__ == "__main__":
    main()
