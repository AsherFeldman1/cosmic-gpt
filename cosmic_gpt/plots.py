"""results/*.csv -> charts/*.png. Usage: python -m cosmic_gpt.plots"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, PercentFormatter

from cosmic_gpt.model import ROOT

RESULTS, CHARTS = ROOT / "results", ROOT / "charts"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#b5b3ac", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.axisbelow": True, "font.size": 11, "axes.labelsize": 11, "legend.frameon": False,
    "legend.fontsize": 11, "xtick.major.size": 0, "ytick.major.size": 0,
})


def friendly(x, _=None):
    """1, 10, 100, 1k, 10k, 100k, 1M"""
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if x >= div:
            return f"{x / div:g}{suffix}"
    return f"{x:g}"


def titled(fig, title, subtitle):
    fig.text(0.012, 0.975, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.012, 0.91, subtitle, fontsize=11, color=INK2, va="top")


def n50(df):
    """Flips at which half the runs are broken (log-interpolated)."""
    g = df.groupby("n_flips")["alive"].mean()
    x, y = np.log10(g.index.values.astype(float)), g.values
    i = np.flatnonzero(y < 0.5)[0]
    t = (y[i - 1] - 0.5) / (y[i - 1] - y[i])
    return 10 ** (x[i - 1] + t * (x[i] - x[i - 1]))


def working_lines(ax, series, xmax):
    """series: list of (label, df). % of runs still working vs flips, one line each."""
    for i, (label, df) in enumerate(series):
        g = df[df.n_flips <= xmax].groupby("n_flips")["alive"].mean() * 100
        ax.plot(g.index, g.values, color=SERIES[i], lw=2.5, marker="o", ms=6, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([10 ** k for k in range(int(np.log10(xmax)) + 1)]))
    ax.xaxis.set_major_formatter(FuncFormatter(friendly))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(0.8, xmax * 1.25)
    ax.set_ylim(-3, 103)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("number of random bits flipped (log scale)")
    ax.set_ylabel("runs still working")
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.legend(loc="upper right")


def save(fig, name):
    CHARTS.mkdir(exist_ok=True)
    fig.savefig(CHARTS / name, dpi=160)
    plt.close(fig)
    print("wrote", CHARTS / name)


def load(k):
    p = RESULTS / f"sweep{k}.csv"
    return pd.read_csv(p) if p.exists() else None


def line_figure():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(top=0.8, left=0.1, right=0.97, bottom=0.13)
    return fig, ax


def plot1(s1):
    fig, ax = line_figure()
    series = [(f"{lbl} ({d}): half broken at {n50(s1[s1.dtype == d]):.0f} flips", s1[s1.dtype == d])
              for d, lbl in [("fp32", "32-bit"), ("fp16", "16-bit"), ("bf16", "16-bit")]]
    working_lines(ax, series, 1000)
    titled(fig, "A dozen random bit flips break GPT-2",
           "Out of 2-4 billion bits of weights. 20 runs per point; every run with 300+ flips broke.")
    save(fig, "survival_vs_flips.png")


def plot2(s2, s6):
    d = s2[s2.dtype == "fp16"]
    broken = (1 - d.groupby("bit")["alive"].mean()) * 100
    single = (1 - s6[(s6.dtype == "fp16") & (s6.bit == 14)]["alive"].mean()) * 100
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(top=0.8, left=0.1, right=0.97, bottom=0.2)
    colors = [SERIES[1] if b == 14 else SERIES[0] for b in broken.index]
    ax.bar(broken.index, broken.values, width=0.8, color=colors)
    ax.bar_label(ax.containers[0], labels=[f"{v:.0f}%" for v in broken.values], padding=3, fontsize=9, color=INK2)
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.7, 15.7)
    ax.set_xticks(range(16))
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel("runs broken")
    ax.grid(axis="x", visible=False)
    ax.annotate(f"top exponent bit: a single flip\nhere breaks GPT-2 {single:.0f}% of the time",
                xy=(13.6, 96), xytext=(12.9, 70), fontsize=10, color=INK, ha="right",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    for lo, hi, name in [(0, 9, "fraction (bits 0-9)"), (10, 14, "exponent (bits 10-14)"), (15, 15, "sign")]:
        y = -0.1
        ax.annotate("", xy=(lo - 0.4, y), xytext=(hi + 0.4, y), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
        ax.text((lo + hi) / 2, y - 0.03, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=INK2, fontsize=10)
    titled(fig, "It's almost all one bit",
           "16-bit GPT-2. For each bit position, 100 random weights had that bit flipped (10 runs per bar).")
    save(fig, "bit_lethality.png")


GROUP_NAMES = {"embed": "Embeddings", "attn": "Attention", "mlp": "Feed-forward (MLP)", "ln": "LayerNorm"}


def plot3(s3, group_bits):
    order = ["mlp", "embed", "attn", "ln"]
    total = sum(group_bits[g] for g in order)
    share = [group_bits[g] / total * 100 for g in order]
    half = [n50(s3[s3.group == g]) for g in order]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    fig.subplots_adjust(top=0.7, left=0.19, right=0.97, bottom=0.04, wspace=0.12)
    y = np.arange(len(order))[::-1]
    a.barh(y, share, color=SERIES[0], height=0.6)
    a.bar_label(a.containers[0], labels=[f"{v:.2f}%" if v < 1 else f"{v:.0f}%" for v in share], padding=4)
    a.set_title("share of the model's memory", loc="left", fontsize=11, color=INK2)
    a.set_xlim(0, 60)
    b.barh(y, half, color=SERIES[1], height=0.6)
    b.bar_label(b.containers[0], labels=[f"{v:.0f} flips" for v in half], padding=4)
    b.set_title("flips in that part to break half the runs", loc="left", fontsize=11, color=INK2)
    b.set_xlim(0, 20)
    for ax in (a, b):
        ax.set_yticks(y, [GROUP_NAMES[g] for g in order])
        ax.grid(False)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    titled(fig, "Small parts of the model are just as deadly",
           "LayerNorm is 0.03% of GPT-2's memory but breaks after about as many flips as the rest,\n"
           "so per bit it is ~1,000x more fragile. 16-bit GPT-2, 20 runs per point.")
    save(fig, "group_survival.png")


def plot4(s1, s4):
    plain = s1[s1.dtype == "fp16"]
    fig, ax = line_figure()
    working_lines(ax, [(f"no fix: half broken at {n50(plain):,.0f} flips", plain),
                       (f"with clamping: half broken at {n50(s4):,.0f} flips", s4)], 1_000_000)
    titled(fig, "A cheap fix makes it ~100x tougher",
           "Clamp each weight back into its layer's original min-max range.\n"
           "Same flips in both lines. 16-bit GPT-2, 20 runs per point.")
    save(fig, "clamp.png")


def plot5(s1, s5):
    gpt = s1[s1.dtype == "fp16"]
    fig, ax = line_figure()
    working_lines(ax, [(f"GPT-2 (124M weights): half broken at {n50(gpt):.0f} flips", gpt),
                       (f"Qwen2.5 (494M weights): half broken at {n50(s5):.0f} flips", s5)], 1000)
    titled(fig, "A newer, 4x bigger model is no tougher",
           "Both 16-bit. 20 runs per point for GPT-2, 10 for Qwen2.5-0.5B.")
    save(fig, "modern_model.png")


def model_sizes() -> dict:
    """fp16 weight-bit counts per GPT-2 tensor group and per model, cached in results/sizes.json."""
    import json

    path = RESULTS / "sizes.json"
    if path.exists():
        return json.loads(path.read_text())
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from cosmic_gpt.inject import group_of
    from cosmic_gpt.sweeps import MODERN

    def params(name):
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name))
        return [(n, p.numel()) for n, p in m.named_parameters()]

    gpt2 = params("gpt2")
    group_bits = {}
    for n, k in gpt2:
        group_bits[group_of(n)] = group_bits.get(group_of(n), 0) + 16 * k
    sizes = {"gpt2_group_bits_fp16": group_bits, "gpt2_bits_fp16": 16 * sum(k for _, k in gpt2),
             "qwen_bits_fp16": 16 * sum(k for _, k in params(MODERN))}
    path.write_text(json.dumps(sizes, indent=2))
    return sizes


def main():
    s1, s2, s3, s4, s5, s6 = (load(k) for k in range(1, 7))
    sizes = model_sizes()
    if s1 is not None:
        plot1(s1)
    if s2 is not None and s6 is not None:
        plot2(s2, s6)
    if s3 is not None:
        plot3(s3, sizes["gpt2_group_bits_fp16"])
    if s1 is not None and s4 is not None:
        plot4(s1, s4)
    if s1 is not None and s5 is not None:
        plot5(s1, s5)


if __name__ == "__main__":
    main()
