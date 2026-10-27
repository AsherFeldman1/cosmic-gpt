"""results/*.csv -> charts/*.png. Usage: python -m cosmic_gpt.plots"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cosmic_gpt.model import ROOT

RESULTS, CHARTS = ROOT / "results", ROOT / "charts"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False, "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 11,
})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def survival(df, by="n_flips"):
    g = df.groupby(by)["alive"].agg(["sum", "count"]).reset_index()
    g["frac"] = g["sum"] / g["count"]
    g["lo"], g["hi"] = wilson(g["sum"], g["count"])
    return g


def survival_lines(ax, series, x="n_flips", xlabel="bit flips (log scale)"):
    """series: list of (label, df). Lines with Wilson 95% CI bands + right-end direct labels."""
    for i, (label, df) in enumerate(series):
        g = survival(df, x)
        c = SERIES[i]
        xs = g[x] * 1.06 ** (i - (len(series) - 1) / 2)  # small log-x dodge so CI bars don't overlap
        ax.errorbar(xs, g["frac"], yerr=[g["frac"] - g["lo"], g["hi"] - g["frac"]], color=c, lw=2, marker="o",
                    ms=4, elinewidth=1, capsize=0, alpha=0.95, label=label)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("survival fraction")
    ax.legend(loc="upper right")


def save(fig, name):
    CHARTS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHARTS / name, dpi=150)
    plt.close(fig)
    print("wrote", CHARTS / name)


def load(k):
    p = RESULTS / f"sweep{k}.csv"
    return pd.read_csv(p) if p.exists() else None


def plot1(s1):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    survival_lines(ax, [(d, s1[s1.dtype == d]) for d in ["fp32", "fp16", "bf16"]])
    ax.set_title("GPT-2 survival vs random weight bit flips (20 seeds, bars = 95% Wilson CI)")
    save(fig, "survival_vs_flips.png")


def bit_label(b, dtype):
    exp_bits = {"fp32": 8, "fp16": 5, "bf16": 8}[dtype]
    width = 32 if dtype == "fp32" else 16
    if b == width - 1:
        return "sign"
    return "exp" if b >= width - 1 - exp_bits else "mant"


def plot2(s2):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5))
    for ax, dtype in zip(axes, ["fp32", "fp16", "bf16"]):
        d = s2[s2.dtype == dtype]
        g = d.groupby("bit").agg(hard=("hard_dead", "mean"), dead=("alive", lambda a: 1 - a.mean())).reset_index()
        soft_only = g["dead"] - g["hard"]
        ax.bar(g["bit"], g["hard"], width=0.8, color=SERIES[0], label="hard death (NaN/Inf)", edgecolor=SURFACE, lw=1)
        ax.bar(g["bit"], soft_only, bottom=g["hard"], width=0.8, color=SERIES[1], label="soft death only (ppl > 2x)",
               edgecolor=SURFACE, lw=1)
        width = 32 if dtype == "fp32" else 16
        ax.set_xticks(range(width))
        ax.set_xticklabels([f"{b}\n{bit_label(b, dtype)}" if b in (0, width - 1) or bit_label(b, dtype) != bit_label(b - 1, dtype)
                            else str(b) for b in range(width)], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("death rate")
        ax.set_title(f"{dtype}: flip 100 random weights at bit b (10 seeds)")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("bit position (0 = mantissa LSB)")
    save(fig, "bit_lethality.png")


def plot3(s3, group_bits):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    groups = ["embed", "ln", "attn", "mlp"]
    series = [(f"{g} ({group_bits[g] / 1e6:.1f} Mbit)", s3[s3.group == g]) for g in groups]
    survival_lines(axes[0], series)
    axes[0].set_title("fp16: flips restricted to one tensor group")
    s3 = s3.assign(per_mbit=s3.n_flips / s3.group.map(group_bits) * 1e6)
    survival_lines(axes[1], [(lbl, s3[s3.group == g]) for (lbl, _), g in zip(series, groups)],
                   x="per_mbit", xlabel="flips per Mbit of the group (log scale)")
    axes[1].set_title("same, normalized by group size")
    save(fig, "group_survival.png")


def plot4(s1, s4):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    survival_lines(ax, [("fp16, no mitigation", s1[s1.dtype == "fp16"]), ("fp16 + clamp to clean [min,max]", s4)])
    ax.set_title("Mitigation: clamp weights to clean per-tensor range (same flips)")
    save(fig, "clamp.png")


def plot5(s1, s5, bits):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    gpt = s1[s1.dtype == "fp16"]
    series = [("GPT-2 124M fp16", gpt), ("Qwen2.5-0.5B fp16", s5)]
    survival_lines(axes[0], series)
    axes[0].set_title("GPT-2 vs a modern 0.5B model")
    gpt = gpt.assign(per_gbit=gpt.n_flips / bits["gpt2"] * 1e9)
    s5 = s5.assign(per_gbit=s5.n_flips / bits["qwen"] * 1e9)
    survival_lines(axes[1], [("GPT-2 124M fp16", gpt), ("Qwen2.5-0.5B fp16", s5)],
                   x="per_gbit", xlabel="flips per Gbit of weights (log scale)")
    axes[1].set_title("same, normalized by model size")
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
    s1, s2, s3, s4, s5 = (load(k) for k in range(1, 6))
    sizes = model_sizes()
    if s1 is not None:
        plot1(s1)
    if s2 is not None:
        plot2(s2)
    if s3 is not None:
        plot3(s3, sizes["gpt2_group_bits_fp16"])
    if s1 is not None and s4 is not None:
        plot4(s1, s4)
    if s1 is not None and s5 is not None:
        plot5(s1, s5, {"gpt2": sizes["gpt2_bits_fp16"], "qwen": sizes["qwen_bits_fp16"]})


if __name__ == "__main__":
    main()
