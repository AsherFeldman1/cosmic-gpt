"""Fault-injection sweeps. Usage: python -m cosmic_gpt.sweeps {1,2,3,4,5,all} [--device mps]

Every sweep appends to results/sweep<k>.csv and skips trials already present,
so it can be interrupted and resumed. Every trial's exact flip list is written
to results/flips/ (trials with > SMALL flips go under results/flips/big/, which
is gitignored; replay any trial bit-exactly with --replay <trial_id>).
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from cosmic_gpt.inject import GROUPS, Injector
from cosmic_gpt.model import DTYPES, ROOT, default_device, get_eval_tokens, load_model, perplexity, seed_everything

RESULTS = ROOT / "results"
FLIPS = RESULTS / "flips"
SMALL = 1000

N_GRID = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
GROUP_N_GRID = [n for n in N_GRID if n <= 100_000]
GPT2 = "gpt2"
MODERN = "Qwen/Qwen2.5-0.5B"

DTYPE_ID = {d: i for i, d in enumerate(DTYPES)}
GROUP_ID = {g: i + 1 for i, g in enumerate(GROUPS)}

FIELDS = ["trial_id", "sweep", "model", "dtype", "n_flips", "seed", "bit", "group", "clamp",
          "n_applied", "ppl", "baseline_ppl", "hard_dead", "soft_dead", "alive", "seconds"]


def trial_rng(sweep: int, dtype: str, n: int, seed: int, extra: int = 0) -> np.random.Generator:
    return np.random.default_rng([sweep, DTYPE_ID[dtype], n, seed, extra])


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {row["trial_id"] for row in csv.DictReader(f)}


class Runner:
    def __init__(self, model_name: str, dtype: str, device: str):
        seed_everything(0)
        self.model_name, self.dtype = model_name, dtype
        self.model, tok = load_model(model_name, dtype, device)
        self.tokens = get_eval_tokens(tok, model_name)
        self.inj = Injector(self.model)
        self.baseline, nonfinite = perplexity(self.model, self.tokens)
        assert not nonfinite
        print(f"[{model_name} {dtype}] baseline ppl {self.baseline:.3f}, {self.inj.total_bits / 1e9:.2f} Gbit", flush=True)

    def trial(self, sweep: int, trial_id: str, n: int, seed: int, rng, bit=None, group=None, clamp=False) -> dict:
        t0 = time.time()
        self.inj.restore()
        flips = self.inj.flip_bits(n, rng, bit=bit, group=group)
        if clamp:
            self.inj.clamp_touched()
        ppl, nonfinite = perplexity(self.model, self.tokens)
        self.inj.restore()
        save_flips(sweep, trial_id, flips, self.inj.names)
        soft = not (ppl <= 2 * self.baseline)  # NaN ppl counts as dead
        return dict(trial_id=trial_id, sweep=sweep, model=self.model_name, dtype=self.dtype, n_flips=n, seed=seed,
                    bit="" if bit is None else bit, group=group or "", clamp=int(clamp),
                    n_applied=len(flips["bit"]), ppl=ppl, baseline_ppl=self.baseline,
                    hard_dead=int(nonfinite), soft_dead=int(soft), alive=int(not (soft or nonfinite)),
                    seconds=round(time.time() - t0, 3))


def save_flips(sweep: int, trial_id: str, flips: dict, names) -> None:
    big = len(flips["bit"]) > SMALL
    d = FLIPS / ("big" if big else "") / f"sweep{sweep}"
    d.mkdir(parents=True, exist_ok=True)
    pnames = np.array([names[i] for i in np.unique(flips["param"])]) if not big else np.array([])
    np.savez_compressed(d / f"{trial_id}.npz", param_names_used=pnames, **flips)


def run(sweep: int, model_name: str, dtypes, jobs, device: str, **kw) -> None:
    """jobs: list of (trial_id_suffix, n, seed, rng_extra, trial kwargs)."""
    path = RESULTS / f"sweep{sweep}.csv"
    done = load_done(path)
    rng_sweep = kw.pop("rng_sweep", sweep)
    for dtype in dtypes:
        todo = [j for j in jobs if f"s{sweep}_{dtype}_{j[0]}" not in done]
        if not todo:
            continue
        runner = Runner(model_name, dtype, device)
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            t0 = time.time()
            for k, (suffix, n, seed, extra, tkw) in enumerate(todo):
                tid = f"s{sweep}_{dtype}_{suffix}"
                row = runner.trial(sweep, tid, n, seed, trial_rng(rng_sweep, dtype, n, seed, extra), **tkw, **kw)
                w.writerow(row)
                f.flush()
                if k % 50 == 0 or k == len(todo) - 1:
                    print(f"  sweep{sweep} {dtype} {k + 1}/{len(todo)} {tid} ppl={row['ppl']:.3g} "
                          f"alive={row['alive']} ({time.time() - t0:.0f}s)", flush=True)
        del runner


def grid_jobs(grid, seeds, extra=0, tag="", **tkw):
    return [(f"{tag}n{n}_seed{s}", n, s, extra, tkw) for n in grid for s in range(seeds)]


def sweep1(device):
    run(1, GPT2, ["fp32", "fp16", "bf16"], grid_jobs(N_GRID, 20), device)


def sweep2(device):
    for dtype in ["fp32", "fp16", "bf16"]:
        width = 32 if dtype == "fp32" else 16
        jobs = [(f"bit{b}_seed{s}", 100, s, 100 + b, {"bit": b}) for b in range(width) for s in range(10)]
        run(2, GPT2, [dtype], jobs, device)


def sweep3(device):
    jobs = [j for g in GROUPS for j in grid_jobs(GROUP_N_GRID, 20, extra=GROUP_ID[g], tag=f"{g}_", group=g)]
    run(3, GPT2, ["fp16"], jobs, device)


def sweep4(device):
    # Same RNG streams as sweep 1 fp16, so clamp vs no-clamp see identical flips.
    run(4, GPT2, ["fp16"], grid_jobs(N_GRID, 20), device, clamp=True, rng_sweep=1)


def sweep5(device):
    run(5, MODERN, ["fp16"], grid_jobs(N_GRID, 10), device)


def sweep6(device):
    """Single-flip lethality at bits that sweep 2 found saturated (>= 50% death at 100 flips).

    Sweep 2 can't resolve per-flip lethality once 100 flips always kill, so for
    those bits we measure P(death | 1 flip) directly. Used by cosmic.py.
    """
    import pandas as pd

    s2 = pd.read_csv(RESULTS / "sweep2.csv")
    for dtype in ["fp32", "fp16", "bf16"]:
        dead = 1 - s2[s2.dtype == dtype].groupby("bit")["alive"].mean()
        jobs = [(f"bit{b}_seed{s}", 1, s, 200 + b, {"bit": int(b)}) for b in dead.index[dead >= 0.5] for s in range(50)]
        run(6, GPT2, [dtype], jobs, device)


def replay(trial_id: str, device: str) -> None:
    import pandas as pd

    sweep = int(trial_id.split("_")[0][1:])
    row = pd.read_csv(RESULTS / f"sweep{sweep}.csv").set_index("trial_id").loc[trial_id]
    path = next(FLIPS.rglob(f"{trial_id}.npz"), None)
    runner = Runner(row["model"], row["dtype"], device)
    if path is not None:
        flips = dict(np.load(path))
        runner.inj.apply({k: flips[k] for k in ("param", "elem", "bit")})
    else:
        bit = None if pd.isna(row["bit"]) or row["bit"] == "" else int(row["bit"])
        group = row["group"] if isinstance(row["group"], str) else None
        extra = (200 if sweep == 6 else 100) + bit if bit is not None else GROUP_ID.get(group, 0)
        rng = trial_rng(1 if sweep == 4 else sweep, row["dtype"], int(row["n_flips"]), int(row["seed"]), extra)
        runner.inj.flip_bits(int(row["n_flips"]), rng, bit=bit, group=group)
    if row["clamp"]:
        runner.inj.clamp_touched()
    print(f"replayed {trial_id}: ppl {perplexity(runner.model, runner.tokens)} (logged {row['ppl']})")


SWEEPS = {"1": sweep1, "2": sweep2, "3": sweep3, "4": sweep4, "5": sweep5, "6": sweep6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", nargs="?", default="all", choices=[*SWEEPS, "all"])
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--replay", metavar="TRIAL_ID")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.replay:
        return replay(args.replay, args.device)
    t0 = time.time()
    for k in SWEEPS if args.sweep == "all" else [args.sweep]:
        s = time.time()
        SWEEPS[k](args.device)
        print(f"sweep {k} done in {time.time() - s:.0f}s", flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
