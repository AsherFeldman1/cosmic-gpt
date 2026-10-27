"""Bit-flip fault injection on model weights."""

import numpy as np
import torch

GROUPS = ("embed", "ln", "attn", "mlp")

INT_VIEW = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}
WIDTH = {torch.float32: 32, torch.float16: 16, torch.bfloat16: 16}


def group_of(name: str) -> str:
    """Tensor group for a parameter name (GPT-2 and Llama/Qwen-style naming)."""
    if any(k in name for k in ("wte", "wpe", "embed", "lm_head")):
        return "embed"
    if "ln_" in name or "norm" in name:
        return "ln"
    if "attn" in name:
        return "attn"
    if "mlp" in name:
        return "mlp"
    return "other"


class Injector:
    """Holds the parameter universe and a clean copy of every weight.

    named_parameters() dedupes tied weights (GPT-2 wte/lm_head), so each
    physical weight appears exactly once. Buffers (e.g. causal masks) are excluded.
    """

    def __init__(self, model):
        self.model = model
        self.names, self.params = zip(*model.named_parameters())
        self.groups = np.array([group_of(n) for n in self.names])
        self.numel = np.array([p.numel() for p in self.params], dtype=np.int64)
        dtype = self.params[0].dtype
        assert all(p.dtype == dtype for p in self.params)
        self.dtype = dtype
        self.width = WIDTH[dtype]
        self.int_dtype = INT_VIEW[dtype]
        self.clean = [p.detach().clone() for p in self.params]
        self.lo = [float(c.min()) for c in self.clean]
        self.hi = [float(c.max()) for c in self.clean]
        self.touched: set[int] = set()

    @property
    def total_bits(self) -> int:
        return int(self.numel.sum()) * self.width

    def restore(self) -> None:
        """Restore every tensor modified since the last restore to its clean value."""
        with torch.no_grad():
            for i in self.touched:
                self.params[i].copy_(self.clean[i])
        self.touched.clear()

    def flip_bits(self, n: int, rng: np.random.Generator, bit: int | None = None, group: str | None = None) -> dict:
        """XOR n distinct, uniformly random weight bits in place.

        bit:   restrict to bit position b (0 = LSB of mantissa, width-1 = sign);
               then n distinct elements are chosen, each flipped at bit b.
        group: restrict to one of GROUPS.
        Returns the exact flip list: param index, flat element index, bit position.
        """
        pidx = np.flatnonzero(self.groups == group) if group else np.arange(len(self.params))
        sizes = self.numel[pidx]
        cum = np.concatenate([[0], np.cumsum(sizes)])
        if bit is None:
            g = _sample_distinct(rng, int(cum[-1]) * self.width, n)
            elem_global, bits = g // self.width, (g % self.width).astype(np.int8)
        else:
            elem_global = _sample_distinct(rng, int(cum[-1]), n)
            bits = np.full(elem_global.shape, bit, dtype=np.int8)
        local = np.searchsorted(cum, elem_global, side="right") - 1
        flips = {
            "param": pidx[local].astype(np.int32),
            "elem": (elem_global - cum[local]).astype(np.int64),
            "bit": bits,
        }
        self.apply(flips)
        return flips

    def apply(self, flips: dict) -> None:
        """Apply a flip list (also used to replay a logged trial)."""
        param, elem, bit = flips["param"], flips["elem"], flips["bit"]
        if param.size == 0:
            return
        # Combine multiple flipped bits within the same element into one mask.
        key = param.astype(np.int64) * (1 << 40) + elem
        order = np.argsort(key, kind="stable")
        key, param, elem = key[order], param[order], elem[order]
        masks = np.left_shift(np.int64(1), bit[order].astype(np.int64))
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        masks = np.bitwise_or.reduceat(masks, starts)
        param, elem = param[starts], elem[starts]
        masks = np.where(masks >= 1 << (self.width - 1), masks - (1 << self.width), masks)  # to signed

        pstarts = np.flatnonzero(np.r_[True, param[1:] != param[:-1]])
        pends = np.r_[pstarts[1:], param.size]
        with torch.no_grad():
            for s, e in zip(pstarts, pends):
                i = int(param[s])
                self._xor(self.params[i], elem[s:e], masks[s:e])
                self.touched.add(i)

    def _xor(self, p: torch.Tensor, idx: np.ndarray, masks: np.ndarray) -> None:
        try:
            dev = p.device
            flat = p.data.view(-1).view(self.int_dtype)
            t_idx = torch.from_numpy(idx).to(dev)
            t_mask = torch.from_numpy(masks).to(self.int_dtype).to(dev)
            flat[t_idx] = flat[t_idx] ^ t_mask
        except (RuntimeError, NotImplementedError):
            # Fallback: do the XOR on a CPU copy if the device lacks an int op.
            cpu = p.data.detach().cpu().view(-1).view(self.int_dtype)
            t_idx = torch.from_numpy(idx)
            cpu[t_idx] = cpu[t_idx] ^ torch.from_numpy(masks).to(self.int_dtype)
            p.data.copy_(cpu.view(p.dtype).view(p.shape))

    def clamp_touched(self) -> None:
        """Mitigation: NaN -> 0, then clamp each modified tensor to its clean [min, max]."""
        with torch.no_grad():
            for i in self.touched:
                p = self.params[i]
                p.nan_to_num_(nan=0.0)
                p.clamp_(self.lo[i], self.hi[i])


def _sample_distinct(rng: np.random.Generator, high: int, n: int) -> np.ndarray:
    """n distinct uniform integers in [0, high) (n is capped at high)."""
    n = min(n, high)
    if n > high // 4:
        return rng.permutation(high)[:n].astype(np.int64)
    out = np.unique(rng.integers(0, high, size=n, dtype=np.int64))
    while out.size < n:
        out = np.unique(np.concatenate([out, rng.integers(0, high, size=n - out.size, dtype=np.int64)]))
    return rng.permutation(out)
