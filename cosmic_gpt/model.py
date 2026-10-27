"""Model loading, fixed eval set, and perplexity."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
EVAL_TOKENS = 4096
WINDOW = 1024


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name: str, dtype: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=DTYPES[dtype], attn_implementation="eager")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def get_eval_tokens(tok, name: str) -> torch.Tensor:
    """First EVAL_TOKENS tokens of WikiText-2-raw test, as (EVAL_TOKENS // WINDOW, WINDOW)."""
    DATA.mkdir(exist_ok=True)
    cache = DATA / f"eval_tokens_{name.replace('/', '_')}.pt"
    if cache.exists():
        return torch.load(cache)
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text[:200_000], return_tensors="pt").input_ids[0][:EVAL_TOKENS]
    assert ids.numel() == EVAL_TOKENS
    ids = ids.view(-1, WINDOW).clone()
    torch.save(ids, cache)
    return ids


@torch.no_grad()
def perplexity(model, tokens: torch.Tensor) -> tuple[float, bool]:
    """Returns (ppl, nonfinite) where nonfinite is True if any logit is NaN/Inf."""
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    # Large vocabularies (e.g. Qwen's 151k) get one forward pass per window to bound memory.
    batches = tokens.split(1) if model.config.vocab_size > 100_000 else [tokens]
    nonfinite, nll = False, []
    for batch in batches:
        logits = model(batch).logits
        nonfinite |= not bool(torch.isfinite(logits).all())
        # Row by row in fp32 to bound memory.
        nll += [torch.nn.functional.cross_entropy(logits[i, :-1].float(), batch[i, 1:]) for i in range(len(batch))]
        del logits
    ppl = float(torch.exp(torch.stack(nll).mean()))
    return ppl, nonfinite
