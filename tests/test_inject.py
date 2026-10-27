import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from cosmic_gpt.inject import Injector


def tiny(dtype):
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=50, n_positions=32)
    return GPT2LMHeadModel(cfg).to(dtype).eval()


def bits(inj):
    return [p.detach().view(-1).view(inj.int_dtype).clone() for p in inj.params]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_flip_twice_is_identity_and_counts(dtype):
    inj = Injector(tiny(dtype))
    before = bits(inj)
    flips = inj.flip_bits(200, np.random.default_rng(1))
    after = bits(inj)
    changed = sum(int((a ^ b).to(torch.int64).bitwise_and((1 << inj.width) - 1).ne(0).sum()) for a, b in zip(before, after))
    assert 0 < changed <= 200
    popcount = 0
    for a, b in zip(before, after):
        x = (a ^ b).to(torch.int64) & ((1 << inj.width) - 1)
        popcount += sum(bin(int(v)).count("1") for v in x[x != 0])
    assert popcount == 200
    inj.apply(flips)
    assert all(torch.equal(a, b) for a, b in zip(before, bits(inj)))


def test_restore():
    inj = Injector(tiny(torch.float16))
    before = bits(inj)
    inj.flip_bits(1000, np.random.default_rng(2))
    inj.restore()
    assert all(torch.equal(a, b) for a, b in zip(before, bits(inj)))


@pytest.mark.parametrize("b", [0, 7, 15])
def test_bit_restriction(b):
    inj = Injector(tiny(torch.float16))
    before = bits(inj)
    flips = inj.flip_bits(50, np.random.default_rng(3), bit=b)
    assert (flips["bit"] == b).all()
    for x, y in zip(before, bits(inj)):
        d = (x ^ y).to(torch.int64) & 0xFFFF
        assert ((d == 0) | (d == (1 << b))).all()


@pytest.mark.parametrize("group", ["embed", "ln", "attn", "mlp"])
def test_group_restriction(group):
    inj = Injector(tiny(torch.float32))
    flips = inj.flip_bits(30, np.random.default_rng(4), group=group)
    assert all(inj.groups[i] == group for i in np.unique(flips["param"]))


def test_sign_bit_negates_fp32():
    inj = Injector(tiny(torch.float32))
    before = [p.detach().clone() for p in inj.params]
    flips = inj.flip_bits(10, np.random.default_rng(5), bit=31)
    for i, e in zip(flips["param"], flips["elem"]):
        assert inj.params[i].view(-1)[e] == -before[i].view(-1)[e]
