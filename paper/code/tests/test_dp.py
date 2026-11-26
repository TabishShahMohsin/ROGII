"""The soft and hard segment-Viterbi agree with brute-force enumeration on tiny lattices."""
import torch
from ssv.dp import forward_backward, viterbi, brute_force, BIG


def _case(seed, KC=3, MV=1, B=2, C=4):
    g = torch.Generator().manual_seed(seed)
    NA, NM = 2 * KC + 1, 2 * MV + 1
    cost = torch.rand(B, C, NA, NM, generator=g) * 3
    tau = torch.rand(B, C, NM, NM, generator=g) * 2
    tau0 = torch.rand(B, NM, generator=g)
    AV = torch.arange(-KC, KC + 1)
    for k in range(NM):
        cost[:, :, (AV + k - MV).abs() > KC, k] = BIG
    return cost, tau, tau0, KC, MV


def test_soft_and_hard_match_enumeration():
    for seed in range(3):
        for T in (0.3, 1.0):
            cost, tau, tau0, KC, MV = _case(seed)
            S = forward_backward(cost.double(), tau.double(), tau0.double(), T, KC, MV)
            V = viterbi(cost, tau, tau0, KC, MV)
            for b, (soft, best) in enumerate(brute_force(cost.double(), tau.double(), tau0.double(), T, KC, MV)):
                assert torch.allclose(S[b].float(), soft, atol=1e-4), (S[b], soft)
                assert torch.equal(V[b], best), (V[b], best)


def test_zero_temperature_limit_is_viterbi():
    cost, tau, tau0, KC, MV = _case(7)
    S = forward_backward(cost.double(), tau.double(), tau0.double(), 1e-3, KC, MV)
    V = viterbi(cost, tau, tau0, KC, MV)
    assert torch.allclose(S.float(), V, atol=1e-3)


def test_gradient_flows():
    cost, tau, tau0, KC, MV = _case(3)
    cost = cost.clone().requires_grad_(True)
    S = forward_backward(cost, tau, tau0, 0.5, KC, MV)
    S.sum().backward()
    assert torch.isfinite(cost.grad).all() and cost.grad.abs().sum() > 0
