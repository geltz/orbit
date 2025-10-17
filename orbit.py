# ORBIT — Orthogonal Residual Blend In Tensors
# Injects model B's novelty that is orthogonal to model A (structure-preserving),
# plus a gentle parallel adjustment toward B's direction.

from sd_mecha import merge_method, Parameter, Return
from torch import Tensor
import torch

def _mad(x: Tensor, eps: Tensor) -> Tensor:
    flat = x.flatten()
    med = flat.median()
    return (flat - med).abs().median().clamp_min(eps)

def _trust_clamp(a: Tensor, y: Tensor, trust_k: float, eps: Tensor) -> Tensor:
    r = float(trust_k) * _mad(a, eps)
    return a + (y - a).clamp(-r, r)

def _finite_or_a(a: Tensor, y: Tensor) -> Tensor:
    return torch.where(torch.isfinite(y), y, a)

@merge_method
def orbit(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    # how much to move along A's direction vs inject orthogonal novelty
    alpha_par: Parameter(float) = 0.20,
    alpha_orth: Parameter(float) = 0.60,
    # robustness knobs
    trust_k: Parameter(float) = 3.0,
    eps: Parameter(float) = 1e-8,
    # optional: cap the parallel projection coefficient (stability)
    coef_clip: Parameter(float) = 8.0,
) -> Return(Tensor):
    """
    a: tensor from model A (kept as structure anchor)
    b: tensor from model B (feature donor)
    returns: merged tensor, dtype/device preserved
    """
    # scalar tensors in a's device/dtype
    eps_t = torch.as_tensor(float(eps), device=a.device, dtype=a.dtype)
    w_par = torch.as_tensor(float(alpha_par), device=a.device, dtype=a.dtype)
    w_orth = torch.as_tensor(float(alpha_orth), device=a.device, dtype=a.dtype)

    # compute parallel/orthogonal decomposition of B w.r.t. A
    af = a.flatten()
    bf = b.flatten()

    # coef = <B,A> / <A,A>
    denom = (af @ af).clamp_min(eps_t)
    coef = (bf @ af) / denom

    # optional stability: clip the projection coefficient
    if float(coef_clip) > 0.0:
        cmax = torch.as_tensor(float(coef_clip), device=a.device, dtype=a.dtype)
        coef = coef.clamp(-cmax, cmax)

    b_par = coef * a
    b_orth = b - b_par

    # blend: keep A, adjust slightly toward B along A (parallel),
    # and inject orthogonal novelty (structure-preserving change)
    y = a + w_par * (b_par - a) + w_orth * b_orth

    # robust trust clamp around A using MAD, and non-finite fallback
    y = _trust_clamp(a, y, trust_k, eps_t)
    y = _finite_or_a(a, y)
    return y
