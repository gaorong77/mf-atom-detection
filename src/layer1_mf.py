"""
layer1_mf.py  --  L1: Matched Filter fast-decision layer (PA-3HDA-MF v1.0).

Key design: CUSUM is gated on *previous-frame* Bayesian belief.
  - Only accumulates loss evidence when log_odds_{t-1} > 0 (atom believed present).
  - Prevents false ERASURE for perpetually empty sites.
  - L1.5 router removed; direct L2 routing for uncertain sites.
"""

import numpy as np
from numpy.typing import NDArray
from enum import IntEnum

LOG_ODDS_CLIP = 15.0


class Decision(IntEnum):
    ATOM_PRESENT = 1
    ATOM_ABSENT  = 0
    ROUTE_L2     = -1
    ERASURE_LOSS = 2
    CORR_LOSS    = 3
    DRIFT_ALARM  = 4


class L1State:
    def __init__(self, n_sites: int):
        self.n_sites = n_sites
        self.log_odds   = np.zeros(n_sites, dtype=np.float64)
        self.S_cusum    = np.zeros(n_sites, dtype=np.float64)
        self.frame_count: int = 0

    def reset(self) -> None:
        self.log_odds[:] = 0.0
        self.S_cusum[:]  = 0.0
        self.frame_count  = 0


def _safe_logit64(p: NDArray, eps: float = 1e-9) -> NDArray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def run_l1(p_mf, state, theta_H, theta_L,
           p_survival, lambda_obs, h_loss, h_corr, warmup):
    """
    Returns (decisions: int8 K, p_post: float32 K).
    """
    K = state.n_sites
    state.frame_count += 1
    use_temporal = state.frame_count > warmup

    p_mf = p_mf.astype(np.float64)

    # -- Save previous belief for CUSUM gating
    prev_log_odds = state.log_odds.copy()

    # -- Bayesian log-odds update
    survival_logit = np.log(p_survival / (1.0 - p_survival))
    obs_logit      = lambda_obs * _safe_logit64(p_mf)

    if use_temporal:
        raw = state.log_odds + survival_logit + obs_logit
    else:
        raw = obs_logit  # warmup: single-frame fallback

    state.log_odds = np.clip(raw, -LOG_ODDS_CLIP, LOG_ODDS_CLIP)

    p_post = (1.0 / (1.0 + np.exp(-state.log_odds))).astype(np.float32)
    p_post = np.clip(p_post, 1e-7, 1.0 - 1e-7)

    # -- CUSUM update (gated: only when prev belief > 0)
    cusum_incr = _safe_logit64(1.0 - p_mf)          # log((1-p)/p) > 0 when no atom
    cusum_incr = np.clip(cusum_incr, -LOG_ODDS_CLIP, LOG_ODDS_CLIP)

    # Gate: accumulate only when previous frame believed atom was present
    gate = (prev_log_odds > 0.0).astype(np.float64)
    state.S_cusum = np.maximum(0.0, state.S_cusum + gate * cusum_incr)

    # -- Decision (priority: ERASURE > ACCEPT > REJECT > ROUTE_L2)
    decisions = np.full(K, int(Decision.ROUTE_L2), dtype=np.int8)

    erasure_mask = state.S_cusum > h_loss
    decisions[erasure_mask] = int(Decision.ERASURE_LOSS)
    state.S_cusum[erasure_mask] = 0.0       # reset after trigger

    accept_mask = (~erasure_mask) & (p_post >= theta_H)
    reject_mask = (~erasure_mask) & (p_post <= theta_L)
    decisions[accept_mask] = int(Decision.ATOM_PRESENT)
    decisions[reject_mask] = int(Decision.ATOM_ABSENT)
    # Reset CUSUM for confidently absent sites
    state.S_cusum[reject_mask] = 0.0

    return decisions, p_post
