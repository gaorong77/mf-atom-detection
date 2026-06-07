"""
system.py  --  MF-Array two-layer detection system (PA-3HDA-MF v1.0).

Architecture (L1.5 removed):
    L0  : Drift awareness (ROI offset correction, background thread -- stubbed here)
    L1  : MF-Array fast layer + Bayesian temporal fusion + CUSUM
    L2  : MF-Array precise layer (long exposure)
    Out : 5-type structured output  {ATOM_PRESENT, ATOM_ABSENT, ERASURE_LOSS,
                                     CORR_LOSS, DRIFT_ALARM}
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from src.config import SystemConfig
from src.psf import build_template_array
from src.mf_detector import (compute_mf_scores, scores_to_probs,
                               extract_rois)
from src.layer1_mf import L1State, run_l1, Decision
from src.layer2_mf import run_l2


@dataclass
class FrameResult:
    """Per-frame detection result for all K sites."""
    decisions:   NDArray[np.int8]    # (K,)  Decision enum value
    p_post:      NDArray[np.float32] # (K,)  L1 posterior probability
    l2_routed:   NDArray[np.bool_]   # (K,)  mask of sites routed to L2
    n_erasure:   int = 0
    n_l2_routed: int = 0


class MFSystem:
    """Two-layer Matched Filter detection system."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        K = cfg.array.n_sites

        # Build site center coordinates
        rows = np.arange(cfg.array.n_rows)
        cols = np.arange(cfg.array.n_cols)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        pitch = cfg.array.site_pitch_px
        margin = pitch
        self.site_centers = np.stack(
            [rr.ravel() * pitch + margin,
             cc.ravel() * pitch + margin], axis=1
        ).astype(np.int32)   # (K, 2)

        # Build MF templates
        self.roi_offsets: NDArray[np.float32] = np.zeros((K, 2), dtype=np.float32)
        self.templates = build_template_array(cfg.array, cfg.psf,
                                              roi_offsets=None)  # (K, R, R)

        # L1 state
        self.l1_state = L1State(K)

        # MF calibration params (set after calibrate() or from cfg.mf)
        self._mu_atom    = cfg.mf.mu_atom
        self._sigma_atom = cfg.mf.sigma_atom
        self._mu_bg      = cfg.mf.mu_bg
        self._sigma_bg   = cfg.mf.sigma_bg

    def set_mf_params(self, mu_atom: float, sigma_atom: float,
                      mu_bg: float, sigma_bg: float) -> None:
        self._mu_atom    = mu_atom
        self._sigma_atom = sigma_atom
        self._mu_bg      = mu_bg
        self._sigma_bg   = sigma_bg

    def process_frame(self,
                      frame_l1: NDArray[np.float32],
                      frame_l2: NDArray[np.float32] | None = None
                      ) -> FrameResult:
        """
        Process one imaging cycle.

        Parameters
        ----------
        frame_l1 : (H, W) float32  -- short-exposure fluorescence image
        frame_l2 : (H, W) float32 | None  --  long-exposure image
                   (only required for sites routed to L2; may be None if
                   the caller guarantees no L2 routing)

        Returns
        -------
        FrameResult
        """
        cfg = self.cfg

        # --- L1: extract ROIs and compute MF scores ---
        rois_l1 = extract_rois(frame_l1, self.site_centers,
                                cfg.array.roi_size, self.roi_offsets)
        scores_l1 = compute_mf_scores(rois_l1, self.templates)
        p_mf_l1   = scores_to_probs(scores_l1,
                                     self._mu_atom, self._sigma_atom,
                                     self._mu_bg,   self._sigma_bg)

        # --- L1: temporal fusion + CUSUM + threshold decision ---
        decisions, p_post = run_l1(
            p_mf        = p_mf_l1,
            state       = self.l1_state,
            theta_H     = cfg.l1.theta_H,
            theta_L     = cfg.l1.theta_L,
            p_survival  = cfg.physics.p_survival,
            lambda_obs  = cfg.l1.lambda_obs,
            h_loss      = cfg.cusum.h_loss,
            h_corr      = cfg.cusum.h_corr,
            warmup      = cfg.l1.warmup_frames,
        )

        # --- L2: precise MF for uncertain sites ---
        l2_mask = decisions == int(Decision.ROUTE_L2)
        n_l2    = int(l2_mask.sum())

        if n_l2 > 0 and frame_l2 is not None:
            l2_indices = np.where(l2_mask)[0].astype(np.int64)
            rois_l2 = extract_rois(frame_l2, self.site_centers[l2_indices],
                                   cfg.array.roi_size,
                                   self.roi_offsets[l2_indices])
            # Use same templates for L2 (longer exposure -> same PSF shape)
            scores_l2 = compute_mf_scores(rois_l2, self.templates[l2_indices])
            p_mf_l2   = scores_to_probs(scores_l2,
                                         self._mu_atom, self._sigma_atom,
                                         self._mu_bg,   self._sigma_bg)
            l2_decs = run_l2(p_mf_l2, l2_indices, threshold=cfg.l2.threshold)
            decisions[l2_indices] = l2_decs

        n_erasure = int((decisions == int(Decision.ERASURE_LOSS)).sum())

        return FrameResult(
            decisions   = decisions,
            p_post      = p_post,
            l2_routed   = l2_mask,
            n_erasure   = n_erasure,
            n_l2_routed = n_l2,
        )

    def reset(self) -> None:
        """Reset temporal state (call at start of each new atom loading cycle)."""
        self.l1_state.reset()
