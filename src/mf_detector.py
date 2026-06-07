"""
mf_detector.py  --  Vectorized Matched Filter Array core.

For K sites and an image frame, compute:
    scores[k] = dot(roi_k.flat, h_k.flat)          (raw MF score)
    p_mf[k]   = sigmoid((scores[k] - bias) / scale) (calibrated probability)

The MF is the SNR-optimal linear detector for a known signal in additive
white Gaussian noise (Neyman-Pearson lemma).
"""

import numpy as np
from numpy.typing import NDArray


def extract_rois(frame: NDArray[np.float32],
                 site_centers: NDArray[np.int32],
                 roi_size: int,
                 roi_offsets: NDArray[np.float32] | None = None
                 ) -> NDArray[np.float32]:
    """
    Extract K ROI patches from a full fluorescence frame.

    Parameters
    ----------
    frame        : (H, W) float32 image (photoelectron counts)
    site_centers : (K, 2) int32 array of [row, col] site centers
    roi_size     : patch side length (pixels), must be odd
    roi_offsets  : (K, 2) float32 [dy, dx] drift correction (sub-pixel, rounded)

    Returns
    -------
    rois : (K, roi_size, roi_size) float32
    """
    K = site_centers.shape[0]
    half = roi_size // 2
    H, W = frame.shape
    rois = np.zeros((K, roi_size, roi_size), dtype=np.float32)

    for k in range(K):
        r0, c0 = int(site_centers[k, 0]), int(site_centers[k, 1])
        if roi_offsets is not None:
            r0 += int(round(roi_offsets[k, 0]))
            c0 += int(round(roi_offsets[k, 1]))
        r_lo = max(r0 - half, 0)
        r_hi = min(r0 + half + 1, H)
        c_lo = max(c0 - half, 0)
        c_hi = min(c0 + half + 1, W)
        # copy into output patch (handles boundary sites)
        dr = r_lo - (r0 - half)
        dc = c_lo - (c0 - half)
        patch_h = r_hi - r_lo
        patch_w = c_hi - c_lo
        rois[k, dr:dr+patch_h, dc:dc+patch_w] = frame[r_lo:r_hi, c_lo:c_hi]
    return rois


def compute_mf_scores(rois: NDArray[np.float32],
                      templates: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Vectorized MF score computation.

    Parameters
    ----------
    rois      : (K, R, R) float32 -- extracted ROI patches
    templates : (K, R, R) float32 -- unit-norm Gaussian templates

    Returns
    -------
    scores : (K,) float32  -- raw matched-filter output per site
    """
    # scores[k] = sum_{i,j} rois[k,i,j] * templates[k,i,j]
    # Vectorized as batch inner product: reshape to (K, R*R), then row-wise dot
    K = rois.shape[0]
    R2 = rois.shape[1] * rois.shape[2]
    scores = (rois.reshape(K, R2) * templates.reshape(K, R2)).sum(axis=1)
    return scores


def scores_to_probs(scores: NDArray[np.float32],
                    mu_atom: float, sigma_atom: float,
                    mu_bg: float,   sigma_bg: float) -> NDArray[np.float32]:
    """
    Convert raw MF scores to posterior probabilities via Bayesian LRT.

    Assumes equal prior P(H1) = P(H0) = 0.5 and Gaussian score distributions:
        score | H1 ~ N(mu_atom, sigma_atom^2)
        score | H0 ~ N(mu_bg,   sigma_bg^2)

    LLR = log p(s|H1) - log p(s|H0)
    p_post = sigmoid(LLR)

    Parameters
    ----------
    scores     : (K,) float32
    mu_atom, sigma_atom : H1 Gaussian parameters
    mu_bg,   sigma_bg   : H0 Gaussian parameters

    Returns
    -------
    probs : (K,) float32 in (0, 1)
    """
    # log p(s|H1) - log p(s|H0)
    llr = (
        -0.5 * ((scores - mu_atom) / sigma_atom) ** 2
        + 0.5 * ((scores - mu_bg)  / sigma_bg)  ** 2
        + np.log(sigma_bg / sigma_atom)
    )
    llr_clipped = np.clip(llr, -20.0, 20.0)
    probs = (1.0 / (1.0 + np.exp(-llr_clipped))).astype(np.float32)
    return probs.astype(np.float32)


def calibrate_mf_params(rois_atom: NDArray[np.float32],
                        rois_bg:   NDArray[np.float32],
                        templates: NDArray[np.float32]
                        ) -> tuple[float, float, float, float]:
    """
    Estimate MF score distribution parameters from calibration ROI sets.

    Parameters
    ----------
    rois_atom : (N, R, R) -- ROIs confirmed to contain atoms
    rois_bg   : (N, R, R) -- ROIs confirmed to be empty
    templates : (K, R, R) -- templates (use mean template if K>1 for simplicity)

    Returns
    -------
    (mu_atom, sigma_atom, mu_bg, sigma_bg)
    """
    # Use a single representative template (mean, re-normalized)
    tmpl = templates.mean(axis=0)
    norm = np.linalg.norm(tmpl)
    if norm > 0:
        tmpl = tmpl / norm
    tmpl_rep = tmpl[np.newaxis, :, :]   # (1, R, R)

    # broadcast templates
    N_atom = rois_atom.shape[0]
    N_bg   = rois_bg.shape[0]
    t_atom = np.broadcast_to(tmpl_rep, (N_atom,) + tmpl.shape)
    t_bg   = np.broadcast_to(tmpl_rep, (N_bg,)   + tmpl.shape)

    s_atom = compute_mf_scores(rois_atom, t_atom)
    s_bg   = compute_mf_scores(rois_bg,   t_bg)

    return (float(s_atom.mean()), float(s_atom.std()) + 1e-6,
            float(s_bg.mean()),   float(s_bg.std())   + 1e-6)
