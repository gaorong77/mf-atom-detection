"""
psf.py  --  PSF model and matched-filter template generation.

Each site gets a normalized Gaussian template h_k (shape: roi_size x roi_size).
The MF score for site k is:
    score_k = dot(roi_k.flatten(), h_k.flatten()) / ||h_k||
which is the SNR-optimal statistic for detecting a known signal in white Gaussian noise.
"""

import numpy as np
from typing import Tuple
from src.config import PSFConfig, ArrayConfig


def make_gaussian_template(roi_size: int, sigma: float,
                            cx: float | None = None,
                            cy: float | None = None) -> np.ndarray:
    """
    Build a normalized Gaussian template centered at (cx, cy) within an roi_size x roi_size patch.
    Default center: exact geometric center ((roi_size-1)/2, (roi_size-1)/2).

    Returns
    -------
    h : ndarray, shape (roi_size, roi_size), float32, unit-norm (||h||_F == 1).
    """
    if cx is None:
        cx = (roi_size - 1) / 2.0
    if cy is None:
        cy = (roi_size - 1) / 2.0

    ys = np.arange(roi_size, dtype=np.float32)
    xs = np.arange(roi_size, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")   # shape (roi_size, roi_size)

    h = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
    norm = np.linalg.norm(h)
    if norm > 0:
        h = h / norm
    return h.astype(np.float32)


def build_template_array(array_cfg: ArrayConfig,
                         psf_cfg: PSFConfig,
                         roi_offsets: np.ndarray | None = None
                         ) -> np.ndarray:
    """
    Build matched-filter templates for all K = n_rows * n_cols sites.

    Parameters
    ----------
    array_cfg   : ArrayConfig
    psf_cfg     : PSFConfig
    roi_offsets : ndarray, shape (K, 2) [dy, dx] in pixels -- PSF drift correction.
                  If None, all templates are centered.

    Returns
    -------
    templates : ndarray, shape (K, roi_size, roi_size), float32, each unit-norm.
    """
    K = array_cfg.n_sites
    R = array_cfg.roi_size
    sigma = psf_cfg.sigma_px

    templates = np.empty((K, R, R), dtype=np.float32)
    center = (R - 1) / 2.0

    for k in range(K):
        if roi_offsets is not None:
            dy, dx = float(roi_offsets[k, 0]), float(roi_offsets[k, 1])
            cy = center - dy    # shift template opposite to drift
            cx = center - dx
        else:
            cy = center
            cx = center
        templates[k] = make_gaussian_template(R, sigma, cx=cx, cy=cy)

    return templates


def compute_expected_mf_score(photons_mean: float,
                               template: np.ndarray) -> float:
    """
    Theoretical mean MF score when the atom is present.

    Under H1:  E[score] = photons_mean * ||h||^2 / ||h|| = photons_mean * ||h||
    But since h is already unit-norm, ||h|| = 1, so E[score] = photons_mean * sum(h^2) = photons_mean.
    More precisely: E[s] = signal_amplitude * <h, h> = A * 1 = A
    where A is the peak pixel value of the un-normalized PSF times the photon count.

    For simulation purposes this function returns the expected normalized score.
    """
    # With unit-norm template and Poisson signal:
    # E[s] = n_photons * sum(h * h) = n_photons * ||h||^2 = n_photons (since ||h||=1)
    return float(photons_mean * np.sum(template * template))
