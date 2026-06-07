"""
simulate.py  --  Synthetic fluorescence image simulator.

Models Yb-171 atom array on sCMOS camera:
  - Poisson photon statistics
  - Gaussian readout noise
  - PSF Gaussian profile per site
  - Background photons
  - PSF tail crosstalk from neighbors
  - Per-frame atom survival (stochastic loss)
"""

import numpy as np
from numpy.typing import NDArray
from src.config import SystemConfig
from src.psf import make_gaussian_template


def _frame_size(cfg: SystemConfig) -> tuple[int, int]:
    pitch = cfg.array.site_pitch_px
    margin = pitch
    H = cfg.array.n_rows * pitch + 2 * margin
    W = cfg.array.n_cols * pitch + 2 * margin
    return H, W


def simulate_frame(atom_states: NDArray[np.bool_],
                   cfg: SystemConfig,
                   rng: np.random.Generator,
                   exposure_ms: float,
                   roi_offsets: NDArray[np.float32] | None = None
                   ) -> NDArray[np.float32]:
    """
    Render a single fluorescence frame.

    Parameters
    ----------
    atom_states : (K,) bool -- True = atom present at site k
    cfg         : SystemConfig
    rng         : numpy random generator
    exposure_ms : exposure time in ms
    roi_offsets : (K,2) sub-pixel drift offsets (ignored for full-frame sim)

    Returns
    -------
    frame : (H, W) float32 -- photoelectron image
    """
    H, W = _frame_size(cfg)
    frame = np.zeros((H, W), dtype=np.float32)

    pitch  = cfg.array.site_pitch_px
    margin = pitch
    sigma  = cfg.psf.sigma_px
    R      = cfg.array.roi_size
    exp_s  = exposure_ms * 1e-3

    n_photons_mean = (cfg.physics.photon_rate
                      * cfg.physics.qe
                      * exp_s)               # mean photons per atom
    n_bg_mean      = (cfg.physics.bg_photon_rate
                      * cfg.physics.qe
                      * exp_s)               # mean bg photons per site

    K = cfg.array.n_sites

    for k in range(K):
        row = k // cfg.array.n_cols
        col = k %  cfg.array.n_cols
        cr  = row * pitch + margin   # center row in full frame
        cc  = col * pitch + margin   # center col in full frame

        # Background
        bg_counts = rng.poisson(n_bg_mean, size=(R, R)).astype(np.float32)
        r0 = cr - R // 2
        c0 = cc - R // 2
        frame[r0:r0+R, c0:c0+R] += bg_counts

        if atom_states[k]:
            # Signal: PSF-distributed Poisson photons
            tmpl = make_gaussian_template(R, sigma)   # (R,R), unit norm
            # Expected signal image: photon counts ~ N * tmpl (unnormalized)
            # Un-normalize to get raw pixel expectation
            tmpl_raw = (tmpl / (tmpl.max() + 1e-12)) * n_photons_mean
            signal   = rng.poisson(tmpl_raw.clip(0)).astype(np.float32)
            frame[r0:r0+R, c0:c0+R] += signal

    # Crosstalk: add fraction of neighbor signal
    ct = cfg.physics.crosstalk_fraction
    if ct > 0:
        # Simple approximation: shift copies of the frame
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            shifted = np.roll(frame, (dr*pitch, dc*pitch), axis=(0,1))
            frame += ct * shifted

    # Readout noise (Gaussian, independent per pixel)
    read_noise = rng.normal(0.0, cfg.physics.readout_noise_e,
                             size=frame.shape).astype(np.float32)
    frame += read_noise
    return frame.clip(0)


def simulate_atom_states(K: int, p_survival: float,
                         prev_states: NDArray[np.bool_] | None,
                         rng: np.random.Generator
                         ) -> NDArray[np.bool_]:
    """
    Propagate atom states by one frame with stochastic loss.
    Initial call: prev_states=None -> random 50% loading.
    """
    if prev_states is None:
        return rng.random(K) < 0.60   # ~60% loading probability
    survive = rng.random(K) < p_survival
    return prev_states & survive


def build_site_centers(cfg: SystemConfig) -> NDArray[np.int32]:
    pitch  = cfg.array.site_pitch_px
    margin = pitch
    rows   = np.arange(cfg.array.n_rows)
    cols   = np.arange(cfg.array.n_cols)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.ravel() * pitch + margin,
                     cc.ravel() * pitch + margin], axis=1).astype(np.int32)
