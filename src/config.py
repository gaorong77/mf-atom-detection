"""
config.py  --  Global configuration for MF-Array two-layer atom detection system.

Architecture: PA-3HDA-MF v1.0
  L1  : Matched Filter Array  (short exposure) + Bayesian temporal fusion + CUSUM
  L2  : Matched Filter Array  (long  exposure) -- final adjudication
  Note: L1.5 physical-diagnostic router removed; no CNN layers.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ArrayConfig:
    # --- atom array geometry ---
    n_rows: int = 20          # rows of tweezer sites
    n_cols: int = 20          # cols of tweezer sites
    site_pitch_px: int = 54   # inter-site spacing in pixels
    roi_size: int = 17        # ROI side length for each site (pixels)

    @property
    def n_sites(self) -> int:
        return self.n_rows * self.n_cols


@dataclass
class PSFConfig:
    sigma_px: float = 4.6    # PSF Gaussian sigma (pixels)
    roi_size: int = 17        # must match ArrayConfig.roi_size


@dataclass
class PhysicsConfig:
    # --- Yb-171 photon statistics (399 nm imaging line) ---
    photon_rate: float = 35_000.0   # photons/s/atom at saturation
    qe: float = 0.72                # camera QE @ 399 nm
    readout_noise_e: float = 0.3    # sCMOS readout noise (electrons rms)
    bg_photon_rate: float = 800.0   # background photons/s/site
    crosstalk_fraction: float = 0.01  # PSF tail leakage to nearest neighbor

    # --- frame survival probability (171Yb, ~0.48 %/frame loss) ---
    p_survival: float = 0.9952


@dataclass
class L1Config:
    exposure_ms: float = 5.09   # short exposure time (ms)
    theta_H: float = 0.92       # upper fast-accept threshold
    theta_L: float = 0.08       # lower fast-reject threshold
    lambda_obs: float = 0.8     # Bayesian observation weight
    warmup_frames: int = 5      # frames before temporal fusion activates


@dataclass
class L2Config:
    exposure_ms: float = 18.0   # long exposure time (ms); independently calibrated
    threshold: float = 0.50     # MF score -> decision threshold (probability)


@dataclass
class CUSUMConfig:
    h_loss: float = 5.0         # CUSUM threshold for individual loss
    h_corr: float = 4.0         # CUSUM threshold for correlated loss (target site)
    h_corr_nbr: float = 3.0     # CUSUM threshold for correlated loss (neighbor)


@dataclass
class MFConfig:
    """Matched filter calibration parameters (filled by calibrate step)."""
    mu_atom: float = 0.0        # mean MF score when atom present  (calibrated)
    sigma_atom: float = 1.0     # std  MF score when atom present
    mu_bg: float = -4.0         # mean MF score when atom absent   (calibrated)
    sigma_bg: float = 1.0       # std  MF score when atom absent


@dataclass
class SystemConfig:
    array: ArrayConfig = field(default_factory=ArrayConfig)
    psf: PSFConfig = field(default_factory=PSFConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    cusum: CUSUMConfig = field(default_factory=CUSUMConfig)
    mf: MFConfig = field(default_factory=MFConfig)

    def summary(self) -> str:
        lines = [
            "=== MF-Array Two-Layer System Config ===",
            f"  Array      : {self.array.n_rows}x{self.array.n_cols} sites, "
            f"pitch={self.array.site_pitch_px}px, ROI={self.array.roi_size}px",
            f"  PSF sigma  : {self.psf.sigma_px} px",
            f"  L1 exposure: {self.l1.exposure_ms} ms, "
            f"theta=({self.l1.theta_H:.2f}, {self.l1.theta_L:.2f})",
            f"  L2 exposure: {self.l2.exposure_ms} ms",
            f"  CUSUM h_loss: {self.cusum.h_loss}",
            f"  p_survival : {self.physics.p_survival}",
        ]
        return "\n".join(lines)
