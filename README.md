# mf-atom-detection

**PA-3HDA-MF v1.0** — Matched Filter Array two-layer atom detection system for neutral atom quantum computing.

Replaces the L1 / L2 lightweight CNNs from PA-3HDA v10.0 with a **Matched Filter (MF) Array**.
L1.5 physical-diagnostic router is removed. Architecture reduced to two active detection layers.

---

## Architecture

```
Frame (short exp. 5.09 ms)
  └─► L1: MF-Array + Bayesian temporal fusion + CUSUM
        ├─► ATOM_PRESENT   (p_post > θ_H = 0.92)
        ├─► ATOM_ABSENT    (p_post < θ_L = 0.08)
        ├─► ERASURE_LOSS   (CUSUM threshold crossed)
        └─► ROUTE_L2

Frame (long  exp. 18.0 ms)
  └─► L2: MF-Array precise judgment
        ├─► ATOM_PRESENT
        └─► ATOM_ABSENT
```

### Why Matched Filter?

The Matched Filter is the SNR-optimal linear detector for a known signal in additive
white Gaussian noise (Neyman-Pearson lemma). For a Gaussian PSF atom signal:

```
score_k  = dot(roi_k.flat, h_k.flat)   # h_k: unit-norm Gaussian template
LLR(s_k) = -0.5*((s-mu_atom)/sigma_atom)^2 + 0.5*((s-mu_bg)/sigma_bg)^2 + log(sigma_bg/sigma_atom)
p_mf(k)  = sigmoid(LLR)               # calibrated posterior probability
```

Advantages over CNN:
- **Zero learned parameters** — no training required, no overfitting
- **Theoretically optimal** for the physical Poisson+Gaussian noise model
- **Pure dot-product** — faster inference, FPGA-friendly (DSP blocks)
- **Analytical performance bound** — SNR fully determines FAR/MDR at calibration time

### Bayesian Temporal Fusion (retained from PA-3HDA)

```
x(k,t) = x(k,t-1) + log(p_s/(1-p_s)) + λ · log(p_mf/(1-p_mf))
p_post  = sigmoid(x(k,t))
```

- p_s = 0.9952 (atom survival probability per frame)
- λ = observation weight (default 0.8)
- Warmup: 5 frames (single-frame fallback)

### CUSUM Erasure Detection (retained, with gating fix)

```
S(k,t) = max(0, S(k,t-1) + log((1-p_mf)/p_mf)) · gate(prev_log_odds > 0)
```

Key fix vs PA-3HDA: CUSUM only accumulates when **previous frame** believed atom present
(`prev_log_odds > 0`). Prevents false ERASURE for perpetually empty sites.

---

## Benchmark Results

### Simulation (synthetic Yb-171, 20×20 array, 500 frames)

Config: 20×20 array (400 sites), 500 frames, simulated Yb-171 platform.

| Metric | Value |
|--------|-------|
| **Fidelity** | **100.000000 %** |
| FAR (False Alarm Rate) | 0.00e+00 (0.000 ppm) |
| MDR (Miss Detection Rate) | 0.00e+00 (0.000 ppm) |
| TP / FP / TN / FN | 46,955 / 0 / 153,045 / 0 |
| L2 routing rate | 0.0000 % |
| Erasure rate | 0.2250 % |
| MF SNR | 84.8 |
| Total decisions | 200,000 |
| Speed | 10.3 fps (97.4 ms/frame, pure Python) |

MF calibration: mu_atom=1100.6 ± 14.0, mu_bg=63.8 ± 10.5.

---

### Real Data (hf_dataset — Yb-171 fluorescence, sCMOS Qbit-4610)

Config: 32×32 array (1024 sites), 100 test frames, 6 exposure times.
MF calibrated independently for each exposure using 801 training frames.
PSF sigma: 1.274 px (auto-estimated from 160 ms frames).
Background floor: 784 ADU (2nd percentile).

#### Single-frame MF detection

| Exposure | MF SNR | Fidelity | FAR | MDR |
|----------|--------|----------|-----|-----|
| 8 ms | 0.71 | 63.67 % | 28.20 % | 44.46 % |
| 10 ms | 1.01 | 70.14 % | 22.63 % | 37.10 % |
| 20 ms | 1.79 | 82.96 % | 11.50 % | 22.57 % |
| 40 ms | 3.07 | 95.24 % | 3.12 % | 6.40 % |
| 80 ms | 4.64 | **99.58 %** | 0.61 % | 0.24 % |
| 160 ms | 5.84 | **99.87 %** | 0.24 % | 0.01 % |

Short-exposure performance is limited by physics (photon shot noise), not by the
MF algorithm: at 8 ms the signal is < 80 ADU above background, giving SNR < 1.

#### Two-layer MF routing

| Config | L1 exp | L2 exp | Fidelity | L2 route rate |
|--------|--------|--------|----------|---------------|
| A | 8 ms | 40 ms | 94.33 % | 97.35 % (L1 almost always uncertain) |
| **B (recommended)** | **40 ms** | **160 ms** | **98.88 %** | **16.52 %** |

Config B provides the best accuracy-speed trade-off: 83 % of sites are resolved
at L1 (40 ms) and only 16.5 % are routed to the high-SNR L2 frame.

> Full results: `results/real_data_benchmark.json`

---

## Repository Structure

```
mf_atom_detection/
├── src/
│   ├── config.py           System-wide parameters (dataclasses)
│   ├── psf.py              Gaussian PSF model, unit-norm template generation
│   ├── mf_detector.py      Vectorized MF score + LRT probability conversion
│   ├── layer1_mf.py        L1: MF-Array + Bayesian log-odds fusion + CUSUM
│   ├── layer2_mf.py        L2: MF-Array precise judgment
│   ├── simulate.py         Synthetic Yb-171 fluorescence image simulator
│   └── system.py           Two-layer system orchestration + 5-type output
├── eval/
│   ├── benchmark.py        Simulation benchmark (FAR/MDR/Fidelity/speed)
│   └── benchmark_real.py   Real-data benchmark (hf_dataset, per-exp calibration)
├── results/
│   ├── benchmark.json           Simulation evaluation results
│   └── real_data_benchmark.json Real-data evaluation results
└── README.md
```

## Quick Start

```bash
pip install numpy
python eval/benchmark.py
```

---

## Physical Parameters (default)

| Parameter | Value | Source |
|-----------|-------|--------|
| Photon rate | 35,000 ph/s/atom | ¹⁷¹Yb 399 nm imaging line |
| Camera QE | 72 % | Qbit 4610 sCMOS |
| Readout noise | 0.3 e⁻ | sCMOS spec |
| PSF sigma | 4.6 px | NA=0.5 objective |
| Site pitch | 54 px | 5 µm trap spacing |
| Background | 800 ph/s/site | |
| Crosstalk | 1 % | PSF tail leakage |
| t_L1 | 5.09 ms | ~128 photons/atom |
| t_L2 | 18.0 ms | independently calibrated |
| p_survival | 0.9952 | 0.48 %/frame loss rate |

---

## Five-Type Structured Output

| # | Output | Source | QEC Value |
|---|--------|--------|-----------|
| 1 | `ATOM_PRESENT` | L1 / L2 | — |
| 2 | `ATOM_ABSENT` | L1 / L2 | — |
| 3 | `ERASURE_LOSS` | L1 CUSUM | Erasure error → QEC threshold ~50 % |
| 4 | `CORR_LOSS` | reserved (L1.5 removed) | Correlated erasure |
| 5 | `DRIFT_ALARM` | L0 background (stub) | Triggers recalibration |

---

## Reference

PA-3HDA v10.0 patent disclosure — Hierarchical Adaptive Fluorescence Detection
System and Method for Neutral Atom Arrays in Quantum Computing (2026).
