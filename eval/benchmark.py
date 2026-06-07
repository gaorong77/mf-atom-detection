"""
benchmark.py  --  Evaluation harness for MF-Array two-layer system.

Metrics:
  FAR          False Alarm Rate  = FP / (FP + TN)
  MDR          Miss Detection Rate = FN / (FN + TP)
  Fidelity     = 1 - (FAR + MDR) / 2   (or exact: TP_rate * P(H1) + TN_rate * P(H0))
  L2 routing rate = sites routed to L2 / total decisions
  Erasure rate = ERASURE_LOSS triggers / total decisions
  Throughput   = frames processed per second (wall clock)
"""

import sys, time, json, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import SystemConfig, MFConfig
from src.psf import build_template_array
from src.mf_detector import (compute_mf_scores, scores_to_probs,
                               extract_rois, calibrate_mf_params)
from src.simulate import (simulate_frame, simulate_atom_states,
                           build_site_centers)
from src.layer1_mf import Decision
from src.system import MFSystem


# ─────────────────────────────────────────────────────────────────────────────
def calibrate_system(cfg: SystemConfig,
                     n_calib: int = 50_000,
                     rng: np.random.Generator | None = None
                     ) -> tuple[float, float, float, float]:
    """
    Estimate MF score distributions from simulated calibration data.
    Returns (mu_atom, sigma_atom, mu_bg, sigma_bg).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    K   = cfg.array.n_sites
    R   = cfg.array.roi_size
    templates = build_template_array(cfg.array, cfg.psf)
    tmpl = templates.mean(axis=0)
    tmpl /= (np.linalg.norm(tmpl) + 1e-12)
    tmpl_rep = tmpl[np.newaxis]

    rois_atom_list, rois_bg_list = [], []

    n_frames = n_calib // K + 1
    for _ in range(n_frames):
        # half atom, half empty for calibration
        states = np.zeros(K, dtype=bool)
        states[:K//2] = True
        rng.shuffle(states)

        frame = simulate_frame(states, cfg, rng, cfg.l1.exposure_ms)
        centers = build_site_centers(cfg)
        rois = extract_rois(frame, centers, R)

        rois_atom_list.append(rois[states])
        rois_bg_list.append(rois[~states])

    rois_atom = np.concatenate(rois_atom_list, axis=0)
    rois_bg   = np.concatenate(rois_bg_list,   axis=0)

    t_a = np.broadcast_to(tmpl_rep, (rois_atom.shape[0],) + tmpl.shape)
    t_b = np.broadcast_to(tmpl_rep, (rois_bg.shape[0],)   + tmpl.shape)
    s_a = compute_mf_scores(rois_atom, t_a)
    s_b = compute_mf_scores(rois_bg,   t_b)

    return (float(s_a.mean()), max(float(s_a.std()), 1e-6),
            float(s_b.mean()), max(float(s_b.std()), 1e-6))


# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(cfg: SystemConfig,
                  n_frames: int = 500,
                  seed: int = 0) -> dict:
    """
    Full benchmark: calibrate, run N frames, compute all metrics.
    """
    rng = np.random.default_rng(seed)
    print(f"Calibrating MF parameters on {cfg.array.n_sites * 200} samples ...")
    mu_atom, sig_atom, mu_bg, sig_bg = calibrate_system(cfg, n_calib=200*cfg.array.n_sites, rng=rng)
    print(f"  mu_atom={mu_atom:.3f}, sigma_atom={sig_atom:.3f}")
    print(f"  mu_bg  ={mu_bg:.3f},   sigma_bg  ={sig_bg:.3f}")
    print(f"  SNR    ={(mu_atom-mu_bg)/((sig_atom+sig_bg)/2):.2f}")

    system = MFSystem(cfg)
    system.set_mf_params(mu_atom, sig_atom, mu_bg, sig_bg)

    K     = cfg.array.n_sites
    # accumulators
    tp = fp = tn = fn = 0
    n_l2_total = n_erasure_total = n_decisions_total = 0
    atom_states = None

    t0 = time.perf_counter()
    for frame_idx in range(n_frames):
        atom_states = simulate_atom_states(K, cfg.physics.p_survival,
                                           atom_states, rng)
        frame_l1 = simulate_frame(atom_states, cfg, rng, cfg.l1.exposure_ms)
        frame_l2 = simulate_frame(atom_states, cfg, rng, cfg.l2.exposure_ms)

        result = system.process_frame(frame_l1, frame_l2)

        n_l2_total        += result.n_l2_routed
        n_erasure_total   += result.n_erasure
        n_decisions_total += K

        # Evaluate only ATOM_PRESENT / ATOM_ABSENT decisions (skip ERASURE for FAR/MDR)
        final_dec = result.decisions.copy()
        # Treat ERASURE as ATOM_ABSENT for this evaluation
        erasure_mask = final_dec == int(Decision.ERASURE_LOSS)
        final_dec[erasure_mask] = int(Decision.ATOM_ABSENT)

        pred_1 = final_dec == int(Decision.ATOM_PRESENT)
        true_1 = atom_states

        tp += int(( pred_1 &  true_1).sum())
        fp += int(( pred_1 & ~true_1).sum())
        tn += int((~pred_1 & ~true_1).sum())
        fn += int((~pred_1 &  true_1).sum())

    elapsed = time.perf_counter() - t0

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    mdr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fidelity = 1.0 - (far + mdr) / 2.0

    total_positive = tp + fn
    total_negative = fp + tn

    results = {
        "n_frames"       : n_frames,
        "n_sites"        : K,
        "total_decisions": n_decisions_total,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "FAR"            : far,
        "MDR"            : mdr,
        "Fidelity"       : fidelity,
        "L2_routing_rate": n_l2_total  / n_decisions_total,
        "erasure_rate"   : n_erasure_total / n_decisions_total,
        "fps"            : n_frames / elapsed,
        "ms_per_frame"   : elapsed / n_frames * 1000,
        "mf_mu_atom"     : mu_atom,
        "mf_sig_atom"    : sig_atom,
        "mf_mu_bg"       : mu_bg,
        "mf_sig_bg"      : sig_bg,
        "mf_snr"         : (mu_atom - mu_bg) / ((sig_atom + sig_bg) / 2),
        "l1_theta_H"     : cfg.l1.theta_H,
        "l1_theta_L"     : cfg.l1.theta_L,
        "l1_exposure_ms" : cfg.l1.exposure_ms,
        "l2_exposure_ms" : cfg.l2.exposure_ms,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
def print_report(r: dict) -> None:
    print("\n" + "="*52)
    print("  MF-Array Two-Layer System — Benchmark Report")
    print("="*52)
    print(f"  Frames : {r['n_frames']:,}   Sites : {r['n_sites']}   "
          f"Total decisions : {r['total_decisions']:,}")
    print(f"  TP={r['TP']:,}  FP={r['FP']:,}  TN={r['TN']:,}  FN={r['FN']:,}")
    print(f"  FAR          : {r['FAR']:.6f}  ({r['FAR']*100:.4f} %)")
    print(f"  MDR          : {r['MDR']:.6f}  ({r['MDR']*100:.4f} %)")
    print(f"  Fidelity     : {r['Fidelity']:.6f}  ({r['Fidelity']*100:.4f} %)")
    print(f"  L2 routing   : {r['L2_routing_rate']*100:.3f} %")
    print(f"  Erasure rate : {r['erasure_rate']*100:.4f} %")
    print(f"  MF SNR       : {r['mf_snr']:.2f}")
    print(f"  Speed        : {r['fps']:.1f} frame/s  "
          f"({r['ms_per_frame']:.1f} ms/frame)")
    print("="*52)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = SystemConfig()
    results = run_benchmark(cfg, n_frames=500, seed=42)
    print_report(results)

    out_path = pathlib.Path(__file__).parent.parent / "results" / "benchmark.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
