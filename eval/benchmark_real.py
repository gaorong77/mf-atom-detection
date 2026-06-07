"""
Real-data benchmark for the MF-Array two-layer atom detection system.

Dataset: hf_dataset (real Yb-171 fluorescence, sCMOS Qbit-4610)
  - 1024 sites per frame
  - 6 exposure times: 8 / 10 / 20 / 40 / 80 / 160 ms
  - labels: {-1=invalid, 0=empty, 1=loaded}
  - centers: fixed (1024, 2) float32 in image coordinates

Usage:
    python eval/benchmark_real.py \
        --data-dir ./hf_dataset \
        --out-dir  ./results \
        [--roi-size 5] [--sigma auto] [--calib-exp 160ms]
"""

import argparse, json, pathlib, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.psf import make_gaussian_template
from src.mf_detector import compute_mf_scores, scores_to_probs

EXPS = ["8ms", "10ms", "20ms", "40ms", "80ms", "160ms"]


# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------

def extract_rois(img: np.ndarray, centers: np.ndarray, R: int):
    """Return (K, R, R) ROI array and boolean validity mask."""
    h = R // 2
    H, W = img.shape
    K = len(centers)
    rois  = np.zeros((K, R, R), dtype=np.float32)
    valid = np.zeros(K, dtype=bool)
    for k, (r, c) in enumerate(centers):
        ri, ci = int(round(r)), int(round(c))
        if ri - h >= 0 and ri + h + 1 <= H and ci - h >= 0 and ci + h + 1 <= W:
            rois[k]  = img[ri - h: ri + h + 1, ci - h: ci + h + 1]
            valid[k] = True
    return rois, valid


# ---------------------------------------------------------------------------
# PSF sigma estimation
# ---------------------------------------------------------------------------

def estimate_sigma(imgs_160: np.ndarray, labels: np.ndarray,
                   centers: np.ndarray, R: int, bg: float,
                   n_max: int = 2000, rng_seed: int = 42) -> float:
    """Estimate PSF sigma (px) from bright atom ROIs in 160 ms frames."""
    rng      = np.random.default_rng(rng_seed)
    h        = R // 2
    atom_idx = np.argwhere(labels == 1)
    sel      = atom_idx[rng.choice(len(atom_idx), min(n_max, len(atom_idx)), replace=False)]
    stack = []
    for fr, k in sel:
        if fr >= len(imgs_160):
            continue
        rois, vld = extract_rois(imgs_160[fr], centers[[k]], R)
        if vld[0]:
            p = rois[0] - bg
            if p.max() > 5:
                stack.append(p / p.max())
    if not stack:
        return 2.0
    mean_roi = np.mean(stack, axis=0)
    profile  = mean_roi[h, :]
    above    = np.where(profile > profile.max() / 2)[0]
    fwhm     = float(above[-1] - above[0] + 1) if len(above) > 1 else R // 2
    return fwhm / 2.355


# ---------------------------------------------------------------------------
# Per-exposure MF calibration
# ---------------------------------------------------------------------------

def calibrate(imgs: np.ndarray, labels: np.ndarray,
              centers: np.ndarray, tmpl_K: np.ndarray,
              R: int, bg: float, n_max: int = 5000,
              rng_seed: int = 42) -> dict:
    """
    Estimate MF score distributions for atom / background from training data.
    Returns dict with mu_atom, sg_atom, mu_bg, sg_bg, snr.
    """
    rng  = np.random.default_rng(rng_seed)
    s_a, s_b = [], []
    for fr in range(len(imgs)):
        rois, vld = extract_rois(imgs[fr], centers, R)
        rois -= bg
        sc = compute_mf_scores(rois, tmpl_K)
        v  = vld & (labels[fr] != -1)
        s_a.append(sc[v & (labels[fr] == 1)])
        s_b.append(sc[v & (labels[fr] == 0)])
    sa = np.concatenate(s_a)
    sb = np.concatenate(s_b)
    if len(sa) > n_max:
        sa = rng.choice(sa, n_max, replace=False)
    if len(sb) > n_max:
        sb = rng.choice(sb, n_max, replace=False)
    mu_a = float(sa.mean()); sg_a = max(float(sa.std()), 1e-6)
    mu_b = float(sb.mean()); sg_b = max(float(sb.std()), 1e-6)
    snr  = (mu_a - mu_b) / ((sg_a + sg_b) / 2)
    return dict(mu_atom=round(mu_a, 4), sg_atom=round(sg_a, 4),
                mu_bg=round(mu_b, 4),   sg_bg=round(sg_b, 4),
                snr=round(snr, 4))


# ---------------------------------------------------------------------------
# Single-frame evaluation
# ---------------------------------------------------------------------------

def eval_single(imgs_te: np.ndarray, labels_te: np.ndarray,
                centers: np.ndarray, tmpl_K: np.ndarray,
                R: int, bg: float, cal: dict, thresh: float = 0.5) -> dict:
    """Run single-frame MF detection; return TP/FP/TN/FN/FAR/MDR/Fidelity/fps."""
    tp = fp = tn = fn = 0
    t0 = time.perf_counter()
    for fr in range(len(imgs_te)):
        rois, vld = extract_rois(imgs_te[fr], centers, R)
        rois -= bg
        sc   = compute_mf_scores(rois, tmpl_K)
        prob = scores_to_probs(sc, cal["mu_atom"], cal["sg_atom"],
                               cal["mu_bg"],   cal["sg_bg"])
        pred = (prob >= thresh).astype(int)
        v    = vld & (labels_te[fr] != -1)
        tr   = labels_te[fr][v]; pr = pred[v]
        tp += int(((pr == 1) & (tr == 1)).sum())
        fp += int(((pr == 1) & (tr == 0)).sum())
        tn += int(((pr == 0) & (tr == 0)).sum())
        fn += int(((pr == 0) & (tr == 1)).sum())
    fps = len(imgs_te) / (time.perf_counter() - t0)
    far = fp / (fp + tn) if fp + tn > 0 else 0.0
    mdr = fn / (fn + tp) if fn + tp > 0 else 0.0
    fid = 1.0 - (far + mdr) / 2.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                FAR=round(far, 6), MDR=round(mdr, 6),
                Fidelity=round(fid, 6), fps=round(fps, 1))


# ---------------------------------------------------------------------------
# Two-layer evaluation
# ---------------------------------------------------------------------------

def eval_two_layer(imgs_l1: np.ndarray, imgs_l2: np.ndarray,
                   labels_te: np.ndarray, centers: np.ndarray,
                   tmpl_K: np.ndarray, R: int, bg: float,
                   cal_l1: dict, cal_l2: dict,
                   theta_H: float = 0.80, theta_L: float = 0.20) -> dict:
    """
    Two-layer MF routing:
      - p >= theta_H  → accept (atom)
      - p <= theta_L  → reject (empty)
      - else          → route to L2
    """
    tp = fp = tn = fn = n_l2 = n_tot = 0
    t0 = time.perf_counter()
    for fr in range(len(imgs_l1)):
        rois1, vld = extract_rois(imgs_l1[fr], centers, R)
        rois1 -= bg
        sc1  = compute_mf_scores(rois1, tmpl_K)
        p1   = scores_to_probs(sc1, cal_l1["mu_atom"], cal_l1["sg_atom"],
                                cal_l1["mu_bg"],   cal_l1["sg_bg"])
        pred = np.full(len(centers), -1, dtype=np.int8)
        pred[(p1 >= theta_H) & vld] = 1
        pred[(p1 <= theta_L) & vld] = 0
        uncertain = (pred == -1) & vld
        if uncertain.sum() > 0:
            rois2, _ = extract_rois(imgs_l2[fr], centers, R)
            rois2 -= bg
            sc2  = compute_mf_scores(rois2, tmpl_K)
            p2   = scores_to_probs(sc2, cal_l2["mu_atom"], cal_l2["sg_atom"],
                                    cal_l2["mu_bg"],   cal_l2["sg_bg"])
            pred[uncertain] = (p2[uncertain] >= 0.5).astype(np.int8)
            n_l2 += int(uncertain.sum())
        v  = vld & (labels_te[fr] != -1)
        tr = labels_te[fr][v]; pr = pred[v]
        tp += int(((pr == 1) & (tr == 1)).sum())
        fp += int(((pr == 1) & (tr == 0)).sum())
        tn += int(((pr == 0) & (tr == 0)).sum())
        fn += int(((pr == 0) & (tr == 1)).sum())
        n_tot += int(v.sum())
    fps    = len(imgs_l1) / (time.perf_counter() - t0)
    far    = fp / (fp + tn) if fp + tn > 0 else 0.0
    mdr    = fn / (fn + tp) if fn + tp > 0 else 0.0
    fid    = 1.0 - (far + mdr) / 2.0
    l2_pct = n_l2 / n_tot * 100 if n_tot > 0 else 0.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                FAR=round(far, 6), MDR=round(mdr, 6),
                Fidelity=round(fid, 6),
                L2_routing_pct=round(l2_pct, 4),
                fps=round(fps, 1),
                theta_H=theta_H, theta_L=theta_L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-data MF benchmark")
    parser.add_argument("--data-dir",  default="./hf_dataset",
                        help="Path to hf_dataset directory")
    parser.add_argument("--out-dir",   default="./results",
                        help="Output directory for benchmark JSON")
    parser.add_argument("--roi-size",  type=int, default=5,
                        help="ROI window size (odd integer, default 5)")
    parser.add_argument("--sigma",     type=float, default=0.0,
                        help="PSF sigma in px; 0 = auto-estimate (default)")
    parser.add_argument("--calib-exp", default="160ms",
                        choices=EXPS,
                        help="Exposure used for PSF sigma estimation (default 160ms)")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir  = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    R = args.roi_size
    if R % 2 == 0:
        R += 1
        print(f"[warn] ROI size forced to odd: {R}")

    print("Loading data …")
    train_d   = np.load(data_dir / "train.npz")
    test_d    = np.load(data_dir / "test.npz")
    labels_tr = train_d["labels"]   # (M, K)
    labels_te = test_d["labels"]    # (N, K)
    centers   = train_d["centers"]  # (K, 2)

    # Background floor (2nd percentile of 8 ms train images)
    bg = float(np.percentile(train_d["8ms"], 2))
    print(f"  sites={len(centers)}  train_frames={len(labels_tr)}"
          f"  test_frames={len(labels_te)}  ROI={R}x{R}  BG={bg:.1f} ADU")

    # PSF sigma
    if args.sigma > 0:
        sigma = args.sigma
        print(f"  PSF sigma (user-specified): {sigma:.3f} px")
    else:
        imgs_calib = train_d[args.calib_exp]
        sigma = estimate_sigma(imgs_calib, labels_tr, centers, R, bg)
        print(f"  PSF sigma (auto, {args.calib_exp}): {sigma:.3f} px")

    tmpl   = make_gaussian_template(R, sigma)
    tmpl_K = np.tile(tmpl[np.newaxis], (len(centers), 1, 1))  # (K, R, R)

    # Per-exposure calibration
    print("\nCalibrating MF per exposure …")
    calib = {}
    for exp in EXPS:
        c = calibrate(train_d[exp], labels_tr, centers, tmpl_K, R, bg)
        calib[exp] = c
        print(f"  {exp:5s}  mu_atom={c['mu_atom']:8.2f}  mu_bg={c['mu_bg']:7.2f}"
              f"  SNR={c['snr']:.2f}")

    # Single-frame evaluation
    print("\nSingle-frame evaluation (test set) …")
    single_results = {}
    for exp in EXPS:
        r = eval_single(test_d[exp], labels_te, centers, tmpl_K, R, bg, calib[exp])
        single_results[exp] = {**r, "exp": exp}
        print(f"  {exp:5s}  Fidelity={r['Fidelity']*100:.4f}%"
              f"  FAR={r['FAR']*100:.4f}%  MDR={r['MDR']*100:.4f}%")

    # Two-layer evaluation: config A (L1=8ms, L2=40ms)
    print("\nTwo-layer A: L1=8ms / L2=40ms …")
    tl_a = eval_two_layer(
        test_d["8ms"], test_d["40ms"], labels_te, centers, tmpl_K, R, bg,
        calib["8ms"], calib["40ms"], theta_H=0.80, theta_L=0.20)
    print(f"  Fidelity={tl_a['Fidelity']*100:.4f}%  L2_route={tl_a['L2_routing_pct']:.2f}%")

    # Two-layer evaluation: config B (L1=40ms, L2=160ms)  — recommended
    print("Two-layer B: L1=40ms / L2=160ms …")
    tl_b = eval_two_layer(
        test_d["40ms"], test_d["160ms"], labels_te, centers, tmpl_K, R, bg,
        calib["40ms"], calib["160ms"], theta_H=0.90, theta_L=0.10)
    print(f"  Fidelity={tl_b['Fidelity']*100:.4f}%  L2_route={tl_b['L2_routing_pct']:.2f}%")

    # Save
    output = dict(
        dataset              = "hf_dataset (real Yb-171 fluorescence, sCMOS Qbit-4610)",
        n_train_frames       = int(len(labels_tr)),
        n_test_frames        = int(len(labels_te)),
        n_sites              = int(len(centers)),
        roi_size             = int(R),
        psf_sigma_px         = round(sigma, 4),
        background_floor_adu = round(bg, 1),
        per_exp_calibration  = calib,
        single_frame_mf      = single_results,
        two_layer_A_8ms_40ms   = tl_a,
        two_layer_B_40ms_160ms = tl_b,
    )
    out_path = out_dir / "real_data_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return output


if __name__ == "__main__":
    main()
