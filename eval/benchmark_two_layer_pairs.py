"""
Two-layer MF benchmark with strict 2x exposure pairing.

Design rules
------------
- Each exposure has its own MF model, calibrated independently on
  the training split for that exposure.
- L1 decides immediately when confidence = max(p, 1-p) >= CONF_THR (default 0.9).
- Otherwise the site is routed to L2, which uses the 2x exposure model.
- Available strict-2x pairs from hf_dataset (8/10/20/40/80/160 ms):
    10 ms -> 20 ms
    20 ms -> 40 ms
    40 ms -> 80 ms
    80 ms -> 160 ms
  8 ms -> 16 ms is NOT in the dataset; 20 ms (2.5x) is used as a proxy.

Reported metrics per L1 exposure
---------------------------------
  L1 Fidelity  -- single-frame MF at 0.5 threshold (L1 capability alone)
  L1 FAR / MDR
  Route %      -- fraction of valid sites routed to L2
  2L Fidelity  -- combined two-layer decision
  2L FAR / MDR
  Fid Gain     -- Fidelity improvement from L1 alone to two-layer

Usage
-----
    python eval/benchmark_two_layer_pairs.py \\
        --data-dir ./hf_dataset \\
        --out-dir  ./results \\
        [--roi-size 5] [--sigma 1.274] [--conf-thr 0.9]
"""

import argparse, json, pathlib, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.psf import make_gaussian_template
from src.mf_detector import compute_mf_scores, scores_to_probs

EXPS = ["8ms", "10ms", "20ms", "40ms", "80ms", "160ms"]

# Strict 2x pairs available in hf_dataset; 8ms uses 20ms proxy (2.5x).
PAIRS = [
    ("8ms",  "20ms",  "proxy: 16ms not in dataset"),
    ("10ms", "20ms",  ""),
    ("20ms", "40ms",  ""),
    ("40ms", "80ms",  ""),
    ("80ms", "160ms", ""),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def roi_batch(img: np.ndarray, centers: np.ndarray, R: int):
    h = R // 2; H, W = img.shape; K = len(centers)
    out  = np.zeros((K, R, R), dtype=np.float32)
    vld  = np.zeros(K, dtype=bool)
    for k, (r, c) in enumerate(centers):
        ri, ci = int(round(r)), int(round(c))
        if ri - h >= 0 and ri + h + 1 <= H and ci - h >= 0 and ci + h + 1 <= W:
            out[k] = img[ri - h: ri + h + 1, ci - h: ci + h + 1]
            vld[k] = True
    return out, vld


def calibrate(data_dir: pathlib.Path, exp: str,
              labels_tr: np.ndarray, centers: np.ndarray,
              tmpl_K: np.ndarray, R: int, bg: float,
              n_max: int = 5000, rng_seed: int = 42) -> dict:
    """Calibrate MF score distribution for one exposure from training data."""
    rng  = np.random.default_rng(rng_seed)
    sa, sb = [], []
    with np.load(data_dir / "train.npz") as d:
        imgs = d[exp]
        for fr in range(len(imgs)):
            rois, vld = roi_batch(imgs[fr], centers, R)
            rois -= bg
            sc = compute_mf_scores(rois, tmpl_K)
            v  = vld & (labels_tr[fr] != -1)
            sa.append(sc[v & (labels_tr[fr] == 1)])
            sb.append(sc[v & (labels_tr[fr] == 0)])
    sa = np.concatenate(sa);  sb = np.concatenate(sb)
    if len(sa) > n_max: sa = rng.choice(sa, n_max, replace=False)
    if len(sb) > n_max: sb = rng.choice(sb, n_max, replace=False)
    mu_a = float(sa.mean()); sg_a = max(float(sa.std()), 1e-6)
    mu_b = float(sb.mean()); sg_b = max(float(sb.std()), 1e-6)
    snr  = (mu_a - mu_b) / ((sg_a + sg_b) / 2)
    return dict(mu_atom=round(mu_a, 4), sg_atom=round(sg_a, 4),
                mu_bg=round(mu_b, 4),   sg_bg=round(sg_b, 4),
                snr=round(snr, 4))


def eval_pair(data_dir: pathlib.Path,
              l1_exp: str, l2_exp: str,
              labels_te: np.ndarray, centers: np.ndarray,
              tmpl_K: np.ndarray, R: int, bg: float,
              cal1: dict, cal2: dict, conf_thr: float) -> dict:
    """Evaluate one L1/L2 pair on the test set."""
    tp1=fp1=tn1=fn1 = 0
    tp2=fp2=tn2=fn2 = 0
    n_route = 0; n_tot = 0

    with np.load(data_dir / "test.npz") as td:
        imgs_l1 = td[l1_exp].copy()
        imgs_l2 = td[l2_exp].copy()

    t0 = time.perf_counter()
    for fr in range(len(imgs_l1)):
        rois1, vld = roi_batch(imgs_l1[fr], centers, R)
        rois1 -= bg
        sc1  = compute_mf_scores(rois1, tmpl_K)
        p1   = scores_to_probs(sc1, cal1["mu_atom"], cal1["sg_atom"],
                                cal1["mu_bg"],   cal1["sg_bg"])
        v    = vld & (labels_te[fr] != -1)
        true = labels_te[fr]

        # L1 solo (0.5 threshold)
        pr1 = (p1[v] >= 0.5).astype(int); tr = true[v]
        tp1 += int(((pr1==1)&(tr==1)).sum())
        fp1 += int(((pr1==1)&(tr==0)).sum())
        tn1 += int(((pr1==0)&(tr==0)).sum())
        fn1 += int(((pr1==0)&(tr==1)).sum())

        # Routing: confidence = max(p, 1-p)
        conf      = np.maximum(p1, 1.0 - p1)
        certain   = v & (conf >= conf_thr)
        uncertain = v & (conf <  conf_thr)
        n_route  += int(uncertain.sum())
        n_tot    += int(v.sum())

        # L2 for uncertain sites
        rois2, _ = roi_batch(imgs_l2[fr], centers, R)
        rois2 -= bg
        sc2  = compute_mf_scores(rois2, tmpl_K)
        p2   = scores_to_probs(sc2, cal2["mu_atom"], cal2["sg_atom"],
                                cal2["mu_bg"],   cal2["sg_bg"])

        pred = np.full(len(centers), -1, dtype=np.int8)
        pred[certain]   = (p1[certain]   >= 0.5).astype(np.int8)
        pred[uncertain] = (p2[uncertain] >= 0.5).astype(np.int8)

        pr2 = pred[v]
        tp2 += int(((pr2==1)&(tr==1)).sum())
        fp2 += int(((pr2==1)&(tr==0)).sum())
        tn2 += int(((pr2==0)&(tr==0)).sum())
        fn2 += int(((pr2==0)&(tr==1)).sum())

    elapsed = time.perf_counter() - t0

    far1 = fp1/(fp1+tn1) if fp1+tn1>0 else 0.
    mdr1 = fn1/(fn1+tp1) if fn1+tp1>0 else 0.
    fid1 = 1. - (far1 + mdr1) / 2.

    far2 = fp2/(fp2+tn2) if fp2+tn2>0 else 0.
    mdr2 = fn2/(fn2+tp2) if fn2+tp2>0 else 0.
    fid2 = 1. - (far2 + mdr2) / 2.

    route_pct = n_route / n_tot * 100 if n_tot > 0 else 0.

    return dict(
        l1_single = dict(Fidelity=round(fid1,6), FAR=round(far1,6), MDR=round(mdr1,6),
                         TP=tp1, FP=fp1, TN=tn1, FN=fn1),
        two_layer = dict(Fidelity=round(fid2,6), FAR=round(far2,6), MDR=round(mdr2,6),
                         TP=tp2, FP=fp2, TN=tn2, FN=fn2),
        l2_routing_pct   = round(route_pct, 4),
        fidelity_gain_pct= round((fid2 - fid1) * 100, 6),
        fps              = round(len(imgs_l1) / elapsed, 1),
        n_total_decisions= n_tot,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",  default="./hf_dataset")
    ap.add_argument("--out-dir",   default="./results")
    ap.add_argument("--roi-size",  type=int,   default=5)
    ap.add_argument("--sigma",     type=float, default=1.274,
                    help="PSF sigma in px (default 1.274)")
    ap.add_argument("--conf-thr",  type=float, default=0.9,
                    help="Route to L2 when L1 confidence < this (default 0.9)")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir  = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    R        = args.roi_size | 1  # force odd
    sigma    = args.sigma
    conf_thr = args.conf_thr

    # Load fixed arrays
    print("Loading metadata …")
    with np.load(data_dir / "train.npz") as d:
        labels_tr = d["labels"].copy()
        centers   = d["centers"].copy()
    with np.load(data_dir / "test.npz") as d:
        labels_te = d["labels"].copy()

    bg     = float(np.percentile(
        np.load(data_dir / "train.npz")["8ms"][:10], 2))
    tmpl_K = np.tile(make_gaussian_template(R, sigma)[np.newaxis],
                     (len(centers), 1, 1))

    print(f"sites={len(centers)}  ROI={R}x{R}  sigma={sigma}px  "
          f"BG={bg:.0f} ADU  conf_thr={conf_thr}\n")

    # Calibrate all exposures
    print("Calibrating per-exposure MF models …")
    calib = {}
    for e in EXPS:
        c = calibrate(data_dir, e, labels_tr, centers, tmpl_K, R, bg)
        calib[e] = c
        print(f"  {e:5s}  SNR={c['snr']:.2f}  "
              f"mu_atom={c['mu_atom']:.1f}  mu_bg={c['mu_bg']:.1f}")

    # Evaluate pairs
    print(f"\n{'L1':>5} {'L2':>6}  | {'L1 Fid':>9} {'L1 FAR':>9} {'L1 MDR':>9} "
          f"{'Route%':>8} | {'2L Fid':>9} {'2L FAR':>8} {'2L MDR':>8} {'Gain':>7}")
    print("─" * 92)

    pair_results = []
    for l1_exp, l2_exp, note in PAIRS:
        r = eval_pair(data_dir, l1_exp, l2_exp, labels_te, centers,
                      tmpl_K, R, bg, calib[l1_exp], calib[l2_exp], conf_thr)
        f1 = r["l1_single"]; f2 = r["two_layer"]
        print(f"{l1_exp:>5} {l2_exp:>6}  | "
              f"{f1['Fidelity']*100:9.4f}% {f1['FAR']*100:9.4f}% {f1['MDR']*100:9.4f}% "
              f"{r['l2_routing_pct']:8.2f}% | "
              f"{f2['Fidelity']*100:9.4f}% {f2['FAR']*100:8.4f}% {f2['MDR']*100:8.4f}% "
              f"{r['fidelity_gain_pct']:+7.4f}%"
              + (f"  ({note})" if note else ""))
        pair_results.append(dict(l1_exp=l1_exp, l2_exp=l2_exp,
                                 mf_snr_l1=calib[l1_exp]["snr"],
                                 mf_snr_l2=calib[l2_exp]["snr"],
                                 note=note, **r))

    # Save JSON
    output = dict(
        dataset          = "hf_dataset (real Yb-171 fluorescence, sCMOS Qbit-4610)",
        n_test_frames    = 100,
        n_sites          = int(len(centers)),
        roi_size         = int(R),
        psf_sigma_px     = sigma,
        background_adu   = round(bg, 1),
        conf_threshold   = conf_thr,
        description      = (
            "Two-layer MF with 2x exposure pairing. "
            "Each model calibrated on its own exposure training data. "
            f"Sites routed to L2 when L1 confidence < {conf_thr}."
        ),
        per_exp_snr      = {e: calib[e]["snr"] for e in EXPS},
        pairs            = pair_results,
    )
    out_path = out_dir / "two_layer_pairs_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
