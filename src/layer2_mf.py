"""
layer2_mf.py  --  L2: Matched Filter precise-judgment layer.

Applied only to sites that L1 routes to L2 (decisions == ROUTE_L2).
Uses a longer exposure (higher photon count) for definitive judgment.
"""

import numpy as np
from numpy.typing import NDArray
from src.layer1_mf import Decision


def run_l2(p_mf_l2:   NDArray[np.float32],
           l2_indices: NDArray[np.int64],
           threshold:  float = 0.50
           ) -> NDArray[np.int8]:
    """
    Final adjudication for routed sites.

    Parameters
    ----------
    p_mf_l2    : (M,) float32 -- MF probability from long-exposure ROI
    l2_indices : (M,) int64   -- site indices that were routed to L2
    threshold  : decision threshold

    Returns
    -------
    decisions : (M,) int8 -- ATOM_PRESENT or ATOM_ABSENT
    """
    decisions = np.where(
        p_mf_l2 >= threshold,
        np.int8(Decision.ATOM_PRESENT),
        np.int8(Decision.ATOM_ABSENT)
    ).astype(np.int8)
    return decisions
