"""
ALAKAZAM Jones Matrix Algebra.

Re-exports from submodules for backward compatibility:
  from alakazam.jones import delay_to_jones   # works
  from alakazam.jones.algebra import jones_unapply  # also works

Developed by Arpan Pal 2026, NRAO / NCRA
"""

# Algebra: 2×2 ops, apply/unapply, residuals, chain composition
from .algebra import (
    FeedBasis,
    detect_feed_basis,
    corr_labels,
    jones_multiply,
    jones_inverse,
    jones_herm,
    jones_apply,
    jones_unapply,
    jones_apply_freq,
    jones_unapply_freq,
    compute_residual_2x2,
    compute_residual_diag,
    compute_residual_cross,
    compute_residual_diag_freq,
    compute_residual_cross_freq,
    compose_jones_chain,
)

# Constructors: native params → Jones matrices
from .constructors import (
    delay_to_jones,
    crossdelay_to_jones,
    crossphase_to_jones,
    gain_to_jones,
    leakage_to_jones,
)

# Parallactic angle
from .parang import (
    compute_parallactic_angles,
    parang_to_jones_linear,
    parang_to_jones_circular,
    parang_to_jones,
)
