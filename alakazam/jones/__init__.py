"""ALAKAZAM Jones Matrix subpackage.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

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

from .constructors import (
    delay_to_jones,
    crossdelay_to_jones,
    crossphase_to_jones,
    gain_to_jones,
    leakage_to_jones,
)

from .parang import (
    compute_parallactic_angles,
    parang_to_jones_linear,
    parang_to_jones_circular,
    parang_to_jones,
)
