"""ALAKAZAM v1 Jones Matrix subpackage.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from .algebra import (
    FeedBasis, detect_feed_basis, corr_labels,
    jones_multiply, jones_inverse, jones_herm,
    unapply_rows_full, unapply_rows_full_freqdep,
    unapply_rows_diag, unapply_rows_diag_freqdep,
    unapply_jones_to_rows, is_diagonal_jones,
    compute_residual_2x2, compute_residual_diag,
    compute_residual_cross, compute_residual_diag_freq,
    compute_residual_cross_freq, compose_jones_chain,
)

from .constructors import (
    parallel_delay_to_jones, gains_to_jones,
    leakage_to_jones, cross_delay_to_jones,
    cross_phase_to_jones,
)

from .parang import (
    compute_parallactic_angles,
    parang_to_jones_linear, parang_to_jones_circular,
    parang_to_jones,
)
