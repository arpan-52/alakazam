"""ALAKAZAM v1 Jones Matrix subpackage.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from .algebra import (
    FeedBasis, detect_feed_basis,
    jones_multiply,
    unapply_rows_full_freqdep, unapply_rows_diag_freqdep,
    unapply_jones_to_rows, is_diagonal_jones,
    compose_jones_chain,
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
