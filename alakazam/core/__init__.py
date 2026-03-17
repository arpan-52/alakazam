"""ALAKAZAM v1 Core subpackage.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from .ms_io import (MSMetadata, detect_metadata, validate_selections,
                    read_data, write_corrected, write_flags,
                    compute_solint_grid, query_times_scans, raw_to_2x2)
from .averaging import (average_per_baseline_full,
                        average_per_baseline_time_only)
from .interpolation import (interpolate_jones, interpolate_jones_multifield)
from .memory import (get_available_ram_gb, get_available_vram_gb,
                     estimate_slot_memory_gb, tier_strategy)
from .rfi import flag_rfi
from .quality import compute_quality
