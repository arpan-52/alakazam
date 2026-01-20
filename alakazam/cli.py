#!/usr/bin/env python3
"""
JACKAL Command-Line Interface.

Commands:
- jackal run config.yaml          : Run calibration from YAML config
- jackal info solution.h5         : Show solution information
- jackal version                  : Show JACKAL version
"""

import sys
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def cmd_run(args):
    """Run calibration from YAML config."""
    from .config_io import solve_from_yaml

    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"JACKAL Calibration")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    try:
        results = solve_from_yaml(config_path)

        print(f"\n{'='*60}")
        print(f"✓ Calibration completed successfully")
        print(f"{'='*60}")

        # Show summary
        for block_name, block_info in results.items():
            print(f"\n{block_name}:")
            print(f"  Jones types: {', '.join(block_info['jones_types'])}")
            print(f"  Output file: {block_info['output_file']}")

        print()

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Calibration failed")
        print(f"{'='*60}")
        print(f"Error: {e}")

        if args.verbose:
            import traceback
            traceback.print_exc()

        sys.exit(1)


def cmd_info(args):
    """Show solution information."""
    from .io import read_solution_hdf5
    import numpy as np

    solution_path = Path(args.solution)

    if not solution_path.exists():
        print(f"Error: Solution file not found: {solution_path}")
        sys.exit(1)

    try:
        result = read_solution_hdf5(solution_path)

        print(f"\nJACKAL Solution Info")
        print(f"{'='*60}")
        print(f"File: {solution_path}")
        print(f"{'='*60}\n")

        # Metadata
        meta = result.metadata
        print(f"Jones Type:      {result.jones_type}")
        print(f"Antennas:        {meta.n_ant}")
        print(f"Channels:        {meta.n_chan}")
        print(f"Times:           {meta.n_time}")
        print(f"Baselines:       {meta.n_row}")
        print(f"Feed Basis:      {meta.feed_basis}")
        print(f"Reference Ant:   {result.ref_ant}")

        # Jones shape
        print(f"\nJones Shape:     {result.jones.shape}")
        print(f"  (time, ant, chan, 2, 2)")

        # Time/freq coverage
        if hasattr(meta, 'time_centroids') and meta.time_centroids is not None:
            t0 = meta.time_centroids.min()
            t1 = meta.time_centroids.max()
            dt = (t1 - t0) * 86400  # days to seconds
            print(f"\nTime Coverage:")
            print(f"  Start:         {t0:.6f} MJD")
            print(f"  End:           {t1:.6f} MJD")
            print(f"  Duration:      {dt:.1f} s")

        if hasattr(meta, 'chan_freq') and meta.chan_freq is not None:
            f0 = meta.chan_freq.min() / 1e9  # Hz to GHz
            f1 = meta.chan_freq.max() / 1e9
            bw = (f1 - f0) * 1e3  # GHz to MHz
            print(f"\nFrequency Coverage:")
            print(f"  Start:         {f0:.3f} GHz")
            print(f"  End:           {f1:.3f} GHz")
            print(f"  Bandwidth:     {bw:.1f} MHz")

        # Convergence info
        if hasattr(result, 'convergence_info') and result.convergence_info:
            info = result.convergence_info
            print(f"\nConvergence:")
            print(f"  Iterations:    {info.get('n_iter', 'N/A')}")
            print(f"  Converged:     {info.get('converged', 'N/A')}")
            if 'final_residual' in info:
                print(f"  Final Resid:   {info['final_residual']:.3e}")

        # Check for flagged data
        n_total = result.jones.size
        n_flagged = np.sum(~np.isfinite(result.jones))
        flag_pct = 100.0 * n_flagged / n_total
        print(f"\nFlags:")
        print(f"  Total values:  {n_total}")
        print(f"  Flagged:       {n_flagged} ({flag_pct:.1f}%)")

        # Check for multiple Jones types in file
        try:
            import h5py
            with h5py.File(solution_path, 'r') as f:
                jones_types = [k for k in f.keys() if k != 'metadata']
                if len(jones_types) > 1:
                    print(f"\nMulti-Jones File:")
                    print(f"  Contains: {', '.join(jones_types)}")
        except:
            pass

        print()

    except Exception as e:
        print(f"Error reading solution: {e}")

        if args.verbose:
            import traceback
            traceback.print_exc()

        sys.exit(1)


def cmd_version(args):
    """Show JACKAL version."""
    from . import __version__

    print(f"JACKAL version {__version__}")
    print(f"Chain-based Algebraic Calibration for Aperture Arrays and Large-scale interferometry")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='jackal',
        description='JACKAL - Fast radio interferometry calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jackal run calibration.yaml       Run calibration from YAML config
  jackal info solution.h5           Show solution information
  jackal version                    Show JACKAL version

For more information, see: https://github.com/yourusername/jackal
        """
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (show tracebacks on errors)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # jackal run
    parser_run = subparsers.add_parser(
        'run',
        help='Run calibration from YAML config',
        description='Run JACKAL calibration from YAML configuration file'
    )
    parser_run.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser_run.set_defaults(func=cmd_run)

    # jackal info
    parser_info = subparsers.add_parser(
        'info',
        help='Show solution information',
        description='Display information about JACKAL solution file'
    )
    parser_info.add_argument(
        'solution',
        type=str,
        help='Path to HDF5 solution file'
    )
    parser_info.set_defaults(func=cmd_info)

    # jackal version
    parser_version = subparsers.add_parser(
        'version',
        help='Show JACKAL version',
        description='Display JACKAL version information'
    )
    parser_version.set_defaults(func=cmd_version)

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
