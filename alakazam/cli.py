#!/usr/bin/env python3
"""
ALAKAZAM Command-Line Interface.

Commands:
  alakazam run config.yaml           Run calibration pipeline
  alakazam apply config.yaml         Apply solutions to MS
  alakazam info solution.h5          Show solution information
  alakazam version                   Show version

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import sys
import argparse
import logging
from pathlib import Path


def _setup_logging(verbose: bool = False):
    """Configure logging with Rich handler if available."""
    level = logging.DEBUG if verbose else logging.INFO
    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
    except ImportError:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )


def cmd_run(args):
    """Run calibration pipeline."""
    _setup_logging(args.verbose)
    from .pipeline import run_from_yaml
    run_from_yaml(args.config)


def cmd_apply(args):
    """Apply calibration solutions."""
    _setup_logging(args.verbose)
    from .apply import apply_calibration
    apply_calibration(
        ms_path=args.ms,
        solution_path=args.solution,
        output_col=args.output_col,
    )


def cmd_info(args):
    """Show solution info."""
    from .io.hdf5 import print_summary
    try:
        from rich.console import Console
        Console().print(print_summary(args.solution))
    except ImportError:
        print(print_summary(args.solution))


def cmd_version(args):
    """Show version."""
    from . import __version__
    print(f"ALAKAZAM version {__version__}")
    print("A Radio Interferometric Calibration Suite for Arrays")
    print("Developed by Arpan Pal 2026, NRAO / NCRA")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="alakazam",
        description="ALAKAZAM — A Radio Interferometric Calibration Suite for Arrays",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run calibration from YAML config")
    p_run.add_argument("config", help="Path to YAML config file")
    p_run.set_defaults(func=cmd_run)

    # apply
    p_apply = sub.add_parser("apply", help="Apply calibration solutions to MS")
    p_apply.add_argument("ms", help="Measurement Set path")
    p_apply.add_argument("solution", help="HDF5 solution file")
    p_apply.add_argument("--output-col", default="CORRECTED_DATA", help="Output column")
    p_apply.set_defaults(func=cmd_apply)

    # info
    p_info = sub.add_parser("info", help="Show solution file info")
    p_info.add_argument("solution", help="HDF5 solution file")
    p_info.set_defaults(func=cmd_info)

    # version
    p_ver = sub.add_parser("version", help="Show version")
    p_ver.set_defaults(func=cmd_version)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
