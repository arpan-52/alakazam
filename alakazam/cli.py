"""
ALAKAZAM CLI.

  alakazam run config.yaml     Run solve + apply blocks
  alakazam info solution.h5    Print solution summary

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import sys
import argparse
import logging
from pathlib import Path


def cmd_run(args):
    """Run calibration pipeline from YAML config."""
    from .config import load_config
    from .pipeline import run_pipeline

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(str(config_path))
    run_pipeline(config)


def cmd_info(args):
    """Print solution summary."""
    from .io.hdf5 import print_summary

    path = Path(args.solution)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    print_summary(str(path))


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="alakazam",
        description="ALAKAZAM — Radio Interferometric Calibration",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run calibration from YAML config")
    p_run.add_argument("config", help="YAML config file")
    p_run.set_defaults(func=cmd_run)

    p_info = sub.add_parser("info", help="Show solution info")
    p_info.add_argument("solution", help="HDF5 solution file")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
