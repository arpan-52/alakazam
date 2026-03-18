"""ALAKAZAM v1 CLI.

Usage:
  alakazam run config.yaml
  alakazam info cal.h5
  alakazam fluxscale-info cal.h5

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import argparse
import logging
import sys


def main():
    # Suppress casacore C++ warnings (ZENITH direction, etc.)
    import os, warnings
    os.environ["CASACORE_LOG_LEVEL"] = "SEVERE"
    os.environ["CASACORE_LOG_ORIGIN"] = ""
    warnings.filterwarnings("ignore", message=".*ZENITH.*")
    warnings.filterwarnings("ignore", message=".*MeasureHolder.*")

    # Suppress JAX TPU probe message
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

    parser = argparse.ArgumentParser(
        prog="alakazam",
        description="ALAKAZAM v1 — Radio Interferometric Calibration Pipeline",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run pipeline from config YAML")
    p_run.add_argument("config", help="Path to YAML config file")

    # info
    p_info = sub.add_parser("info", help="Print HDF5 solution file summary")
    p_info.add_argument("h5file", help="Path to HDF5 solution file")

    # fluxscale-info
    p_fs = sub.add_parser("fluxscale-info",
                          help="Print fluxscale factors from HDF5 file")
    p_fs.add_argument("h5file", help="Path to HDF5 solution file")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "run":
        from .config import load_config
        from .flow import run_pipeline
        cfg = load_config(args.config)
        run_pipeline(cfg)

    elif args.command == "info":
        from .io.hdf5 import print_summary
        print_summary(args.h5file)

    elif args.command == "fluxscale-info":
        from .io.hdf5 import print_fluxscale_summary
        print_fluxscale_summary(args.h5file)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
