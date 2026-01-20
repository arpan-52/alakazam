"""
ALAKAZAM CLI.

Usage:
    alakazam run config.yaml
    alakazam run config.yaml --ref-ant 5
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog='alakazam',
        description='ALAKAZAM - Radio Interferometry Calibration'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # run command
    run_parser = subparsers.add_parser('run', help='Run calibration from YAML config')
    run_parser.add_argument('config', help='YAML configuration file')
    run_parser.add_argument('--ref-ant', '-r', type=int, default=None,
                           help='Override reference antenna')
    run_parser.add_argument('--verbose', '-v', action='store_true', default=True,
                           help='Verbose output')
    run_parser.add_argument('--quiet', '-q', action='store_true',
                           help='Quiet output')

    # info command
    info_parser = subparsers.add_parser('info', help='Show calibration table info')
    info_parser.add_argument('table', help='HDF5 calibration table')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == 'run':
        from alakazam.pipeline import run_pipeline
        run_pipeline(args.config, verbose=not args.quiet)

    elif args.command == 'info':
        from alakazam.io import print_summary
        print_summary(args.table)


if __name__ == '__main__':
    main()
