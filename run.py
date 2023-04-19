#!/usr/bin/env python
import argparse
import sys

import yaixm.cli
import yaixm.cli_util
import yaixm.overlay


def cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="commands", dest="cmd", required=True)

    # check sub-command
    sub_parser = subparsers.add_parser("check", help="check against schema")
    sub_parser.add_argument("airspace_filepath", help="YAML airspace file")
    sub_parser.set_defaults(func=yaixm.cli.check)

    # convert sub-command
    sub_parser = subparsers.add_parser("gis", help="convert to GIS format")
    sub_parser.add_argument(
        "airspace_filepath", help="airspace file (YAIXM or Openair)"
    )
    sub_parser.add_argument("gis_filepath", help="GIS output file")
    sub_parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=15,
        help="angular resolution, per 90 degrees",
    )
    sub_parser.set_defaults(func=yaixm.cli.gis)

    # navplot sub-command
    sub_parser = subparsers.add_parser("navplot", help="make NavPlot airspace")
    sub_parser.add_argument("airspace_filepath", help="Airspace input file")
    sub_parser.add_argument("navplot_filepath", help="NavPlot output file")
    sub_parser.set_defaults(func=yaixm.cli.navplot)

    # openair sub-command
    sub_parser = subparsers.add_parser("openair", help="convert to OpenAir")
    sub_parser.add_argument(
        "airspace_file",
        nargs="?",
        help="YAML airspace file",
        type=argparse.FileType("r"),
        default=sys.stdin,
    )
    sub_parser.add_argument(
        "openair_file",
        nargs="?",
        help="Openair output file, stdout if not specified",
        type=argparse.FileType("w"),
        default=sys.stdout,
    )
    sub_parser.add_argument("--comp", help="Competition airspace", action="store_true")
    sub_parser.set_defaults(func=yaixm.cli.openair)

    # release sub-command
    sub_parser = subparsers.add_parser("release", help="make ASSelect airspace")
    sub_parser.add_argument("yaixm_dir", help="YAML input directory")
    sub_parser.add_argument(
        "yaixm_file", type=argparse.FileType("w"), help="JSON output file"
    )
    sub_parser.add_argument("openair_file", help="OpenAir output file")
    sub_parser.add_argument(
        "--indent",
        "-i",
        type=int,
        default=None,
        help="JSON file indentation level (default none)",
    )
    sub_parser.add_argument(
        "--note",
        "-n",
        help="Release note file",
        type=argparse.FileType("r"),
        default=None,
    )
    group = sub_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--prev",
        "-p",
        action="store_const",
        default=0,
        dest="offset",
        const=-28,
        help="Use previous AIRAC date",
    )
    group.add_argument(
        "--next",
        action="store_const",
        default=0,
        dest="offset",
        const=28,
        help="Use next AIRAC date",
    )
    sub_parser.set_defaults(func=yaixm.cli.release)

    # overlay sub-command
    sub_parser = subparsers.add_parser("overlay", help="make height overlay")
    sub_parser.add_argument("airspace_filepath", help="YAML airspace file")
    sub_parser.add_argument(
        "output_file",
        nargs="?",
        help="Openair output file, stdout if not specified",
        type=argparse.FileType("wt"),
        default=sys.stdout,
    )
    sub_parser.add_argument(
        "--max_alt", type=int, default=10400, help="Maximum base altitude"
    )
    sub_parser.add_argument(
        "--debug_file",
        help="GeoJSON output file for debug",
        type=argparse.FileType("wb"),
    )
    sub_parser.set_defaults(func=yaixm.overlay.overlay)

    # ils sub-command
    sub_parser = subparsers.add_parser("ils", help="calculate ILS coordinates")
    sub_parser.add_argument("lat", help="Centre latitude, DMS e.g. 512345N")
    sub_parser.add_argument("lon", help="Centre longitude, DMS e.g. 0012345W")
    sub_parser.add_argument("bearing", type=float, help="Runway bearing, degrees")
    sub_parser.add_argument(
        "radius", type=float, nargs="?", default=2, help="ATZ radius, in nm (default 2)"
    )
    sub_parser.set_defaults(func=yaixm.cli_util.calc_ils)

    # stub sub-command
    sub_parser = subparsers.add_parser("stub", help="calculate MATZ stub coordinates")
    sub_parser.add_argument("lat", help="Centre latitude, DMS e.g. 512345N")
    sub_parser.add_argument("lon", help="Centre longitude, DMS e.g. 0012345W")
    sub_parser.add_argument("bearing", type=float, help="R/W bearing, degrees (true)")
    sub_parser.add_argument(
        "length", type=float, nargs="?", default=5, help="Stub length (nm)"
    )
    sub_parser.add_argument(
        "width", type=float, nargs="?", default=4, help="Stub width (nm)"
    )
    sub_parser.add_argument(
        "radius", type=float, nargs="?", default=5, help="Circle radius (nm)"
    )
    sub_parser.set_defaults(func=yaixm.cli_util.calc_stub)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()
