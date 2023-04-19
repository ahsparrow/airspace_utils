# Copyright 2023 Alan Sparrow
#
# This file is part of YAIXM
#
# YAIXM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# YAIXM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with YAIXM.  If not, see <http://www.gnu.org/licenses/>.

import datetime
import json
import os
import subprocess
import sys

from geopandas import GeoDataFrame
import pandas
import yaml

from yaixm.boundary import boundary_polygon
from yaixm.openair import Type, default_openair, openair, parse as parse_openair
from yaixm.util import (
    get_airac_date,
    validate,
)
from yaixm.yaixm import load_airspace

HEADER = """* UK Airspace
* Alan Sparrow (airspace@asselect.uk)
*
* To the extent possible under law Alan Sparrow has waived all
* copyright and related or neighbouring rights to this file. The data
* is sourced from the UK Aeronautical Information Package (AIP)
*
* AIRAC: {airac_date}
* asselect.uk: Default airspace file
* Commit: {commit}
"""


def check(args):
    # Load airspace
    airspace = yaml.safe_load(open(args.airspace_file))

    # Validate and write any errors to stderr
    e = validate(airspace)
    if e:
        print(e.message, file=sys.stderr)
        sys.exit(1)


def openair(args):
    # Aggregate YAIXM files
    data = {}
    for f in ["airspace", "loa", "obstacle", "rat", "service"]:
        with open(os.path.join(args.yaixm_dir, f + ".yaml")) as f:
            data.update(yaml.safe_load(f))

    # Write default openair data
    filename = args.openair_file or "-"
    f = sys.stdout if filename == "-" else open(filename, "wt", newline="\r\n")
    f.write(default_openair(data))


# Convert either yaxim or openair file to GIS format
def gis(args):
    # Load airspace
    with open(args.airspace_file) as f:
        if args.airspace_file.endswith("yaml"):
            # YAML input
            data = yaml.safe_load(f)
            airspace = load_airspace(data["airspace"])
        else:
            # Openair input
            airspace = pandas.DataFrame(parse_openair(f.read()))

    airspace["geometry"] = airspace["boundary"].apply(
        lambda x: boundary_polygon(x, resolution=args.resolution)
    )
    airspace = airspace.drop("boundary", axis=1)

    # Convert to GeoDataFrame and write to file
    df = GeoDataFrame(airspace, crs="EPSG:4326")
    df.to_file(args.gis_file)


def navplot(args):
    # Load airspace
    with open(args.airspace_file) as f:
        data = yaml.safe_load(f)
        airspace = load_airspace(data["airspace"])

    navplot_airspace = airspace[
        (
            airspace["type"].isin(["CTA", "CTR", "D", "TMA"])
            | (airspace["localtype"] == "MATZ")
        )
        & (airspace["normlower"] < 6000)
    ]

    geometry = navplot_airspace["boundary"].apply(
        lambda x: boundary_polygon(x, resolution=9)
    )

    # Convert to GeoDataFrame and write to file
    df = GeoDataFrame({"geometry": geometry}, crs="EPSG:4326")
    df.to_file(args.navplot_file)


# Convert collection of YAIXM files containing airspace, LOAs and
# obstacles to JSON file with release header and to default Openair file
def release(args):
    # Aggregate YAIXM files
    data = {}
    for f in ["airspace", "loa", "obstacle", "rat", "service"]:
        with open(os.path.join(args.yaixm_dir, f + ".yaml")) as f:
            data.update(yaml.safe_load(f))

    # Append release header
    header = {
        "schema_version": 1,
        "airac_date": get_airac_date(args.offset),
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    # Add notes
    if args.note:
        header["note"] = args.note.read()
    else:
        header["note"] = open(os.path.join(args.yaixm_dir, "release.txt")).read()
    data.update({"release": header})

    # Get Git commit
    try:
        # Get head revision
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", "HEAD"],
            cwd=args.yaixm_dir,
            check=True,
            stdout=subprocess.PIPE,
        )

        # Check for pending commits
        diff = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"], cwd=args.yaixm_dir
        )
        if diff.returncode:
            commit = "changed"
        else:
            commit = head.stdout.decode("ascii").strip()

    except subprocess.CalledProcessError:
        commit = "unknown"

    header["commit"] = commit

    # Validate final output
    error = validate(data)
    if error:
        print(error)
        sys.exit(-1)

    with open(args.yaixm_file, "wt") as f:
        json.dump(data, f, sort_keys=True, indent=args.indent)

    # Default Openair file
    with open(args.openair_file, "wt", newline="\r\n") as f:
        f.write(HEADER.format(airac_date=header["airac_date"][:10], commit=commit))
        f.write(default_openair(data))
