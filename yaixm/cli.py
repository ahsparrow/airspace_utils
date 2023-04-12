# Copyright 2017 Alan Sparrow
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
from pyogrio import write_dataframe

from .convert import Openair, seq_name, make_filter, make_openair_type
from .helpers import get_airac_date, level, load, merge_loa, merge_service, validate
from .geojson import boundary_polygon
from .parse_openair import parse as parse_openair
from yaixm.yaixm import load_airspace

HEADER = """UK Airspace
Alan Sparrow (airspace@asselect.uk)

To the extent possible under law Alan Sparrow has waived all
copyright and related or neighbouring rights to this file. The data
is sourced from the UK Aeronautical Information Package (AIP)\n\n"""

def check(args):
    # Load airspace
    airspace = load(args.airspace_file)

    # Validate and write any errors to stderr
    e = validate(airspace)
    if e:
        print(e.message, file=sys.stderr)
        sys.exit(1)

def openair(args):
    # Load airspace
    airspace = load(args.airspace_file)

    # Convert to openair
    if args.comp:
        convert = Openair(name_func=seq_name, type_func=make_openair_type(comp=True))
    else:
        convert = Openair()
    oa = convert.convert(airspace['airspace'])

    # Don't accept anything other than ASCII
    output_oa = oa.encode("ascii").decode("ascii")

    args.openair_file.write(output_oa)

def jsonify(args):
    data = load(args.yaml_file)
    json.dump(data, args.json_file, sort_keys=args.sort, indent=args.indent)

    if args.json_file is sys.stdout:
        print()

def merge(args):
    yaixm = load(args.input_file)
    airspace = yaixm['airspace']
    loa = yaixm['loa']

    loa_names = [x.strip() for x in args.merge.split(",")]

    if loa_names[0]:
        loa = [x for x in loa if x['name'] in loa_names]

    merged = {'airspace': merge_loa(airspace, loa)}

    json.dump(merged, args.output_file, sort_keys=True, indent=4)


# Convert either yaxim or openair file to GIS format
def gis(args):
    # Load airspace
    if args.airspace_filepath.endswith('yaml'):
        # YAML input
        airspace = load_airspace(args.airspace_filepath)
    else:
        # Openair input
        airspace = {'airspace': parse_openair(args.airspace_file.read())}

    airspace["geometry"] = airspace["boundary"].apply(lambda x: boundary_polygon(x, resolution=args.resolution))
    airspace = airspace.drop("boundary", axis=1)

    # Convert to GeoDataFrame and write to file
    df = GeoDataFrame(airspace, crs="EPSG:4326")
    write_dataframe(df, args.gis_filepath)


def navplot(args):
    # Load airspace
    yaixm = load(args.yaixm_file)

    volumes = []
    for feature in yaixm["airspace"]:
        for volume in feature["geometry"]:
            rules = feature.get('rules', []) + volume.get('rules', [])

            if (feature["type"] in ["CTA", "CTR", "D", "TMA"] or
                feature["type"] == "OTHER" and feature["localtype"] == "MATZ"):

                if level(volume["lower"]) < 6000 and "NOTAM" not in rules:
                    volumes.append(volume)

    dummy = [{'geometry': volumes, 'name': "DUMMY", 'type': "OTHER"}]

    # Convert to GeoJSON
    gjson = convert_geojson(dummy, resolution=15)

    json.dump(gjson, args.navplot_file, sort_keys=True, indent=4)

# Convert collection of YAIXM files containing airspace, LOAs and
# obstacles to JSON file with release header and to default Openair file
def release(args):
    # Aggregate YAIXM files
    out = {}
    for f in ["airspace", "loa", "obstacle", "rat", "service"]:
        out.update(load(open(os.path.join(args.yaixm_dir, f + ".yaml"))))

    # Append release header
    header = {
        'schema_version': 1,
        'airac_date': get_airac_date(args.offset),
        'timestamp': datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    }

    # Add notes
    if args.note:
        header['note'] = args.note.read()
    else:
        header['note'] = open(os.path.join(args.yaixm_dir, "release.txt")).read()
    out.update({'release': header})

    # Get Git commit
    try:
        # Get head revision
        head = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"],
                              cwd=args.yaixm_dir, check=True,
                              stdout=subprocess.PIPE)

        # Check for pending commits
        diff = subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"],
                              cwd=args.yaixm_dir)
        if diff.returncode:
            commit = "changed"
        else:
            commit = head.stdout.decode("ascii").strip()

    except subprocess.CalledProcessError:
        commit = "unknown"

    header['commit'] = commit

    # Validate final output
    error = validate(out)
    if error:
        print(error)
        sys.exit(-1)

    json.dump(out, args.yaixm_file, sort_keys=True, indent=args.indent)

    # Default Openair file
    hdr = HEADER
    hdr += f"AIRAC: {header['airac_date'][:10]}\n"
    hdr += "asselect.uk: Default airspace file\n"
    hdr += f"Commit: {commit}\n"
    hdr += "\n"
    if args.note:
        hdr += header['note']

    loas = [loa for loa in out['loa'] if loa['name'] == "CAMBRIDGE RAZ"]
    airspace = merge_loa(out['airspace'], loas)

    services = {}
    for service in out['service']:
        for control in service['controls']:
            services[control] = service['frequency']
    airspace = merge_service(airspace, services)

    type_func = make_openair_type(atz="CTR", ils="G", glider="W", noatz="G",
                                  ul="G", comp=False)

    exclude = [{'name': a['name'], 'type': "D_OTHER"} for a in out['airspace']
            if "TRA" in a.get('rules', []) or "NOSSR" in a.get('rules', [])]
    filter_func = make_filter(microlight=False, hgl=False, exclude=exclude)

    convert = Openair(type_func=type_func, filter_func=filter_func, header=hdr)
    oa = convert.convert(airspace)

    # Don't accept anything other than ASCII
    output_oa = oa.encode("ascii").decode("ascii")

    # Convert to DOS format
    dos_oa = output_oa.replace("\n", "\r\n") + "\r\n"

    args.openair_file.write(dos_oa)
