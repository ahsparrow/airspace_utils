# Copyright 2017 Alan Sparrow
#
# This file is part of YAIXM utils
#
# YAIXM utils is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# YAIXM utils is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with YAIXM utils.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import json
import math
import os.path
import re
import subprocess
import sys
import tempfile

from pygeodesy.ellipsoidalVincenty import LatLon
import yaml

from .helpers import ordered_map_representer, parse_deg
from .obstacle import make_obstacles

# Convert obstacle data XLS spreadsheet from AIS to YAXIM format
def convert_obstacle(args):
    # Using temporary working directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Convert xls to xlsx
        subprocess.run(["libreoffice",
             "--convert-to", "xlsx",
             "--outdir", tmp_dir,
             args.obstacle_xls], errors=True)

        base_xls = os.path.basename(args.obstacle_xls)
        base_xlsx = os.path.splitext(base_xls)[0] + ".xlsx"
        xlsx_name = os.path.join(tmp_dir, base_xlsx)

        # Convert xlsx to CSV
        csv_name = os.path.join(tmp_dir, "obstacle.csv")
        subprocess.run(["xlsx2csv",
                        "--sheetname" , "All", xlsx_name, csv_name],
                       errors=True)

        obstacles = make_obstacles(open(csv_name), args.names)

    # Write to YAML file
    yaml.add_representer(dict, ordered_map_representer)
    yaml.dump({'obstacle': obstacles},
              args.yaml_file, default_flow_style=False)

def calc_ils(args):
    lon = parse_deg(args.lon)
    lat = parse_deg(args.lat)
    centre = LatLon(lat, lon)

    bearing = args.bearing + 180
    radius = args.radius * 1852

    distances = [radius, 8 * 1852, 8 * 1852, radius]
    bearings = [bearing -3, bearing -3, bearing + 3, bearing + 3]

    for d, b in zip(distances, bearings):
        p = centre.destination(d, b)
        print("- %s" % p.toStr(form="sec", prec=0, sep=" "))

def calc_point(args):
    lon = parse_deg(args.lon)
    lat = parse_deg(args.lat)
    origin = LatLon(lat, lon)

    dist = args.distance * 1852

    p = origin.destination(dist, args.bearing)
    print(p.toStr(form="sec", prec=0, sep=" "))

def calc_stub(args):
    lon = parse_deg(args.lon)
    lat = parse_deg(args.lat)
    centre = LatLon(lat, lon)

    length = args.length * 1852
    width = args.width * 1852
    radius = args.radius * 1852

    # Inner stub
    theta = math.asin(width / (2 * radius))

    bearing = args.bearing + 180 - math.degrees(theta)
    p1 = centre.destination(radius, bearing)

    bearing = args.bearing + 180 + math.degrees(theta)
    p2 = centre.destination(radius, bearing)

    print("Inner:")
    print(p1.toStr(form="sec", prec=0, sep=" "))
    print(p2.toStr(form="sec", prec=0, sep=" "))

    # Outer stub
    dist = math.sqrt((radius + length) ** 2 + (width / 2) **2)
    theta = math.atan(width / (2 * (radius + length)))

    bearing = args.bearing + 180 + math.degrees(theta)
    p1 = centre.destination(dist, bearing)

    bearing = args.bearing + 180 - math.degrees(theta)
    p2 = centre.destination(dist, bearing)

    print("\nOuter:")
    print(p1.toStr(form="sec", prec=0, sep=" "))
    print(p2.toStr(form="sec", prec=0, sep=" "))

# Check services exist in airspace file
def check_service(args):
    service = yaml.safe_load(args.service_file)
    airspace = yaml.safe_load(args.airspace_file)

    airspace = airspace['airspace']
    service = service['service']

    ids = [feature['id'] for feature in airspace if feature.get('id')]
    for s in service:
        for c in s['controls']:
            if c not in ids:
                print("Missing:", c)
