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
