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

import math

import numpy as np
import pandas

from yaixm.util import dms, parse_latlon
import yaixm.yaixm

LATLON_FMT = (
    "{0[d]:02d}:{0[m]:02d}:{0[s]:02d} {0[ns]} {1[d]:03d}:{1[m]:02d}:{1[s]:02d} {1[ew]}"
)


def name_func(append_freq=False, append_seqno=False):
    def func(volume):
        if volume["name"]:
            name = volume["name"]
        else:
            name = volume['feature_name']

            if (localtype := volume['localtype']):
                if localtype in ["NOATZ", "UL"]:
                    name += " A/F"
                elif localtype in ["MATZ", "DZ", "GVS", "HIRTA", "ILS", "LASER"]:
                    name += " " + localtype

            elif volume['type'] == "ATZ":
                name += " ATZ"

            elif "RAZ" in volume["rules"]:
                name += " " + "RAZ"

            if append_seqno and (seqno := volume["seqno"]):
                name += f"-{seqno}"

            qualifiers = [q for q in ["SI", "NOTAM"] if q in volume["rules"]]
            if qualifiers:
                name += f" ({'/'.join(qualifiers)})"

        if append_freq and not math.isnan(volume["frequency"]):
            name += f" {volume['frequency']:.3f}"

        return name

    return func


def level(level_str):
    if level_str.endswith("ft"):
        # Altitud
        return level_str[:-3] + "ALT"
    else:
        # SFC or FL
        return level_str


def latlon(latlon_str):
    lat, lon = parse_latlon(latlon_str)
    return LATLON_FMT.format(dms(lat), dms(lon))


def openair_type(vol):
    yield f"AC {vol['type']}"


def openair_name(vol, name_func):
    yield f"AN {name_func(vol)}"


def openair_frequency(vol):
    freq = vol["frequency"]
    if not math.isnan(freq):
        yield (f"AF {freq}")


def openair_point(point):
    yield f"DP {latlon(point)}"


def openair_circle(circle):
    centre = circle["centre"]
    radius = circle["radius"].split()[0]

    yield f"V X={latlon(centre)}"
    yield f"DC {radius}"


def openair_arc(arc, prev):
    yield "V D+" if arc["dir"] == "cw" else "V D-"
    yield f"V X={latlon(arc['centre'])}"
    yield f"DB {latlon(prev)}, {latlon(arc['to'])}"


def openair_boundary(boundary):
    first_point = None
    if "line" in boundary[0]:
        first_point = boundary[0]["line"][0]

    for segment in boundary:
        match segment:
            case {"line": points}:
                last_point = points[-1]
                for point in points:
                    yield from openair_point(point)

            case {"circle": circle}:
                yield from openair_circle(circle)

            case {"arc": arc}:
                last_point = arc["to"]
                yield from openair_arc(arc, last_point)

    # Close the polygon if necessary
    if first_point and first_point != last_point:
        yield from openair_point(first_point)


def openair(data, append_freq=False, append_seqno=False):
    airspace = yaixm.yaixm.load_airspace(data["airspace"])
    service = yaixm.yaixm.load_service(data["service"])

    # Merge frequencies
    service.rename(columns={"id": "feature_id"}, inplace=True)
    airspace = pandas.merge(airspace, service, on="feature_id", how="left")

    for _, a in airspace.iterrows():
        yield "*"
        yield from openair_type(a)
        yield from openair_name(a, name_func(append_freq, append_seqno))
        yield from openair_frequency(a)
        yield f"AL {level(a['lower'])}"
        yield f"AH {level(a['upper'])}"
        yield from openair_boundary(a["boundary"])


if __name__ == "__main__":
    import itertools
    import yaml

    airspace = yaml.safe_load(open("/home/ahs/src/airspace/airspace.yaml"))
    service = yaml.safe_load(open("/home/ahs/src/airspace/service.yaml"))

    data = airspace | service

    print("\n".join(openair(data)))
