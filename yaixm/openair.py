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

from enum import StrEnum
import math

import numpy as np
import pandas

from yaixm.util import dms, parse_latlon
import yaixm.yaixm

LATLON_FMT = (
    "{0[d]:02d}:{0[m]:02d}:{0[s]:02d} {0[ns]} {1[d]:03d}:{1[m]:02d}:{1[s]:02d} {1[ew]}"
)


class Type(StrEnum):
    A = "A"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    CTR = "CTR"
    DANGER = "Q"
    PROHIBITED = "P"
    RESTRICTED = "R"
    W = "W"


def namer(volume, append_freq, append_seqno):
    if volume["name"]:
        name = volume["name"]
    else:
        name = volume["feature_name"]

        if localtype := volume["localtype"]:
            if localtype in ["NOATZ", "UL"]:
                name += " A/F"
            elif localtype in ["MATZ", "DZ", "GVS", "HIRTA", "ILS", "LASER"]:
                name += " " + localtype

        elif volume["type"] == "ATZ":
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


def typer(volume, types, format):
    if "NOTAM" in volume["rules"]:
        out = "G"
        return

    comp = format == "competition"
    match volume["type"]:
        case "ATZ":
            out = types["atz"].value
        case "D":
            out = "P" if comp and "SI" in volume["rule"] else "Q"
        case "D_OTHER":
            if volume["localtype"] == "GLIDER":
                out = "W"
            elif comp and volume["localtype"] == "DZ" and "INTENSE" in volume["rules"]:
                out = "P"
            elif volume["localtype"] in ["HIRTA", "GVS", "LASER"]:
                out = types["hirta"].value
            else:
                out = "Q"
        case "OTHER":
            match volume["localtype"]:
                case "GLIDER":
                    out = "W" if "LOA" in volume["rules"] else types["glider"].value
                case "ILS" | "NOATZ" | "UL" as oatype:
                    out = types[oatype.lower()].value
                case "MATZ" | "TMZ" | "RMZ" as oatype:
                    out = oatype
                case "RAT":
                    out = "P"
                case _:
                    out = "OTHER"
        case "P" | "R" | "TMZ" | "RMZ" as oatype:
            out = oatype
        case _:
            out = volume["class"]

    return out


def make_filter(types, max_level, home, wave):
    def func(data):
        exc = False

        # Training airfields
        exc = exc or data["localtype"] == "NOATZ" and types["noatz"] == None

        # Microlight strips
        exc = exc or data["localtype"] == "UL" and types["ul"] == None

        # HIRTAs, etc
        exc = exc or (
            data["localtype"] in ["HIRTA", "GVS", "LASER"] and types["hirta"] == None
        )

        # Gliding sites
        exc = exc or (
            data["type"] == "OTHER"
            and data["localtype"] == "GLIDER"
            and not "LOA" in data["rules"]
            and (types["glider"] == None or home == data["feature_name"])
        )

        # Maximum level
        exc = exc or data["normlower"] >= max_level

        # Wave boxes (excluded by default)
        exc = exc or (
            data["type"] == "D_OTHER"
            and data["localtype"] == "GLIDER"
            and "LOA" not in data["rules"]
            and data["feature_name"] not in wave
        )

        return not exc

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


def openair_type(vol, types, format):
    oa_type = typer(vol, types, format)
    yield f"AC {oa_type}"


def openair_name(vol, append_freq, format):
    name = namer(vol, append_freq, format == "competition")
    yield f"AN {name}"


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


def merge_loa(airspace, loa_data):
    # Add new features
    add = [
        yaixm.yaixm.load_airspace(area["add"])
        for loa in loa_data
        for area in loa["areas"]
    ]
    airspace = pandas.concat([airspace] + add)

    # Replace existing volumes
    replace_vols = []
    for loa in loa_data:
        for area in loa["areas"]:
            for replace in area.get("replace", []):
                # Volume to be replaced
                vol = airspace[airspace["id"] == replace["id"]].iloc[0]

                # Remove it
                airspace = airspace[airspace["id"] != replace["id"]]

                # Make new volumes
                dfs = []
                for v in replace["geometry"]:
                    vol["boundary"] = v["boundary"]
                    vol["upper"] = v["upper"]
                    vol["lower"] = v["lower"]
                    replace_vols.append(vol.copy())

    return pandas.concat([airspace, pandas.DataFrame(replace_vols)])


def openair(
    data,
    types,
    format="openair",
    home="",
    max_level=19500,
    append_freq=False,
    loa_names=[],
    wave_names=[],
    rat_names=[],
):
    airspace = yaixm.yaixm.load_airspace(data["airspace"])
    service = yaixm.yaixm.load_service(data["service"])
    rats = yaixm.yaixm.load_airspace(data["rat"])

    # Merge selected LOAs
    loa_data = list(filter(lambda x: x["name"] in loa_names, data["loa"]))
    airspace = merge_loa(airspace, loa_data)

    # Add RATs
    airspace = pandas.concat(
        [airspace, rats[rats["feature_name"].map(lambda x: x in rat_names)]]
    )

    # Filter airspace
    airspace = airspace[
        airspace.apply(make_filter(types, max_level, home, wave=wave_names), axis=1)
    ]

    # Merge frequencies
    service.rename(columns={"id": "feature_id"}, inplace=True)
    airspace = pandas.merge(airspace, service, on="feature_id", how="left")

    for _, a in airspace.iterrows():
        yield "*"
        yield from openair_type(a, types, format)
        yield from openair_name(a, append_freq, format)
        yield from openair_frequency(a)
        yield f"AL {level(a['lower'])}"
        yield f"AH {level(a['upper'])}"
        yield from openair_boundary(a["boundary"])


if __name__ == "__main__":
    import itertools
    import yaml

    airspace = yaml.safe_load(open("/home/ahs/src/airspace/airspace.yaml"))
    service = yaml.safe_load(open("/home/ahs/src/airspace/service.yaml"))
    rat = yaml.safe_load(open("/home/ahs/src/airspace/rat.yaml"))
    loa = yaml.safe_load(open("/home/ahs/src/airspace/loa.yaml"))

    data = airspace | service | rat | loa

    types = {
        "atz": Type.CTR,
        "ils": Type.G,
        "noatz": Type.F,
        "ul": None,
        "hirta": None,
        "glider": Type.W,
    }

    rat_names = []
    wave_names = []
    loa_names = ["BATH GAP"]

    oa = "".join(
        f"{line}\n"
        for line in openair(
            data,
            types,
            max_level=66000,
            home="RIVAR HILL",
            loa_names=loa_names,
            rat_names=rat_names,
            wave_names=wave_names,
        )
    )
    print(oa)
