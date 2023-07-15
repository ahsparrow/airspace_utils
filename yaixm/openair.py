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

from lark import Lark, Transformer
import pandas

from yaixm.util import dms, parse_latlon
import yaixm.yaixm

LATLON_FMT = (
    "{0[d]:02d}:{0[m]:02d}:{0[s]:02d} {0[ns]} {1[d]:03d}:{1[m]:02d}:{1[s]:02d} {1[ew]}"
)


class Type(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    CTR = "CTR"
    DANGER = "Q"
    PROHIBITED = "P"
    RESTRICTED = "R"
    W = "W"


Grammer = """
    ?start: feature_list
    feature_list: feature+

    feature: airtype airname (freq? upper lower | freq? lower upper | upper lower freq | lower upper freq) boundary

    airtype: "AC" _WS_INLINE+ AIRTYPE _NEWLINE
    airname: "AN" _WS_INLINE+ NAME_STRING _NEWLINE
    lower: "AL" _WS_INLINE + (ALT | FL | SFC) _NEWLINE
    upper: "AH" _WS_INLINE+ (ALT | FL) _NEWLINE
    freq: "AF" _WS_INLINE+ FREQ _NEWLINE

    boundary: (line | circle | arc)+

    line: point+
    circle: centre radius
    arc: dir centre limits

    ?point: "DP" _WS_INLINE+ lat_lon _NEWLINE

    centre: "V" _WS_INLINE+ "X=" lat_lon _NEWLINE
    radius: "DC" _WS_INLINE+ RADIUS _NEWLINE

    dir: "V" _WS_INLINE+ "D=" DIRECTION _NEWLINE
    limits: "DB" _WS_INLINE+ lat_lon "," _WS_INLINE+ lat_lon _NEWLINE

    ?lat_lon: LAT_LON

    AIRTYPE: LETTER+

    NAME_STRING: LETTER (NAME_CHAR | " ")* NAME_CHAR 
    NAME_CHAR: (LETTER | DIGIT | "(" | ")" | "/" | "-" | "." | "'")

    ALT: DIGIT+ "ALT"
    FL: "FL" DIGIT+
    SFC: "SFC"

    FREQ: DIGIT~3 "." DIGIT~3

    RADIUS: NUMBER
    DIRECTION: ("+" | "-")

    LAT_LON: LAT WS_INLINE+ LON
    LAT: DIGIT~2 ":" DIGIT~2 ":" DIGIT~2 WS_INLINE+ LAT_HEMI
    LON: DIGIT~3 ":" DIGIT~2 ":" DIGIT~2 WS_INLINE+ LON_HEMI
    LAT_HEMI: ("N" | "S")
    LON_HEMI: ("E" | "W")

    _NEWLINE: NEWLINE
    _WS_INLINE: WS_INLINE

    COMMENT: /\*[^\\n]*/ NEWLINE
    %ignore COMMENT

    %import common.DIGIT
    %import common.LETTER
    %import common.NEWLINE
    %import common.NUMBER
    %import common.WS_INLINE
"""


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
        name += f" {volume['frequency']:.03f}"

    return name


def typer(volume, types, format):
    if "NOTAM" in volume["rules"]:
        out = "G"
    elif "TMZ" in volume["rules"]:
        out = "TMZ"
    elif "RMZ" in volume["rules"]:
        out = "RMZ"
    else:
        comp = format == "competition"
        match volume["type"]:
            case "ATZ":
                out = types["atz"].value
            case "D":
                out = "P" if comp and "SI" in volume["rule"] else "Q"
            case "D_OTHER":
                if volume["localtype"] == "GLIDER":
                    out = "W"
                elif (
                    comp
                    and volume["localtype"] == "DZ"
                    and "INTENSE" in volume["rules"]
                ):
                    out = "P"
                elif volume["localtype"] in ["HIRTA", "GVS", "LASER"]:
                    out = types["hirta"].value
                elif volume["localtype"] == "OBSTACLE":
                    out = types["obstacle"].value
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
        # Altitude
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
        yield (f"AF {freq:.03f}")


def openair_point(point):
    yield f"DP {latlon(point)}"


def openair_circle(circle):
    centre = circle["centre"]
    radius = circle["radius"].split()[0]

    yield f"V X={latlon(centre)}"
    yield f"DC {radius}"


def openair_arc(arc, prev):
    yield "V D=+" if arc["dir"] == "cw" else "V D=-"
    yield f"V X={latlon(arc['centre'])}"
    yield f"DB {latlon(prev)}, {latlon(arc['to'])}"


def openair_boundary(boundary):
    first_point = None
    if "line" in boundary[0]:
        first_point = boundary[0]["line"][0]

    for segment in boundary:
        match segment:
            case {"line": points}:
                for point in points:
                    yield from openair_point(point)
                last_point = points[-1]

            case {"circle": circle}:
                yield from openair_circle(circle)

            case {"arc": arc}:
                yield from openair_arc(arc, last_point)
                last_point = arc["to"]

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


def openair_generator(
    data, types, format, home, max_level, append_freq, loa_names, wave_names, rat_names
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

    # Add obstacles
    if types.get("obstacle"):
        obstacles = yaixm.yaixm.load_obstacle(data["obstacle"])
        airspace = pandas.concat([airspace, obstacles])

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
    return "".join(
        f"{line}\n"
        for line in openair_generator(
            data,
            types,
            format=format,
            home=home,
            max_level=max_level,
            append_freq=append_freq,
            loa_names=loa_names,
            rat_names=rat_names,
            wave_names=wave_names,
        )
    )

    return oa


def default_openair(data):
    types = {
        "atz": Type.CTR,
        "ils": Type.G,
        "noatz": Type.G,
        "ul": None,
        "hirta": None,
        "glider": Type.W,
    }
    loa_names = [loa["name"] for loa in data["loa"] if loa.get("default")]
    return openair(data, types, append_freq=True, loa_names=loa_names)


class OpenairTransformer(Transformer):
    def LAT_LON(self, latlon):
        t = latlon.replace(":", "").replace(" ", "")
        return t[:7] + " " + t[7:]

    def DIRECTION(self, dirn):
        return "cw" if dirn == "+" else "ccw"

    def RADIUS(self, r):
        return r + " nm"

    def SFC(self, sfc):
        return sfc.value

    def FL(self, fl):
        return fl.value

    def ALT(self, alt):
        return alt[:-3] + " ft"

    def NAME_STRING(self, str):
        return str.value

    def AIRTYPE(self, str):
        return str.value

    def FREQ(self, str):
        return str.value

    def limits(self, data):
        return "to", data[1]

    def dir(self, data):
        return "dir", data[0]

    def radius(self, tree):
        return ("radius", tree[0])

    def centre(self, tree):
        return "centre", tree[0]

    def arc(self, tree):
        return "arc", dict(tree)

    def circle(self, tree):
        return "circle", dict(tree)

    def line(self, tree):
        return "line", tree

    def boundary(self, tree):
        return "boundary", [dict([x]) for x in tree]

    def upper(self, data):
        return "upper", data[0]

    def lower(self, data):
        return "lower", data[0]

    def airname(self, data):
        return "name", data[0]

    def airtype(self, data):
        return "type", data[0]

    def freq(self, data):
        return "freq", data[0]

    feature = dict
    feature_list = list


def parse(data):
    parser = Lark(Grammer)
    tree = parser.parse(data)

    out = OpenairTransformer().transform(tree)
    return out


if __name__ == "__main__":
    import itertools
    import yaml

    airspace = yaml.safe_load(open("/home/ahs/src/airspace/airspace.yaml"))
    service = yaml.safe_load(open("/home/ahs/src/airspace/service.yaml"))
    rat = yaml.safe_load(open("/home/ahs/src/airspace/rat.yaml"))
    loa = yaml.safe_load(open("/home/ahs/src/airspace/loa.yaml"))
    obstacle = yaml.safe_load(open("/home/ahs/src/airspace/obstacle.yaml"))

    data = airspace | service | rat | loa | obstacle

    types = {
        "atz": Type.CTR,
        "ils": Type.G,
        "noatz": Type.F,
        "ul": None,
        "hirta": None,
        "glider": Type.W,
        "obstacle": None,
    }

    rat_names = []
    wave_names = []
    loa_names = []

    oa = openair(
        data,
        types,
        max_level=66000,
        home="RIVAR HILL",
        loa_names=loa_names,
        rat_names=rat_names,
        wave_names=wave_names,
    )
    print(oa)
