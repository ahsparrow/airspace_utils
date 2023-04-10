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

from geopandas import GeoDataFrame
import numpy as np
from pyproj import Transformer
from pyproj.enums import TransformDirection
from shapely import Polygon

from .helpers import parse_latlon, level


def do_line(line):
    return np.array([parse_latlon(p) for p in line])


def do_circle(circle, resolution):
    transformer = Transformer.from_crs(4326, 27700)

    centre_x, centre_y = transformer.transform(*parse_latlon(circle["centre"]))
    delta = 90 / resolution

    # Get radius (assume in nm)
    radius_str = circle["radius"]
    radius = float(radius_str.split()[0]) * 1852

    # Calculate points on circumference
    angle = np.linspace(0, 2 * np.pi, resolution * 4 + 1)

    x = centre_x + radius * np.cos(angle)
    y = centre_y + radius * np.sin(angle)
    pts = transformer.transform(x, y, direction=TransformDirection.INVERSE)

    return np.array(pts).T


def do_arc(arc, from_latlon, resolution):
    transformer = Transformer.from_crs(4326, 27700)

    from_x, from_y = transformer.transform(*from_latlon)
    to_x, to_y = transformer.transform(*parse_latlon(arc["to"]))
    centre_x, centre_y = transformer.transform(*parse_latlon(arc["centre"]))

    # Get radius, either property or calculated
    if (radius_str := arc.get("radius")):
        # assume in nm
        radius = float(radius_str.split()[0]) * 1852
    else:
        radius = np.sqrt((to_x - centre_x) ** 2 + (to_y - centre_y) ** 2)

    # Angle is zero for due East, and increase anticlockwise
    angle_from = np.arctan2(from_y - centre_y, from_x - centre_x)
    angle_to = np.arctan2(to_y - centre_y, to_x - centre_x)

    if arc["dir"] == "ccw":
        if angle_to < angle_from:
            angle_to += 2 * np.pi

        angle = np.linspace(-np.pi, 3 * np.pi, resolution * 8 + 1)
        angle = angle[(angle > angle_from) & (angle < angle_to)]
    else:
        if angle_to > angle_from:
            angle_from += 2 * np.pi

        angle = np.linspace(3 * np.pi, -np.pi, resolution * 8 + 1)
        angle = angle[(angle < angle_from) & (angle > angle_to)]

    x = centre_x + radius * np.cos(angle)
    y = centre_y + radius * np.sin(angle)
    x = np.append(x, to_x)
    y = np.append(y, to_y)

    pts = transformer.transform(x, y, direction=TransformDirection.INVERSE)

    return np.array(pts).T


def boundary_polygon(boundary, resolution):
    line_strs = []
    for segment in boundary:
        match segment:
            case {"circle": circle}:
                line_str = do_circle(circle, resolution)
            case {"line": line}:
                line_str = do_line(line)
            case {"arc": arc}:
                line_str = do_arc(arc, line_str[-1], resolution)

        line_strs.append(line_str)

    return Polygon(np.fliplr(np.concatenate(line_strs)))


def geojson(airspace, resolution=15, append_seqno=True):
    name_list = []
    class_list = []
    type_list = []
    localtype_list = []
    upper_list = []
    lower_list = []
    normlower_list = []
    rules_list = []
    geometry_list = []

    for feature in airspace:
        for volume in feature["geometry"]:
            # Add properties
            name = volume.get("name") or feature.get("name")
            if append_seqno and "seqno" in volume:
                name = "{} {}".format(name, volume["seqno"])

            name_list.append(name)
            class_list.append(volume.get("class") or feature.get("class"))
            type_list.append(feature["type"])
            localtype_list.append(feature.get("localtype"))
            lower_list.append(volume["lower"])
            upper_list.append(volume["upper"])
            normlower_list.append(level(volume["lower"]))

            rules = feature.get("rules", [])[:]
            rules.extend(volume.get("rules", []))
            rules_list.append("".join(rules))

            geometry_list.append(boundary_polygon(volume["boundary"], resolution))

    df = GeoDataFrame(
        {
            "name": name_list,
            "class": class_list,
            "type": type_list,
            "localtype": localtype_list,
            "upper": upper_list,
            "lower": lower_list,
            "normlower": normlower_list,
            "rules": rules_list,
            "geometry": geometry_list,
        },
        crs="EPSG:4326",
    )

    return df
