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

import pandas

from yaixm.util import normlevel


def load_airspace(data):
    airspace_dict = [
        {
            "boundary": volume["boundary"],
            "class": volume.get("class") or feature.get("class"),
            "feature_id": feature.get("id"),
            "feature_name": feature["name"],
            "id": volume.get("id"),
            "localtype": feature.get("localtype"),
            "lower": volume["lower"],
            "name": volume.get("name"),
            "normlower": normlevel(volume["lower"]),
            "rules": ",".join(feature.get("rules", []) + volume.get("rules", [])),
            "seq": s if (s := volume.get("seq")) else "ABCDEFGHIJKLM"[n] if len(feature["geometry"]) > 1 else None,
            "type": feature["type"],
            "upper": volume["upper"],
        }
        for feature in data
        for n, volume in enumerate(feature["geometry"])
    ]

    return pandas.DataFrame(airspace_dict)


def load_service(data):
    service_dict = [
        {"id": control, "frequency": service["frequency"]}
        for service in data
        for control in service["controls"]
    ]

    return pandas.DataFrame(service_dict)

def load_obstacle(data):
    obstacle_dict = [
        {
            "boundary": [{"circle": {"centre": obstacle["position"], "radius": "0.5 nm"}}],
            "name": obstacle["name"],
            "lower": "SFC",
            "upper": obstacle["elevation"],
            "type": "D_OTHER",
            "localtype": "OBSTACLE",
            "rules": []
        }
        for obstacle in data
    ]

    return pandas.DataFrame(obstacle_dict)
