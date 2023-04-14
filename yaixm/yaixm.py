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
import yaml

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
            "seqno": str(s) if (s := volume.get("seqno")) else "ABCDEFGHIJKLM"[n] if len(feature["geometry"]) > 1 else None,
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


def load_loa(data):
    loa_dict = [
        {
            "boundary": volume["boundary"],
            "class": feature.get("class"),
            "loa_name": loa["name"],
            "localtype": feature.get("localtype"),
            "lower": volume["lower"],
            "name": volume.get("name") or feature["name"],
            "normlower": normlevel(volume["lower"]),
            "rules": feature.get("rules", []) + volume.get("rules", []),
            "type": feature["type"],
            "upper": volume["upper"],
        }
        for loa in data
        for area in loa["areas"]
        for feature in area["add"]
        for volume in feature["geometry"]
    ]

    return pandas.DataFrame(loa_dict)
