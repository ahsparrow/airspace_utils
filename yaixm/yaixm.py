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


def load_airspace(path):
    data = yaml.load(open(path), Loader=yaml.CLoader)

    airspace_dict = [
        {
            "boundary": volume["boundary"],
            "class": volume.get("class") or feature.get("class"),
            "feature_id": feature.get("id"),
            "feature_name": feature["name"],
            "id": volume.get("id"),
            "localtype": feature.get("localtype"),
            "lower": volume["lower"],
            "name": volume.get("name") or feature["name"],
            "normlower": normlevel(volume["lower"]),
            "rules": ",".join(feature.get("rules", []) + volume.get("rules", [])),
            "seqno": str(s) if (s := volume.get("seqno")) else None,
            "type": feature["type"],
            "upper": volume["upper"],
        }
        for feature in data["airspace"]
        for volume in feature["geometry"]
    ]

    return pandas.DataFrame(airspace_dict)
