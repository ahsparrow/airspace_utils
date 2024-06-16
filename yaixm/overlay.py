from datetime import date
from importlib.resources import files
import io
import json
from math import sqrt

from freetype import Face, FT_LOAD_DEFAULT, FT_LOAD_NO_BITMAP
from geopandas import GeoDataFrame, GeoSeries
import numpy
import pandas
from shapely import (
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    minimum_bounding_radius,
    polygonize,
)
from shapely.affinity import scale, skew, translate
from shapely.ops import polylabel
from sklearn.cluster import KMeans
import yaml

from yaixm.boundary import boundary_polygon
from yaixm.openair import LATLON_FMT, level
from yaixm.util import dms
from yaixm.yaixm import load_airspace

TEXT_SIZE = 3000
SPLIT_RADIUS = 22500


# Get character glyphs from TTF file
def get_glyphs(font, chars):
    # Set font face
    face = Face(font)
    face.set_char_size(1000)

    glyphs = {"normal": {}, "slanted": {}}
    for char in chars:
        face.load_char(char, FT_LOAD_DEFAULT | FT_LOAD_NO_BITMAP)
        outline = face.glyph.outline

        # List of contour slices
        contours = [0] + [c + 1 for c in outline.contours]
        slices = [slice(contours[i], contours[i + 1]) for i in range(len(contours) - 1)]

        # Polygons for glyph
        mp = MultiPolygon([Polygon(outline.points[s]) for s in slices])
        glyphs["normal"][char] = mp
        glyphs["slanted"][char] = skew(mp, 25)

    return glyphs


# Create a Mulitpolygon representation of text
def make_string(glyphs, text, style="normal"):
    offset = 0
    result = MultiPolygon()

    for char in text:
        poly = glyphs[style][char]
        minx, miny, maxx, maxy = poly.bounds

        poly = translate(poly, offset)
        result = result.union(poly)

        offset += maxx + 50

    return result


def annotation_polys(glyphs, point, clearance, annotation, style="normal"):
    # Create annotation polgons
    txt = make_string(glyphs, annotation, style)
    minx, miny, maxx, maxy = txt.bounds

    # Scale to fit space available
    scl = 2 * min(clearance, TEXT_SIZE) / sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
    txt = scale(txt, scl, scl)
    minx, miny, maxx, maxy = txt.bounds

    # Translate to correct postion
    xoff = point.x - (minx + maxx) / 2
    yoff = point.y - (miny + maxy) / 2
    txt = translate(txt, xoff, yoff)

    return txt


# Create OpenAir data
def make_openair(annotation, hdr):
    openair = hdr.splitlines()
    for index, row in annotation.iterrows():
        for poly in row["geometry"].geoms:
            openair.append("AC B")
            openair.append(f"AN {row['name']}")
            openair.append(f"AL {level(row['lower'])}")
            openair.append(f"AH {level(row['upper'])}")
            for vertex in poly.exterior.coords:
                latlon = LATLON_FMT.format(dms(vertex[1]), dms(vertex[0]))
                openair.append(f"DP {latlon}")

    openair.append("")
    return "\n".join(openair)


# Guess best position for annotation
def get_position(polys):
    pos = []
    dist = []
    for p in polys:
        poi = polylabel(p, tolerance=100)
        poi_dist = poi.distance(p.boundary)

        centroid = p.centroid
        centroid_dist = centroid.distance(p.boundary)

        if p.contains(centroid) and centroid_dist > min(TEXT_SIZE, (poi_dist * 0.90)):
            pos.append(centroid)
            dist.append(centroid_dist)
        else:
            pos.append(poi)
            dist.append(poi_dist)

    return pos, dist


# Recursively cluster points
def cluster_points(out, points, max_size):
    if minimum_bounding_radius(points) < max_size:
        out.append(points)
    else:
        # Split points into two clusters using k-means clustering
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
        cluster = kmeans.fit_predict([(p.x, p.y) for p in points.geoms])

        c1 = MultiPoint(numpy.array(points.geoms)[cluster == 0])
        cluster_points(out, c1, max_size)

        c2 = MultiPoint(numpy.array(points.geoms)[cluster == 1])
        cluster_points(out, c2, max_size)


# Split polygons into smaller parts
def poly_splitter(poly, max_size, grid=500):
    if minimum_bounding_radius(poly) < max_size:
        return [poly]

    # Create array of points inside polygon
    minx, miny, maxx, maxy = poly.bounds
    nx = int((maxx - minx) // grid + 1)
    ny = int((maxy - miny) // grid + 1)

    pts = MultiPoint([Point(x * grid, y * grid) for x in range(nx) for y in range(ny)])

    pts = translate(pts, minx, miny)
    pts = poly.intersection(pts)

    # Split points into clusters
    out = []
    cluster_points(out, pts, max_size)

    # Convert point arrays back to polygons
    return [p.buffer(grid * 0.501, cap_style="square").intersection(poly) for p in out]


def overlay(args):
    # Character glyphs for annotation
    glyphs = get_glyphs(
        files("yaixm").joinpath("data").joinpath("asselect.ttf").open("rb"),
        "0123456789DZ",
    )

    # Create pandas dataframe
    with open(args.airspace_file) as f:
        data = yaml.safe_load(f.read())
        df = load_airspace(data["airspace"])

    # Create boundary geometry and convert to GeoDataFrame
    df["geometry"] = df["boundary"].apply(lambda x: boundary_polygon(x, resolution=9))
    gdf = GeoDataFrame(df, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:27700")

    # Filter CTA, etc. and limit base level
    cta_gdf = gdf[
        gdf["type"].isin(["CTA", "CTR", "TMA"])
        & (gdf["normlower"] <= args.max_alt)
        & (gdf["feature_name"] != "BRIZE NORTON CTR")
        & (gdf["rules"].isna() | gdf["rules"].apply(lambda x: x.count("NOTAM") == 0))
    ]

    # Create polygons from union of CTA geometries
    cta_union = cta_gdf.geometry.exterior.unary_union
    cta_polys = GeoSeries(polygonize(cta_union.geoms), crs="EPSG:27700")

    # Remove slivers between not-quite adjacent airspace
    cta_polys = cta_polys.buffer(-300).buffer(299)

    # Convert multipolygon into polygons
    cta_polys = cta_polys.explode(ignore_index=True)

    # Remove empty polygon
    cta_polys = cta_polys[~cta_polys.is_empty]

    # Remove polygons exterior to CTA
    cta_polys = cta_polys[
        [cta_gdf.geometry.contains(p).any() for p in cta_polys.representative_point()]
    ]

    # Merge to single multipolygon
    cta_multipoly = MultiPolygon([p for p in cta_polys if not p.is_empty])

    # Remove ATZ and DZ geometrys for HG/PG overlay
    if args.atzdz:
        # ATZs (with exceptions)
        atz_gdf = gdf[
            (
                (gdf["type"] == "ATZ")
                & ~gdf["feature_name"].isin(
                    [
                        "BARTON",
                        "BIGGIN HILL",
                        "DENHAM",
                        "DERBY",
                        "ELSTREE",
                        "FAIROAKS",
                        "ODIHAM",
                        "REDHILL",
                        "ROCHESTER",
                        "STAPLEFORD",
                        "WHITE WALTHAM",
                    ]
                )
            )
        ]

        # Dropzones
        dz_gdf = gdf[(gdf["localtype"] == "DZ")]

        # Update cta_multipoly by removing ATZs and dropzones
        cta_multipoly = cta_multipoly.difference(
            pandas.concat([atz_gdf.geometry, dz_gdf.geometry]).unary_union
        )

    # Split bigger polygons
    polys = [poly_splitter(p, SPLIT_RADIUS) for p in cta_multipoly.geoms]
    polys = [poly for sublist in polys for poly in sublist]

    # Convert any multipolygons to polygons
    polys = GeoSeries(polys, crs="EPSG:27700").explode(ignore_index=True).geometry

    # Get label positions and distance to edge
    poi, dist = get_position(polys)

    # Create annotation
    annotation = GeoDataFrame({"geometry": []}, crs="EPSG:27700")
    for pos, clearance in zip(poi, dist):
        # Find lowest airspace at point p
        ctas = cta_gdf.cx[pos.x : pos.x + 1, pos.y : pos.y + 1]
        min_ind = ctas["normlower"].argmin()
        lowest_cta = ctas.iloc[min_ind]

        # Skip if base at surface or clearance is too small
        if lowest_cta["normlower"] == 0 or clearance < 750:
            continue

        # Use slanted glyphs for flight levels, upright for altitude
        style = "slanted" if lowest_cta["lower"].startswith("FL") else "normal"

        # Convert height text to polygons
        txt = annotation_polys(
            glyphs, pos, clearance, str(lowest_cta["normlower"] // 100), style
        )

        data = {
            "geometry": txt,
            "name": [lowest_cta["name"] or lowest_cta["feature_name"]],
            "lower": [lowest_cta["lower"]],
            "upper": [lowest_cta["upper"]],
        }
        annotation = pandas.concat(
            [annotation, GeoDataFrame(data, crs="EPSG:27700")], ignore_index=False
        )

    # Annotate ATZ and DZ for HG/PG overlay
    if args.atzdz:
        # DZ annotation
        for i, dz in dz_gdf.iterrows():
            pos = dz.geometry.centroid
            clearance = minimum_bounding_radius(dz.geometry) * 0.75
            txt = annotation_polys(glyphs, pos, clearance, "DZ")

            data = {
                "geometry": txt,
                "name": [dz["feature_name"]],
                "lower": [dz["lower"]],
                "upper": [dz["upper"]],
            }
            annotation = pandas.concat(
                [annotation, GeoDataFrame(data, crs="EPSG:27700")], ignore_index=False
            )

        # ATZ annotation
        for i, a in atz_gdf.iterrows():
            pos = a.geometry.centroid
            if not dz_gdf.geometry.contains(pos).any():
                clearance = minimum_bounding_radius(a.geometry)
                txt = annotation_polys(
                    glyphs, pos, clearance, str(a["upper"].split()[0])
                )

                data = {
                    "geometry": txt,
                    "name": [a["feature_name"]],
                    "lower": [a["lower"]],
                    "upper": [a["upper"]],
                }
                annotation = pandas.concat(
                    [annotation, GeoDataFrame(data, crs="EPSG:27700")],
                    ignore_index=False,
                )

    # Convert to WGS84
    annotation = annotation.to_crs("EPSG:4326")
    if args.debug_file:
        annotation.to_file(args.debug_file)

    # Convert to OpenAir and write to file
    hdr = f"*\n* Height Overlay {args.max_alt}ALT{' ATZ/DZ' if args.atzdz else ''} ({date.today().isoformat()})\n*\n"
    openair = make_openair(annotation, hdr)
    with open(args.output_file, "wt") as f:
        f.write(openair)
