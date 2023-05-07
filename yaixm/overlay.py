from importlib.resources import files
import io
import json
from math import sqrt

from freetype import Face, FT_LOAD_DEFAULT, FT_LOAD_NO_BITMAP
from geopandas import read_file, GeoDataFrame
import numpy
import pandas
from shapely import MultiPoint, MultiPolygon, Point, Polygon, minimum_bounding_radius
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
        glyphs["slanted"][char] = skew(mp, 20)

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


# Create OpenAir data
def make_openair(annotation):
    openair = []
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


# Calculate best position for annotation
def get_position(polys):
    pos = []
    dist = []
    for p in polys:
        poi = polylabel(p, tolerance=10)
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
    # Create geopandas GeoDataFrame
    with open(args.airspace_file) as f:
        data = yaml.safe_load(f.read())
        df = load_airspace(data["airspace"])

    # Calculate X/Y geometry
    df["geometry"] = df["boundary"].apply(lambda x: boundary_polygon(x, resolution=9))
    gdf = GeoDataFrame(df, crs="EPSG:4326")

    # Filter CTA, etc. and limit base level
    cta = gdf[
        gdf["type"].isin(["CTA", "CTR", "TMA"])
        & (gdf["normlower"] <= args.max_alt)
        & (gdf["feature_name"] != "BRIZE NORTON CTR")
        & (gdf["rules"].isna() | gdf["rules"].apply(lambda x: x.count("NOTAM") == 0))
    ]
    cta_geom = GeoDataFrame({"geometry": cta.geometry})

    # Make union of CTA geometries
    geom = cta_geom[0:1]
    for i in range(1, len(cta_geom)):
        geom = geom.overlay(cta_geom[i : i + 1], how="union", keep_geom_type=True)

    if args.hgpg:
        # ATZs, DZs and Brize
        atzdz = gdf[
            (gdf["type"] == "ATZ")
            | (gdf["localtype"] == "DZ")
            | (gdf["feature_name"] == "BRIZE NORTON CTR")
        ]
        atzdz_geom = GeoDataFrame({"geometry": atzdz.geometry})

        # Remove ATZ geometries
        for i in range(0, len(atzdz)):
            geom = geom.overlay(
                atzdz_geom[i : i + 1], how="difference", keep_geom_type=True
            )

    # Convert to X/Y coordinates
    cta = cta.to_crs("EPSG:27700")
    geom = geom.to_crs("EPSG:27700")

    # Remove slivers between not-quite-adjacent bits of airspace
    geom = geom.geometry.buffer(-300).buffer(300)
    geom = geom.explode(ignore_index=True)
    geom = geom[~geom.geometry.is_empty]

    # Split bigger polygons
    polys = [poly_splitter(p, SPLIT_RADIUS) for p in geom.geometry]
    polys = [poly for sublist in polys for poly in sublist]

    # Convert multipolygons to polygons and discard anything else
    geom = GeoDataFrame({"geometry": polys}, crs="EPSG:27700")
    geom = geom.explode(ignore_index=True)
    geom = geom[geom.geometry.type == "Polygon"]

    # Get label positions and distance to edge
    poi, dist = get_position(geom.geometry)

    # Character glyphs
    glyphs = get_glyphs(
        files("yaixm").joinpath("data").joinpath("asselect.ttf").open("rb"),
        "0123456789",
    )

    # Create annotation
    annotation = GeoDataFrame({"geometry": []}, crs="EPSG:27700")
    for p, d in zip(poi, dist):
        # Find lowest airspace at point p
        ctas = cta.cx[p.x : p.x + 1, p.y : p.y + 1]
        min_ind = ctas["normlower"].argmin()
        lowest_cta = ctas.iloc[min_ind]

        # Skip if base at surface
        if lowest_cta["normlower"] == 0:
            continue

        # Use slanted glyphs for flight levels, upright for altitude
        style = "slanted" if lowest_cta["lower"].startswith("FL") else "normal"

        # Create annotation polgons
        txt = make_string(glyphs, str(lowest_cta["normlower"] // 100), style)
        minx, miny, maxx, maxy = txt.bounds

        # Scale to fit space available
        scl = 2 * min(d, TEXT_SIZE) / sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
        txt = scale(txt, scl, scl)
        minx, miny, maxx, maxy = txt.bounds

        # Translate to correct postion
        xoff = p.x - (minx + maxx) / 2
        yoff = p.y - (miny + maxy) / 2
        txt = translate(txt, xoff, yoff)

        # Store annotation in GeoDataFrame
        data = {
            "geometry": [txt],
            "name": [lowest_cta["name"] or lowest_cta["feature_name"]],
            "lower": [lowest_cta["lower"]],
            "upper": [lowest_cta["upper"]],
        }
        annotation = pandas.concat(
            [annotation, GeoDataFrame(data, crs="EPSG:27700")], ignore_index=False
        )

    # Convert to WGS84
    annotation = annotation.to_crs("EPSG:4326")
    if args.debug_file:
        annotation.to_file(args.debug_file)

    # Convert to OpenAir and write to file
    openair = make_openair(annotation)
    with open(args.output_file, "wt") as f:
        f.write(openair)
