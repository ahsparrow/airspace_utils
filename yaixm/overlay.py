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

from yaixm.convert import OPENAIR_LATLON_FMT, openair_level_str
from yaixm.geojson import geojson as convert_geojson
from yaixm.helpers import dms, load

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
            openair.append(f"AL {openair_level_str(row['lower'])}")
            openair.append(f"AH {openair_level_str(row['upper'])}")
            for vertex in poly.exterior.coords:
                latlon = OPENAIR_LATLON_FMT.format(dms(vertex[1]), dms(vertex[0]))
                openair.append(f"DP {latlon}")

    openair.append("")
    return "\n".join(openair)


# Calculate best position for annotation
def get_position(polys):
    pos = []
    dist = []
    for p in polys:
        poi = polylabel(p, tolerance=10)
        poi_dist = poi.distance(p.exterior)

        centroid = p.centroid
        centroid_dist = centroid.distance(p.exterior)

        if p.contains(centroid) and centroid_dist > min(TEXT_SIZE, (poi_dist * 0.90)):
            pos.append(centroid)
            dist.append(centroid_dist)
        else:
            pos.append(poi)
            dist.append(poi_dist)

    return pos, dist


# Split a polygon into two
def split_poly(poly, grid=500):
    # Create array of points inside polygon
    minx, miny, maxx, maxy = poly.bounds
    nx = int((maxx - minx) // grid + 1)
    ny = int((maxy - miny) // grid + 1)

    pts = MultiPoint([Point(x * grid, y * grid) for x in range(nx) for y in range(ny)])

    pts = translate(pts, minx, miny)
    pts = poly.intersection(pts)

    # Split points into two clusters using k-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    cluster = kmeans.fit_predict([(p.x, p.y) for p in pts.geoms])

    # Create new polgons from clusted points
    c1 = MultiPoint(numpy.array(pts.geoms)[cluster == 0])
    p1 = c1.buffer(grid * 1.6).intersection(poly)

    c2 = MultiPoint(numpy.array(pts.geoms)[cluster == 1])
    p2 = c2.buffer(grid * 1.6).intersection(poly)

    # Return polygon array
    return [p1, p2]


# Iteratively split polygons into smaller parts
def poly_splitter(poly, max_size):
    if minimum_bounding_radius(poly) > max_size:
        p1, p2 = split_poly(poly)
        return poly_splitter(p1, max_size) + poly_splitter(p2, max_size)
    else:
        return [poly]


def overlay(args):
    # Load airspace in geojson format
    airspace = load(args.airspace_file)
    geojson = convert_geojson(airspace["airspace"], append_seqno=False)

    # Create geopandas GeoDataFrame
    df = read_file(io.StringIO(json.dumps(geojson)))

    # Filter CTA, etc. and limit base level
    cta = df[
        df["type"].isin(["CTA", "CTR", "TMA"])
        & (df["normlower"] <= args.max_alt)
        & (df["name"] != "BRIZE NORTON CTR")
        & (
            df["rules"].isna()
            | df["rules"].apply(str).apply(lambda x: x.count("NOTAM") == 0)
        )
    ]

    # Change to X/Y projection
    cta = cta.to_crs("EPSG:27700")

    # Discard all but geometry
    cta_geom = GeoDataFrame({"geometry": cta.geometry})

    # Make airspace union
    geom = cta_geom[0:1]
    for i in range(1, len(cta_geom)):
        geom = geom.overlay(cta_geom[i : i + 1], how="union", keep_geom_type=True)

    # Remove airspace 'slivers' and convert MultiPolygons to Polygons
    polys = geom.geometry.buffer(100).buffer(-600).explode(index_parts=False)

    # Remove empty polygons
    polys = polys[~polys.is_empty]

    # Split bigger polygons
    polys = [poly_splitter(p, SPLIT_RADIUS) for p in polys]
    polys = [poly for sublist in polys for poly in sublist]

    # Get label positions nd distance to edge
    poi, dist = get_position(polys)

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
            "name": [lowest_cta["name"]],
            "lower": [lowest_cta["lower"]],
            "upper": [lowest_cta["upper"]],
        }
        annotation = pandas.concat(
            [annotation, GeoDataFrame(data, crs="EPSG:27700")], ignore_index=False
        )

    # Convert to WGS84
    annotation = annotation.to_crs("EPSG:4326")
    if args.debug_file:
        annotation.to_file(args.debug_file, driver="GeoJSON")

    # Convert to OpenAir and write to file
    openair = make_openair(annotation)
    args.output_file.write(openair)
