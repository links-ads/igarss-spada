import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pyproj
import rasterio
from shapely.geometry import Point
from shapely.ops import transform
from tqdm import tqdm

COUNTRIES = ["Catalonia", "Central_Macedonia", "Corsica", "Ligury", "Piedmont"]

SOURCE_CRS = "EPSG:4326"
DEST_CRS = "EPSG:3035"

projection = pyproj.Transformer.from_crs(
    SOURCE_CRS, DEST_CRS, always_xy=True).transform


def read_tiff(path):
    with rasterio.open(path) as src:
        return src.read(), src.transform


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--s2glc',
        '-s',
        help='Input s2glc path',
        type=str,
        default="data/S2GLC/Test/Sections")
    parser.add_argument(
        '--tiles',
        '-t',
        help='Input tiles path',
        type=str,
        default="data/FuelMap/Test/Sections/ValidMasks")
    parser.add_argument(
        '--points',
        '-p',
        help='Input lucas points path',
        type=str,
        default="data/LUCAS/pts/test/tiles")

    parser.add_argument(
        '--output-folder',
        '-o',
        help='Output folder',
        type=str,
        default="data/LUCAS/pts/test/tiles_filtered")

    return parser.parse_args()


def load_lucas_points(path: Path) -> list:
    df = pd.read_csv(path)

    df = df[["POINT_ID", "X_WGS84", "Y_WGS84", "STR18"]]
    df = df[df["STR18"] != 0]
    return df.values.tolist()


def convert_lucas_pts_crs(X, Y) -> Point:
    p = Point(X, Y)
    return transform(projection, p)


def check_pixel_value_in_point(img, trsf, point):

    x, y = rasterio.transform.rowcol(trsf, point.x, point.y)
    w, h = img.shape if len(img.shape) == 2 else (img.shape[1], img.shape[2])
    if x < w and y < h and x > 0 and y > 0:
        return img[:, x, y].squeeze()
    else:
        return None


args = parse_args()
TILES_PATH = Path(args.tiles)
S2GLC_PATH = Path(args.s2glc)
LUCAS_PATH = Path(args.points)
OUT_PATH = Path(args.output_folder)
d = {"POINT_ID": [], "X_WGS84": [], "Y_WGS84": [], "STR18": []}
for country in COUNTRIES:

    for tile_path in os.listdir(TILES_PATH):
        d = {"POINT_ID": [], "X_WGS84": [], "Y_WGS84": [], "STR18": []}

        pts_path = tile_path.replace("MSK", "PTS").split(".")[0] + ".csv"
        if os.path.isfile(str(LUCAS_PATH / pts_path)):
            tile, tile_trsf = read_tiff(str(TILES_PATH / tile_path))
            # s2glc, s2glc_trsf = read_tiff(str(S2GLC_PATH / s2glc_path))
            lucas_pts = load_lucas_points(str(LUCAS_PATH / pts_path))
            for point in tqdm(lucas_pts):
                p_id, x, y, cls = point
                lucas_point = convert_lucas_pts_crs(x, y)
                # s2glc_px = check_pixel_value_in_point(s2glc, s2glc_trsf,
                #                                       lucas_point)
                tile_px = check_pixel_value_in_point(tile, tile_trsf,
                                                     lucas_point)
                # if s2glc_px is not None and s2glc_px == 1 and tile_px is not None and tile_px == 1:
                if tile_px is not None and tile_px.item() == 1:
                    d["POINT_ID"].append(int(p_id))
                    d["X_WGS84"].append(x)
                    d["Y_WGS84"].append(y)
                    d["STR18"].append(int(cls))

            filtered_pts = pd.DataFrame(d)

        filtered_pts.to_csv(str(OUT_PATH / pts_path), index=False)
