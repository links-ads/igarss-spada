import os
from pathlib import Path

import geopandas as gpd
import pyproj
import rasterio
from shapely.geometry import Point
from shapely.ops import transform
from tqdm import tqdm

SOURCE_CRS = "EPSG:4326"
DEST_CRS = "EPSG:3035"

TILES_PATH = Path("data/FuelMap/Test/Sections/Fuel")
PTS_PATH = Path("data/LUCAS/pts/test/test_disambiguated.csv")
OUT_PATH = Path("data/LUCAS/pts/test/tiles")
ignore_idx = 255
projection = pyproj.Transformer.from_crs(
    SOURCE_CRS, DEST_CRS, always_xy=True).transform


def create_point(row):
    return Point(row['X_WGS84'], row['Y_WGS84'])


def read_tiff(path):
    with rasterio.open(path) as src:
        return src.read(), src.transform


def convert_lucas_pts_crs(X, Y) -> Point:
    p = Point(X, Y)
    return transform(projection, p)


pts = gpd.read_file(PTS_PATH)
pts.crs = 'epsg:4326'
# pts['geometry'] = pts.apply(lambda row: Point(float(row['X_WGS84']), float(row['Y_WGS84'])), axis=1)
d = {"POINT_ID": [], "X_WGS84": [], "Y_WGS84": [], "STR18": [], "STR218": []}
for entry in tqdm(os.listdir(TILES_PATH)):
    pts_in_sections = gpd.GeoDataFrame(d)
    img, trsf = read_tiff(str(TILES_PATH / entry))
    w, h = img.shape if len(img.shape) == 2 else (img.shape[1], img.shape[2])
    for idx, row in pts.iterrows():
        # print(row)
        point = convert_lucas_pts_crs(
            float(row['X_WGS84']), float(row['Y_WGS84']))
        x, y = rasterio.transform.rowcol(trsf, point.x, point.y)
        if x >= 0 and y >= 0 and x < w and y < h:
            pts_in_sections = pts_in_sections.append(
                row[["POINT_ID", "X_WGS84", "Y_WGS84", "STR18", "STR218"]])

    pts_in_sections.to_csv(
        str(OUT_PATH / f"{entry.replace('MAP','PTS').split('.')[0]}.csv"),
        index=False)
    print(str(OUT_PATH / f"{entry.replace('MAP','PTS').split('.')[0]}.csv"))
