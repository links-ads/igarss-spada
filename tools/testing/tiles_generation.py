import os
import os.path as osp
from argparse import ArgumentParser
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyproj
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as wind_transform
from shapely.geometry.polygon import Polygon
from shapely.ops import transform
from tqdm import tqdm

SOURCE_CRS = "EPSG:3035"
DEST_CRS = "EPSG:4326"

projection = pyproj.Transformer.from_crs(
    SOURCE_CRS, DEST_CRS, always_xy=True).transform

projection2 = pyproj.Transformer.from_crs(
    DEST_CRS, SOURCE_CRS, always_xy=True).transform


def read_dataset(file_path: str, type: str = "GTiff"):
    """read dataset information from an image with rasterio

    Args:
        file_path (str): input file
        type (str, optional): image type. Defaults to "GTiff".

    Returns:
        _type_: dataset information
    """
    src = rasterio.open(file_path, mode="r", driver=type)
    return src


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--input-path',
        '-i',
        help='Input data path',
        type=str,
        default="data/ProcessedMosaics",
        dest="DATA_PATH")
    parser.add_argument(
        '--output-path',
        '-o',
        help='Output folder name',
        type=str,
        default="data/lucas_test_tiles",
        dest="OUT_DATA_PATH")
    parser.add_argument(
        '--points',
        '-p',
        help='Points csv file',
        type=str,
        default="data/lucas_test_tiles/filtered_points.csv",
        dest="FILE_POINTS")
    return parser.parse_args()


def main():
    args = parse_args()
    DATA_PATH = Path(args.DATA_PATH)
    OUT_DATA_PATH = Path(args.OUT_DATA_PATH)
    FILE_POINTS = Path(args.FILE_POINTS)
    points_df = pd.read_csv(str(FILE_POINTS))
    points = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df.X_WGS84, points_df.Y_WGS84))

    bounds = []
    files = []
    mappings = {}
    for entry in os.listdir(str(DATA_PATH)):
        if osp.isfile(str(DATA_PATH / entry)) and ".tif" in entry:
            files.append(entry)
            bounds.append(read_dataset(str(DATA_PATH / entry)).bounds)

    for i in range(len(bounds)):
        # select points
        pol_bounds = [[bounds[i].left, bounds[i].top],
                      [bounds[i].right, bounds[i].top],
                      [bounds[i].right, bounds[i].bottom],
                      [bounds[i].left, bounds[i].bottom],
                      [bounds[i].left, bounds[i].top]]
        p = Polygon(pol_bounds)
        p = transform(projection, p)
        selected = points[points.within(p)][["POINT_ID", "geometry", "STR18"]]
        mappings[files[i]] = selected.values.tolist()

    for k, v in tqdm(mappings.items()):
        with rasterio.open(
                str(DATA_PATH / k), mode="r", driver="GTiff") as src:
            for id, p, label in v:
                zone = k.split("_")[0]
                p = transform(projection2, p)
                c_y, c_x = rasterio.transform.rowcol(src.transform, p.x, p.y)

                if c_y - 256 < 0 or c_y + 256 > src.shape[
                        1] or c_x - 256 < 0 or c_x + 256 > src.shape[0]:
                    continue

                window = Window(c_x - 256, c_y - 256, 512, 512)
                image = src.read(window=window)

                count = image.shape[0]
                height = window.height
                width = window.width
                with rasterio.open(
                        str(OUT_DATA_PATH / f"{id}_{zone}_{label}.tiff"),
                        mode="w",
                        driver="GTiff",
                        height=height,
                        width=width,
                        count=count,
                        transform=wind_transform(window, src.transform),
                        dtype=src.dtypes[0],
                        crs=src.crs) as out:
                    out.write(image)


if __name__ == "__main__":
    main()
