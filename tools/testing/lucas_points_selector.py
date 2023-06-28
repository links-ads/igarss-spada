import json
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pyproj
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import transform
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--input-crs',
        '-c',
        help='Input CRS format epsg:XXXX',
        type=str,
        default="epsg:4326",
        dest="SRC_CRS")

    parser.add_argument(
        '--input-file',
        '-i',
        help='Input lucas csv file',
        type=str,
        default="data/LUCAS/GRID_CSVEXP_20171113.csv",
        dest="PATH")
    parser.add_argument(
        '--aoi',
        '-a',
        help='AOIs geometry json file',
        type=str,
        default="scripts/aois.json",
        dest="AOIS_FILE")
    parser.add_argument(
        '--output-folder',
        '-o',
        help='Output folder',
        type=str,
        default="scripts",
        dest="OUT_DATA_FOLDER")
    parser.add_argument(
        '--output-file',
        '-f',
        help='Output file',
        type=str,
        default="out.csv",
        dest="OUT_DATA_FILE")
    parser.add_argument(
        '--separate-out',
        '-s',
        help='if store points in separate files for each aoi',
        action="store_true",
        dest="SEPARATE_OUT")

    return parser.parse_args()


def main():
    args = parse_args()
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init=args.SRC_CRS),  # source coordinate system
        pyproj.Proj(init='epsg:4326'))  # destination coordinate system

    PATH = Path(args.PATH)
    AOIS_FILE = Path(args.AOIS_FILE)
    OUT_DATA_FOLDER = Path(args.OUT_DATA_FOLDER)
    if not os.path.isdir(OUT_DATA_FOLDER):
        os.makedirs(OUT_DATA_FOLDER)
    OUT_DATA_FILE = Path(args.OUT_DATA_FILE)
    points = pd.read_csv(PATH)
    points.head()

    with open(AOIS_FILE, "r") as fp:
        aois = json.load(fp)

    # for each aoi, select the points in the intersection of the aoi

    filtered_points = {}
    filtered_points = []
    for name, aoi in tqdm(aois.items()):
        print(name, aoi)
        g = transform(project.transform,
                      Polygon(aoi["geometry"]["coordinates"]))
        for index, row in points.iterrows():
            p = Point([row["X_WGS84"], row["Y_WGS84"]])
            if p.within(g):
                filtered_points.append(row["POINT_ID"])

        if args.SEPARATE_OUT:
            fpoints = points[points["POINT_ID"].isin(filtered_points)]
            fpoints.to_csv(
                str(OUT_DATA_FOLDER / f"{name}_{OUT_DATA_FILE}"),
                columns=["POINT_ID", "X_WGS84", "Y_WGS84", "STR18", "STR218"],
                index=False)
            filtered_points = []

    if not args.SEPARATE_OUT:
        fpoints = points[points["POINT_ID"].isin(filtered_points)]
        fpoints.to_csv(
            str(OUT_DATA_FOLDER / f"{OUT_DATA_FILE}"),
            columns=["POINT_ID", "X_WGS84", "Y_WGS84", "STR18", "STR218"],
            index=False)


if __name__ == "__main__":
    main()
