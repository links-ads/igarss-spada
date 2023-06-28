import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
from shapely.geometry.point import Point
from shapely.ops import transform
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tabulate import tabulate
from tqdm import tqdm

IMGS_PATH = Path("data/FuelMap/results")
PTS_PATH = Path("data/LUCAS/filtered.csv")
MAPPING_PATH = Path("data/LUCAS/mapping_fuel_to_lucas.json")
SEQ_MAPPING_PATH = Path("data/LUCAS/mapping_seq_to_lucas.json")
SOURCE_CRS = "EPSG:4326"
DEST_CRS = "EPSG:3035"

LUCAS_TO_SEQ = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 7}
LUCAS_TO_SEQ_DISAMBIGUATED = {
    1: 1,
    2: 1,
    3: 2,
    44: 255,
    46: 3,
    47: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 8
}
CLASS_DESC = {
    255: "Ignored",
    12: "1+2: Arable land & Permanent crops",
    3: "3: Grass",
    4: "4: Wooded areas",
    5: "5: Shrubs",
    6: "6: Bare surface, rare or low vegetation",
    7: "7: Artificial, construction and sealed areas",
    89: "8+9: Inland water & Transitional and coastal waters",
    44: "4: Wooded areas",
    46: "4.6: Broadleaf",
    47: "4.7: Coniferous",
}
projection = pyproj.Transformer.from_crs(
    SOURCE_CRS, DEST_CRS, always_xy=True).transform

# 1 -   read and merge the test mosaics

# 2 -   for each LUCAS point, take the 2x2,4x4,6x6,8x8,10x10 bboxes and do a
#       majority voting on the fuel class,
#       then compare to the corresponding lucas class


def load_json(path: Path) -> dict:
    with open(path) as fi:
        mapping = json.load(fi)
    return mapping


def load_lucas_points(path: Path, files=None) -> list:
    if files is None:
        df = pd.read_csv(path)

        df = df[["X_WGS84", "Y_WGS84", "STR18"]]
        df = df[df["STR18"] != 0]
        return df.values.tolist()
    else:
        points = []
        for f in files:
            if "MAP" in f:
                f = f.replace("MAP", "PTS")
            elif "S2GLC" in f:
                f = f.replace("S2GLC", "PTS")
            name = str(path / f"{f.split('.')[0]}.csv")
            if os.path.isfile(name):
                df = pd.read_csv(name)
                df = df[["X_WGS84", "Y_WGS84", "STR18"]]
                df = df[df["STR18"] != 0]
                points.extend(df.values.tolist())
        return points


def read_tiff(path: Path) -> rasterio.DatasetReader:
    return rasterio.open(path)


def merge_test_imgs(data_path: Path, filter: str = "") -> tuple:
    files = []
    imgs = []
    if os.path.isdir(data_path):
        for entry in os.listdir(data_path):
            if entry.startswith(filter) and entry.endswith(".tif"):
                files.append(entry)
                imgs.append(read_tiff(Path(data_path / entry)))
        out, trs = merge(imgs)
    else:
        src = read_tiff(Path(data_path))
        return src, src.transform, [data_path]

    return out, trs, files


def convert_lucas_pts_crs(X, Y) -> Point:
    p = Point(X, Y)
    return transform(projection, p)


def convert_classes(data, mapping):
    data = data.flatten()
    for i, d in enumerate(data):
        data[i] = mapping[str(d)]
    return data


def collect_metrics(cls_ids, tile_size, processed_points, ignored_points,
                    accuracy, precision, recall, f1, m_precision, m_recall,
                    m_f1, map_seq_to_lucas, out_file):

    data = {}
    for id, cl_id in enumerate(cls_ids):
        data[map_seq_to_lucas[str(cl_id)]] = dict(
            precision=precision[id],
            recall=recall[id],
            f1=f1[id],
            accuracy=accuracy,
            m_precision=m_precision,
            m_recall=m_recall,
            m_f1=m_f1)

    rows = []
    for k, v in data.items():
        rows.append([k, v['precision'], v['recall'], v['f1']])
    rows.append(['Average', m_precision, m_recall, m_f1])
    print(f"TILE SIZE: {(2 * tile_size) + 1}x{(2 * tile_size) + 1}")
    tab = tabulate([*rows],
                   headers=['Class', 'Prec', 'Rec', 'F1'],
                   tablefmt='orgtbl')
    print(tab)
    print(
        f"Accuracy: {v['accuracy']}\nMean Precision per class:{v['m_precision']}\nMean Recall per class:{v['m_recall']}\nMean F1: {v['m_f1']}"
    )
    with open(out_file, "a") as f:
        f.write(
            f"######## TILE SIZE: {(2 * tile_size) + 1}x{(2 * tile_size) + 1} ########\n"
        )

        f.write(tab)
        f.write(
            f"\nAccuracy: {v['accuracy']}\nMean Precision:{v['m_precision']}\nMean Recall:{v['m_recall']}\nMean F1: {v['m_f1']}"
        )
        f.write(
            f"\nProcessed points: {processed_points}\nIgnored points:{ignored_points}\n"
        )
        f.write("\n\n")
    return data


def compute_class_score(trues, preds, cls_ids, ignored_points, map, map_trs,
                        point, point_cls, cls_mapping, tile_size):
    x, y = rasterio.transform.rowcol(map_trs, point.x, point.y)
    w, h = map.shape if len(map.shape) == 2 else (map.shape[1], map.shape[2])
    if x >= w or y >= h or x < 0 or y < 0:
        return trues, preds, cls_ids, ignored_points
    tl_coord = (max(0, x - tile_size), max(0, y - tile_size))
    br_coord = (min(w - 1, x + tile_size), min(h - 1, y + tile_size))
    if isinstance(map, rasterio.DatasetReader):
        if tile_size != 0:
            data = map.read(
                1,
                window=Window.from_slices((tl_coord[0], br_coord[0]),
                                          (tl_coord[1], br_coord[1])))
        else:
            data = map.read(1, window=Window(tl_coord[1], tl_coord[0], 1, 1))
    else:
        if tile_size != 0:
            data = map[:, tl_coord[0]:br_coord[0] + 1,
                       tl_coord[1]:br_coord[1] + 1].squeeze(0)
        else:
            data = map[:, x, y].squeeze(0)

    data = convert_classes(data, cls_mapping)
    classes, counts = np.unique(data, return_counts=True)
    counts = counts[classes != 255]
    classes = classes[classes != 255]
    counts = counts[classes != 0]
    classes = classes[classes != 0]
    if len(counts) == 0:
        return trues, preds, cls_ids, ignored_points + 1
    maj_idx = np.argmax(counts)

    maj_cls = classes[maj_idx]
    if maj_cls not in (0, 255) and point_cls not in (0, 255):
        trues.append(point_cls)
        preds.append(maj_cls)
        if point_cls not in cls_ids:
            cls_ids.append(point_cls)
        if maj_cls not in cls_ids:
            cls_ids.append(maj_cls)
    else:
        ignored_points = ignored_points + 1
    return trues, preds, cls_ids, ignored_points
    # CM[point_cls][maj_cls] += 1
    # return CM
    # if maj_cls == point_cls:
    #     true_positives[point_cls] += 1
    # else:
    #     false_positives[maj_cls] += 1
    #     # for id in range(0, len(false_positives)):
    #     #     if id != point_cls:
    #     #         false_positives[id] += 1
    # total[point_cls] += 1
    # return true_positives, false_positives, total


def test_on_lucas(path_test_imgs: Path,
                  countries: list,
                  path_lucas_points: Path,
                  path_cls_mapping: Path,
                  path_cls_seq_mapping: Path,
                  out_file,
                  tile_size: int = 2,
                  filter_country=False):

    cls_score = {}
    for id in range(0, 10):
        cls_score[id] = 0.0

    map_cls = load_json(path_cls_mapping)
    map_seq_cls = load_json(path_cls_seq_mapping)

    if not filter_country:
        lucas_pts = load_lucas_points(path_lucas_points)

    if not os.path.isdir(path_test_imgs) and not str(path_test_imgs).endswith(
            ".tif"):
        raise Exception("Error in data path")

    metrics = []

    for tile_size in tqdm(tile_sizes):
        trues = []
        preds = []
        cls_ids = []
        ignored_points = 0
        processed_points = 0
        # CM = np.zeros((len(map_seq_cls), len(map_seq_cls)))
        # true_positives = np.zeros(len(map_seq_cls))
        # false_positives = np.zeros(len(map_seq_cls))
        # total = np.zeros(len(map_seq_cls))
        if not filter_country:
            merged_imgs, merged_trsf, _ = merge_test_imgs(path_test_imgs)
            for x, y, cls in tqdm(lucas_pts):
                processed_points = processed_points + 1
                lucas_point = convert_lucas_pts_crs(x, y)
                trues, preds, cls_ids = compute_class_score(
                    trues, preds, cls_ids, merged_imgs, merged_trsf,
                    lucas_point, LUCAS_TO_SEQ[int(cls)], map_cls, tile_size)
        else:
            for country in countries:
                merged_imgs, merged_trsf, files = merge_test_imgs(
                    path_test_imgs, filter=country)
                lucas_pts = load_lucas_points(path_lucas_points, files)
                for x, y, cls in tqdm(lucas_pts):
                    processed_points = processed_points + 1
                    lucas_point = convert_lucas_pts_crs(x, y)
                    trues, preds, cls_ids, ignored_points = compute_class_score(
                        trues, preds, cls_ids, ignored_points, merged_imgs,
                        merged_trsf, lucas_point, LUCAS_TO_SEQ[int(cls)],
                        map_cls, tile_size)
        cls_ids.sort()
        precision = precision_score(trues, preds, average=None)
        precision_avg = precision_score(trues, preds, average="weighted")
        recall = recall_score(trues, preds, average=None)
        recall_avg = recall_score(trues, preds, average="weighted")
        accuracy = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average=None)
        f1_avg = f1_score(trues, preds, average="weighted")
        print(
            f"Processed points: {processed_points}\nIgnored points: {ignored_points}"
        )
        metrics.append(
            dict(
                tile_size=tile_size,
                metrics=collect_metrics(cls_ids, tile_size, processed_points,
                                        ignored_points, accuracy, precision,
                                        recall, f1, precision_avg, recall_avg,
                                        f1_avg, map_seq_cls, out_file)))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='test on LUCAS dataset')
    parser.add_argument('--inference', '-i', help='test', action='store_true')
    parser.add_argument('--corine', '-c', help='corine', action='store_true')
    parser.add_argument('--s2glc', '-s', help='s2glc', action='store_true')
    parser.add_argument(
        '--filter',
        '-f',
        help='enable filtering by country',
        action='store_true')
    parser.add_argument(
        '--disambiguated',
        '-a',
        help='enable forest disambiguation',
        action='store_true')
    parser.add_argument('--tile_size', default=1)
    parser.add_argument('--tile_step', default=5)
    parser.add_argument("--out_file", "-o", default="out.txt")

    parser.add_argument(
        "--data", "-d", help="data path", default="data/FuelMap/results")
    parser.add_argument(
        "--points",
        "-p",
        help="lucas points csv path",
        default="data/LUCAS/filtered.csv")
    parser.add_argument(
        "--mapping",
        "-m",
        help="class mapping",
        default="data/LUCAS/mapping_fuel_to_lucas.json")
    parser.add_argument(
        "--sequentialmapping",
        "-q",
        help="sequential lucas mapping",
        default="data/LUCAS/mapping_seq_to_lucas.json")
    parser.add_argument(
        "--countries",
        type=str,
        nargs='+',
        help="countries to test",
        default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    tile_sizes = [0]
    for i in range(0, int(args.tile_step)):
        tile_sizes.append(i + int(args.tile_size))
    # parametrizzare sulla base del tipo di dato da testare con lucas: nostre inferenze / corine / s2glc
    IMGS_PATH = Path(args.data)
    PTS_PATH = Path(args.points)
    MAPPING_PATH = Path(args.mapping)
    SEQ_MAPPING_PATH = Path(args.sequentialmapping)
    if args.disambiguated:
        LUCAS_TO_SEQ = LUCAS_TO_SEQ_DISAMBIGUATED
    if args.countries is None:
        COUNTRIES = ["Central_Macedonia", "Ligury", "Sardinia", "Valencia"]
    else:
        COUNTRIES = args.countries
    metrics = test_on_lucas(
        IMGS_PATH,
        COUNTRIES,
        PTS_PATH,
        MAPPING_PATH,
        SEQ_MAPPING_PATH,
        out_file=args.out_file,
        tile_size=tile_sizes,
        filter_country=args.filter)
