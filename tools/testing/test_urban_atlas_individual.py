import argparse
import os
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

from mmseg.core.evaluation.metrics import eval_metrics

IMGS_PATH = Path("data/FuelMap/Results")
UA_PATH = Path("data/UrbanAtlas/Tiles/Test")
NUM_CLASSES = 7


def read_tiff(path: Path) -> rasterio.DatasetReader:
    with rasterio.open(path) as src:
        return src.read()


def test_on_UA(path_test_imgs: Path, path_ua_imgs: Path):

    if not os.path.isdir(path_test_imgs):
        raise Exception("Error in data path")

    ious = []

    for ua_tile in tqdm(os.listdir(path_ua_imgs)):
        tile = ua_tile.replace("UA", "MAP")
        if os.path.isfile(str(path_ua_imgs / ua_tile)):
            pred = read_tiff(str(path_test_imgs / tile))
            pred[pred == 7] = 6
            preds = [pred]
            gt = read_tiff(str(path_ua_imgs / ua_tile))

            labels = [gt]

        if np.sum(gt == 255) / gt.size >= 0.5:
            continue

        ret_metrics = eval_metrics(
            preds,
            labels,
            NUM_CLASSES,
            ignore_index=255,
            metrics="mIoU",
            label_map=dict(),
            reduce_zero_label=False)

        ious.append((np.round(np.nanmean(ret_metrics['IoU']) * 100, 2), tile))
    ious.sort(reverse=True)
    print(ious[:5])


def parse_args():
    parser = argparse.ArgumentParser(description='test on Urban Atlas dataset')
    parser.add_argument(
        "--data", "-d", help="data path", default="data/FuelMap/Results")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # parametrizzare sulla base del tipo di dato da testare con lucas: nostre inferenze / corine / s2glc
    IMGS_PATH = Path(args.data)

    metrics = test_on_UA(IMGS_PATH, UA_PATH)
