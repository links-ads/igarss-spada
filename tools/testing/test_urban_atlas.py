import argparse
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rasterio
from prettytable import PrettyTable
from tqdm import tqdm

from mmseg.core.evaluation.metrics import eval_metrics

IMGS_PATH = Path("data/FuelMap/Results")
UA_PATH = Path("data/UrbanAtlas/Tiles/Test")
NUM_CLASSES = 7


def read_tiff(path: Path) -> rasterio.DatasetReader:
    with rasterio.open(path) as src:
        return src.read()


def test_on_UA(path_test_imgs: Path, path_ua_imgs: Path, out_file):

    if not os.path.isdir(path_test_imgs):
        raise Exception("Error in data path")

    eval_results = {}
    preds = []
    labels = []
    class_names = ('Artificial', 'Bare', 'Wetlands', 'Water', 'Grassland',
                   'Agricultural', 'Forest')

    for tile in tqdm(os.listdir(path_test_imgs)):
        if "MAP" in tile:
            ua_tile = tile.replace("MAP", "UA")
        elif "S2GLC" in tile:
            ua_tile = tile.replace("S2GLC", "UA")
        if os.path.isfile(str(path_ua_imgs / ua_tile)):
            pred = read_tiff(str(path_test_imgs / tile))
            pred[pred == 7] = 6
            preds.append(pred)
            gt = read_tiff(str(path_ua_imgs / ua_tile))

            labels.append(gt)

    ret_metrics = eval_metrics(
        preds,
        labels,
        NUM_CLASSES,
        ignore_index=255,
        metrics="mIoU",
        label_map=dict(),
        reduce_zero_label=False)

    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    with open(out_file, "w") as f:
        f.write('per class results:')
        f.write('\n' + class_table_data.get_string())
        f.write('Summary:')
        f.write('\n' + summary_table_data.get_string())

    # each metric dict
    for key, value in ret_metrics_summary.items():
        if key == 'aAcc':
            eval_results[key] = value / 100.0
        else:
            eval_results['m' + key] = value / 100.0

    ret_metrics_class.pop('Class', None)
    for key, value in ret_metrics_class.items():
        eval_results.update({
            key + '.' + str(name): value[idx] / 100.0
            for idx, name in enumerate(class_names)
        })

    return ret_metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(description='test on Urban Atlas dataset')
    parser.add_argument("--out_file", "-o", default="out.txt")
    parser.add_argument(
        "--data", "-d", help="data path", default="data/FuelMap/results")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # parametrizzare sulla base del tipo di dato da testare con lucas: nostre inferenze / corine / s2glc
    IMGS_PATH = Path(args.data)

    metrics = test_on_UA(IMGS_PATH, UA_PATH, out_file=args.out_file)
