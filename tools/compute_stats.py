import argparse
import logging
import os
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from tqdm import tqdm

LOG = logging.getLogger(__name__)


def imread(path: Path,
           channels_first: bool = True,
           return_metadata: bool = False) -> np.ndarray:
    """Wraps rasterio open functionality to read the numpy array
       and exit the context.
    Args:
        path (Path): path to the geoTIFF image
        channels_first (bool, optional): whether to return it
                    channels first or not. Defaults to True.
    Returns:
        np.ndarray: image array
    """
    with rasterio.open(str(path), mode="r", driver="GTiff") as src:
        image = src.read()
        metadata = src.profile.copy()
    image = image if channels_first else image.transpose(1, 2, 0)
    if return_metadata:
        return image, metadata
    return image


def _extract_img_id(file_name: str):
    file_name = Path(file_name).stem
    return file_name


def _extract_ann_id(file_name: str):
    file_name = Path(file_name).stem
    return file_name[:file_name.rfind('_')]


def _dims(image: np.ndarray) -> Tuple[int, int]:
    """Returns the first two dimensions of the given array, supposed to be
    an image in channels-last configuration (width - height).
    Args:
        image (np.ndarray): array representing an image
    Returns:
        Tuple[int, int]: height and width
    """
    return image.shape[:2]


def compute_class_weights(data: List[int],
                          smoothing: float = 0.15,
                          clip: float = 10.0):
    assert smoothing >= 0 and smoothing <= 1, "Smoothing factor out of range"
    if smoothing > 0:
        # the larger the smooth factor, the bigger the quantities to sum to the remaining counts (additive smoothing)
        smoothed_maxval = max(data) * smoothing
        for i in range(len(data)):
            data[i] += smoothed_maxval
    # retrieve the (new) max value, divide by counts, round to 2 digits and clip to the given value
    # max / value allows to keep the majority class' weights to 1, while the others will be >= 1 and <= clip
    majority = max(data)
    return [
        np.clip(round(float(majority / v), ndigits=2), 0, clip) for v in data
    ]


def compute_statistics(img_dir: str, ann_dir: str, num_classes: int = 43):
    # A couple prints just to be cool
    """Computes the statistics on the current dataset.
    """
    LOG.info("Computing dataset statistics...")

    img_dir = Path(img_dir)
    ann_dir = Path(ann_dir)
    sar_paths = sorted(list(glob(str(img_dir / "*.tif"))))

    pixel_count = 0
    ch_max = None
    ch_min = None
    ch_avg = None
    ch_std = None
    class_dist = None
    # iterate on the large tiles
    LOG.info("Computing  min, max and mean...")
    for sar_path in tqdm(sar_paths):
        image_id = _extract_img_id(sar_path)
        mask_path = os.path.join(ann_dir, f'{image_id}_MAP.tif')

        # read images
        sar = imread(sar_path, channels_first=False)
        mask = imread(mask_path, channels_first=False)
        mask = mask.reshape(_dims(mask))
        assert _dims(sar) == _dims(mask), f"Shape mismatch for {image_id}"

        # initialize vectors if it's the first iteration
        channel_count = sar.shape[-1]
        if ch_max is None:
            ch_max = np.ones(channel_count) * np.finfo(np.float32).min
            ch_min = np.ones(channel_count) * np.finfo(np.float32).max
            ch_avg = np.zeros(channel_count, dtype=np.float32)
            ch_std = np.zeros(channel_count, dtype=np.float32)
            class_dist = np.zeros(num_classes - 1, dtype=np.float32)

        mask = mask.flatten()
        class_dist += np.bincount(
            mask, minlength=num_classes - 1)[:num_classes - 1]
        valid_pixels = mask != 255
        sar = sar.reshape((-1, sar.shape[-1]))[valid_pixels]
        if sar.size != 0:
            pixel_count += sar.shape[0]
            ch_max = np.maximum(ch_max, sar.max(axis=0))
            ch_min = np.minimum(ch_min, sar.min(axis=0))
            ch_avg += sar.sum(axis=0)
    ch_avg /= float(pixel_count)

    # second pass to compute standard deviation (could be approximated in a
    # single pass, but it's not accurate)
    LOG.info("Computing standard deviation...")
    valid_paths = 0
    for sar_path, mask_path in tqdm(sar_paths):
        # read images
        image_id = _extract_img_id(sar_path)
        mask_path = os.path.join(ann_dir, f'{image_id}_MAP.tif')
        sar = imread(sar_path, channels_first=False)
        mask = imread(mask_path, channels_first=False)
        mask = mask.reshape(_dims(mask))
        assert _dims(sar) == _dims(mask), f"Shape mismatch for {image_id}"
        # prepare arrays by flattening everything except channels
        img_channels = sar.shape[-1]
        valid_pixels = mask.flatten() != 255
        sar = sar.reshape((-1, img_channels))[valid_pixels]
        if sar.size != 0:
            # compute variance
            image_std = ((sar - ch_avg[:img_channels])**2).sum(axis=0) / float(
                sar.shape[0])
            ch_std += image_std
            valid_paths += 1

    # square it to compute std
    ch_std = np.sqrt(ch_std / valid_paths)
    # compute class probabilities
    class_p = class_dist.astype(float) / float(class_dist.sum())
    # print stats
    print("channel-wise max: ", ch_max)
    print("channel-wise min: ", ch_min)
    print("channel-wise avg: ", ch_avg)
    print("channel-wise std: ", ch_std)
    print("normalized avg: ", (ch_avg - ch_min) / (ch_max - ch_min))
    print("normalized std: ", ch_std / (ch_max - ch_min))
    print("class probabilities: ", class_p)
    # print("class weights: ", 1 / (class_p * (num_classes - 1)))
    print("class weights: ", compute_class_weights(class_dist))


def compute_annotations_statistics(ann_dir: str, num_classes: int = 43):
    # A couple prints just to be cool
    """Computes the statistics on the current dataset.
    """
    LOG.info("Computing dataset annotations statistics...")

    ann_dir = Path(ann_dir)
    msk_paths = sorted(list(glob(str(ann_dir / "*.tif"))))

    class_dist = None
    # iterate on the large tiles
    LOG.info("Computing  min, max and mean...")
    for mask_path in tqdm(msk_paths):

        # read images
        mask = imread(mask_path, channels_first=False)
        mask = mask.reshape(_dims(mask))

        # initialize vectors if it's the first iteration
        if class_dist is None:
            class_dist = np.zeros(num_classes - 1, dtype=np.int32)

        mask = mask.flatten()
        class_dist += np.bincount(
            mask, minlength=num_classes - 1)[:num_classes - 1]

    # compute class probabilities
    class_p = class_dist.astype(float) / float(class_dist.sum())

    # print stats
    print("class distribution:", class_dist)
    print("class probabilities: ", class_p)
    # print("class weights: ", 1 / (class_p * (num_classes - 1)))
    print("class weights: ", compute_class_weights(class_dist))


def main():
    parser = argparse.ArgumentParser(description='Data statistics')
    parser.add_argument(
        "-img_dir", type=str, help="Images directory path", default='training')
    parser.add_argument(
        "-ann_dir",
        type=str,
        help="Annotations directory path",
        default='training')
    parser.add_argument(
        "-num_classes", type=int, help="Number of classes", default=43)
    parser.add_argument('--no_img', action='store_true')

    args = parser.parse_args()
    if (args.no_img):
        compute_annotations_statistics(
            ann_dir=args.ann_dir, num_classes=args.num_classes)
    else:
        compute_statistics(
            img_dir=args.img_dir,
            ann_dir=args.ann_dir,
            num_classes=args.num_classes)


if __name__ == "__main__":
    main()
