import os

import numpy as np
import rasterio
from rasterio.enums import Resampling

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadTiffImageFromFile:

    # Remove bands
    def __init__(self, remove_bands=[]):
        self.remove_bands = remove_bands

    def __call__(self, results):
        """Wraps rasterio open functionality to read the numpy array and exit the context.
        Args:
            results: dict
        Returns:
            results: dict
        """
        with rasterio.open(
                os.path.join(results['img_dir'],
                             results['img_info']['filename']),
                mode="r",
                driver="GTiff") as src:
            image = src.read(
                # out_shape=results['out_shape'], resampling=Resampling.bilinear)
                # Remove bands
                out_shape=(12, *results['out_shape'][1:]),
                resampling=Resampling.bilinear)
        results['img'] = image.transpose(1, 2, 0).astype(np.float32)
        # Remove bands
        if self.remove_bands:
            results['img'] = np.delete(
                results['img'], self.remove_bands, axis=-1)
        return results


@PIPELINES.register_module()
class LoadTiffAnnotations:

    def __call__(self, results):
        """Wraps rasterio open functionality to read the numpy array and exit the context.
        Args:
            results: dict
        Returns:
            results: dict
        """
        path = os.path.join(results['ann_dir'],
                            results['img_info']['ann']['seg_map'])
        with rasterio.open(path, mode="r", driver="GTiff") as src:
            labels = src.read(
                out_shape=(1, *results['out_shape'][1:]),
                resampling=Resampling.nearest)
        results['gt_semantic_seg'] = labels.squeeze(0).astype(np.uint8)
        return results


@PIPELINES.register_module()
class LoadTiffWeights:

    def __call__(self, results):
        """Wraps rasterio open functionality to read the numpy array and exit the context.
        Args:
            results: dict
        Returns:
            results: dict
        """
        tile = results['img_info']['ann']['seg_map'][:-8]
        path = os.path.join(results['weight_dir'], f'{tile}_DIST.tif')
        with rasterio.open(path, mode="r", driver="GTiff") as src:
            weights = src.read(
                out_shape=(1, *results['out_shape'][1:]),
                resampling=Resampling.bilinear)
        results['seg_weight'] = weights.squeeze(0).astype(np.float32)
        return results


@PIPELINES.register_module()
class LoadLUCAS:

    def __call__(self, results):
        """Wraps rasterio open functionality to read the numpy array and exit the context.
        Args:
            results: dict
        Returns:
            results: dict
        """
        tile = results['img_info']['ann']['seg_map'][:-8]
        path = os.path.join(results['lucas_dir'], f'{tile}_LUCAS.tif')
        with rasterio.open(path, mode="r", driver="GTiff") as src:
            gt_lucas = src.read(
                out_shape=(1, *results['out_shape'][1:]),
                resampling=Resampling.nearest)
        results['gt_lucas'] = gt_lucas.squeeze(0).astype(np.int64)
        return results


@PIPELINES.register_module()
class LoadDEM:

    def __call__(self, results):
        """Wraps rasterio open functionality to read the numpy array and exit the context.
        Args:
            results: dict
        Returns:
            results: dict
        """
        tile = results['img_info']['filename'][:-4]
        path = os.path.join(results['dem_dir'], f'{tile}_DEM.tif')
        with rasterio.open(path, mode="r", driver="GTiff") as src:
            dem_data = src.read(
                out_shape=(1, *results['out_shape'][1:]),
                resampling=Resampling.bilinear)
        dem_data = dem_data.transpose(1, 2, 0)
        results['img'] = np.concatenate((results['img'], dem_data), axis=2)
        return results


@PIPELINES.register_module()
class LoadTiffAnnotation:

    def __call__(self, results):
        """Wraps rasterio open functionality to read the numpy array and exit the context.
        Args:
            results: dict
        Returns:
            results: dict
        """
        path = os.path.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        with rasterio.open(path, mode="r", driver="GTiff") as src:
            labels = src.read(resampling=Resampling.nearest)
        results['gt_semantic_seg'] = labels.squeeze(0).astype(np.uint8)
        return results
