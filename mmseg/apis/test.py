# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path

import mmcv
import numpy as np
import rasterio
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from scripts.tiling import (_spline_2d, create_subimages_for_inference,
                            predict_smooth_windowing, reconstruct, unpad_image)


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor[:, 1:4])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape']
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                filename = f"{Path(img_meta['ori_filename']).stem}_MAP.tif"

                if out_dir:
                    out_file = osp.join(out_dir, filename)
                else:
                    out_file = None

                write_image(
                    np.array(result, dtype=np.uint8),
                    out_file,
                    )

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)

        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def single_gpu_test_big_tiles(model,
                              data_loader,
                              show=False,
                              out_dir=None,
                              efficient_test=False,
                              opacity=0.5,
                              pre_eval=False,
                              format_only=False,
                              format_args={},
                              save_rgb=False,
                              use_logits=True):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    subdivisions = 2
    tile_size = 512

    # spline = _spline_2d(window_size=tile_size, power=2).squeeze(-1)
    for batch_indices, data in zip(loader_indices, data_loader):
        image_metas = deepcopy(data["img_metas"])
        image_metas[0].data[0][0]['img_shape'] = (tile_size, tile_size)
        image_metas[0].data[0][0]['pad_shape'] = (tile_size, tile_size)
        image_metas[0].data[0][0]['ori_shape'] = (tile_size, tile_size)
        image = data["img"][0].squeeze()
        image = np.transpose(image, axes=[0, 2, 1])
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        result = predict_smooth_windowing(
            image=image,
            img_metas=image_metas,
            tile_size=tile_size,
            subdivisions=subdivisions,
            batch_size=1,
            mirrored=True,
            prediction_fn=model,
            channels_first=True,
            use_logits=use_logits)

        # canvas, padding, subimages = create_subimages_for_inference(
        #     image.squeeze(),
        #     tile_size=tile_size,
        #     subdivisions=subdivisions,
        #     batch_size=len(batch_indices),
        #     channels_first=True)

        # for indexes, img in subimages:
        #     with torch.no_grad():

        #         pred = model(
        #             return_loss=False,
        #             img_metas=image_metas,
        #             img=[torch.tensor(img)])
        #         pred = pred * spline
        #         canvas = reconstruct(
        #             canvas,
        #             tile_size=tile_size,
        #             coords=indexes,
        #             predictions=pred)
        # canvas /= (subdivisions**2)
        # result = unpad_image(canvas, padding)
        # reset shape in order to have height, width
        if use_logits:
            result = torch.tensor(result).argmax(dim=-1).numpy()
        result = np.transpose(result, axes=[1, 0])
        result = np.rint(result)
        image_metas[0].data[0][0]['img_shape'] = result.shape
        image_metas[0].data[0][0]['pad_shape'] = (0, 0)
        image_metas[0].data[0][0]['ori_shape'] = result.shape

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            # print(img_tensor[:, 1:3].shape)
            imgs = tensor2imgs(img_tensor[:, 1:4])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape']
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                save_test_image(
                    img_meta['ori_filename'],
                    img_show,
                    result,
                    classes=model.module.CLASSES,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity,
                    save_rgb=save_rgb)
                # model.module.show_result(
                #     img_show,
                #     result,
                #     palette=dataset.PALETTE,
                #     show=show,
                #     out_file=out_file,
                #     opacity=opacity)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def read_image(file_path: str, type: str = "GTiff"):
    """read image with rasterio

    Args:
        file_path (str): file path
        type (str, optional): image type. Defaults to "GTiff".

    Returns:
        _type_: image
    """
    with rasterio.open(file_path, mode="r", driver=type) as src:
        image = src.read()
    return image


def read_metadata_from_image(file_path: str, type: str = "GTiff"):
    """read the metadata (profile) of an image with rasterio

    Args:
        file_path (str): input file
        type (str, optional): image type. Defaults to "GTiff".

    Returns:
        _type_: metadata
    """
    with rasterio.open(file_path, mode="r", driver=type) as src:
        profile = src.profile
    return profile.data


def write_image(image: np.ndarray,
                file_path: str,
                type: str = "GTiff",
                metadata: dict = None):
    """Save image to file

    Args:
        image (np.ndarray): input image
        file_path (str): output file path
        type (str, optional): image type. Defaults to "GTiff".
        metadata (dict, optional): dict with crs and transform metadata. Defaults to None.
    """
    count = image.shape[0]
    height = image.shape[1]
    width = image.shape[2]
    crs = None
    transform = None

    if metadata is not None and "crs" in metadata.keys():
        crs = metadata["crs"]
    if metadata is not None and "transform" in metadata.keys():
        transform = metadata["transform"]

    with rasterio.open(
            file_path,
            mode="w",
            driver=type,
            height=height,
            width=width,
            count=count,
            dtype=image.dtype,
            crs=crs,
            transform=transform) as src:
        src.write(image)


def save_test_image(input_file,
                    img,
                    result,
                    classes,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5,
                    save_rgb=True):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        palette (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. If None is given, random palette will be
            generated. Default: None
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    # TODO: full path is hardcoded, not so good
    meta = read_metadata_from_image(
        f"data/FuelMap/Test/Sections/Img/{input_file}")
    img = img.copy()
    if len(result.shape) > 2:
        seg = result[0]
    else:
        seg = result
    if save_rgb:
        if palette is None:

            # Get random state before set seed,
            # and restore random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            palette = np.random.randint(0, 255, size=(len(classes), 3))
            np.random.set_state(state)

        palette = np.array(palette)
        assert palette.shape[0] == len(classes)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
    else:
        img = np.expand_dims(seg, axis=-1)
    img = img.astype(np.uint8)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        #print(img.shape)

        write_image(
            np.transpose(img[..., ::-1], axes=[2, 0, 1]),
            out_file,
            metadata=meta)

    if not (show or out_file):
        warnings.warn('show==False and out_file is not specified, only '
                      'result image will be returned')
        return img


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
