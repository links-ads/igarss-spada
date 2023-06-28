from typing import Callable, List, Tuple

import numpy as np
import rasterio
import scipy.signal
import torch

WINDOW_CACHE = dict()

# log = rasterio.logging.getLogger()
# log.setLevel(rasterio.logging.ERROR)


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


def _spline_window(window_size: int, power: int = 2) -> np.ndarray:
    """Generates a 1-dimensional spline of order 'power' (typically 2), in the designated
    window.
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2

    Args:
        window_size (int): size of the interested window
        power (int, optional): Order of the spline. Defaults to 2.

    Returns:
        np.ndarray: 1D spline
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.triang(window_size)))**power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 *
                          (scipy.signal.triang(window_size) - 1))**power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def _spline_2d(window_size: int, power: int = 2) -> np.ndarray:
    """Makes a 1D window spline function, then combines it to return a 2D window function.
    The 2D window is useful to smoothly interpolate between patches.

    Args:
        window_size (int): size of the window (patch)
        power (int, optional): Which order for the spline. Defaults to 2.

    Returns:
        np.ndarray: numpy array containing a 2D spline function
    """
    # Memorization to avoid remaking it for every call
    # since the same window is needed multiple times
    global WINDOW_CACHE
    key = f"{window_size}_{power}"
    if key in WINDOW_CACHE:
        wind = WINDOW_CACHE[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1),
                              1)  # SREENI: Changed from 3, 3, to 1, 1
        wind = wind * wind.transpose(1, 0, 2)
        WINDOW_CACHE[key] = wind
    return wind


def windowed_generator(padded_image: np.ndarray,
                       window_size: int,
                       subdivisions: int,
                       batch_size: int = None):
    """Generator that yield tiles grouped by batch size.

    Args:
        padded_image (np.ndarray): input image to be processed (already padded), supposed channels-first
        window_size (int): size of a single patch
        subdivisions (int): subdivision count on each patch to compute the step
        batch_size (int, optional): amount of patches in each batch. Defaults to None.

    Yields:
        Tuple[List[tuple], np.ndarray]: list of coordinates and respective patches as single batch array
    """
    step = window_size // subdivisions
    width, height, _ = padded_image.shape
    batch_size = batch_size or 1

    batch = []
    coords = []
    # step with fixed window on the image to build up the arrays
    for x in range(0, width - window_size + 1, step):
        for y in range(0, height - window_size + 1, step):
            coords.append((x, y))
            # extract the tile, place channels first for batch
            tile = padded_image[x:x + window_size, y:y + window_size]
            batch.append(np.transpose(tile, axes=[2, 0, 1]))
            # yield the batch once full and restore lists right after
            if len(batch) == batch_size:
                yield coords, np.stack(batch)
                coords = []
                batch = []
    # handle last (possibly unfinished) batch
    if len(batch) > 0:
        yield coords, np.stack(batch)


def compute_pad(image: np.ndarray, tile_size: int, subdivisions: int):
    """Compute padding

    Args:
        image (np.ndarray): input image
        tile_size (int): tile size
        subdivisions (int): subdivisions

    Returns:
        _type_: _description_
    """
    # compute the pad as (window - window/subdivisions)
    pad = int(round(tile_size * (1 - 1.0 / subdivisions)))

    width, height, _ = image.shape

    width_start_pad = pad
    height_start_pad = pad
    width_end_pad = pad
    height_end_pad = pad

    if width % pad != 0:
        width_end_pad = 2 * pad
    if height % pad != 0:
        height_end_pad = 2 * pad
    return (width_start_pad, width_end_pad, height_start_pad, height_end_pad)


def pad_image(image: np.ndarray, tile_size: int,
              subdivisions: int) -> np.ndarray:
    """Add borders to the given image for a "valid" border pattern according to "window_size" and "subdivisions".
    Image is expected as a numpy array with shape (width, height, channels).

    Args:
        image (torch.Tensor): input image, 3D channels-last tensor
        tile_size (int): size of a single patch, useful to compute padding
        subdivisions (int): amount of overlap, useful for padding

    Returns:
        torch.Tensor: same image, padded specularly by a certain amount in every direction
    """
    # compute the pad as (window - window/subdivisions)
    # pad = int(round(tile_size * (1 - 1.0 / subdivisions)))
    width_start_pad, width_end_pad, height_start_pad, height_end_pad = compute_pad(
        image, tile_size, subdivisions)
    # add pad pixels in height and width, nothing channel-wise
    image = np.pad(
        image, ((width_start_pad, width_end_pad),
                (height_start_pad, height_end_pad), (0, 0)),
        mode="reflect")
    return (width_start_pad, width_end_pad, height_start_pad,
            height_end_pad), image


def unpad_image(padded_image: np.ndarray, padding: Tuple[int, int, int,
                                                         int]) -> np.ndarray:
    """Reverts changes made by 'pad_image'. The same padding is removed, so tile_size and subdivisions
    must be coherent.

    Args:
        padded_image (torch.Tensor): image with padding still applied
        tile_size (int): size of a single patch
        subdivisions (int): subdivisions to compute overlap

    Returns:
        torch.Tensor: image without padding, 2D channels-last tensor
    """
    # compute the same amount as before, window - window/subdivisions
    # pad = int(round(tile_size * (1 - 1.0 / subdivisions)))
    # width_start_pad, width_end_pad, height_start_pad, height_end_pad = compute_pad(image, tile_size, subdivisions)
    # crop the image left, right, top and bottom
    result = padded_image[padding[0]:-padding[1], padding[2]:-padding[3]]
    return result


def reconstruct(canvas: np.ndarray,
                tile_size: int,
                coords: List[tuple],
                predictions: np.ndarray,
                use_logits: bool = True) -> np.ndarray:
    """Helper function that iterates the result batch onto the given canvas to reconstruct
    the final result batch after batch.

    Args:
        canvas (torch.Tensor): container for the final image.
        tile_size (int): size of a single patch.
        coords (List[tuple]): list of pixel coordinates corresponding to the batch items
        predictions (torch.Tensor): array containing patch predictions, shape (batch, tile_size, tile_size, num_classes)

    Returns:
        torch.Tensor: the updated canvas, shape (padded_w, padded_h, num_classes)
    """
    predictions = np.asarray(predictions)
    for (x, y), patch in zip(coords, predictions):
        if use_logits:
            patch = np.transpose(patch, axes=[1, 2, 0])
            canvas[x:x + tile_size, y:y + tile_size, :] += patch
        else:
            canvas[x:x + tile_size, y:y + tile_size] += patch
    return canvas


def create_subimages_for_inference(
        image: np.ndarray,
        tile_size: int,
        subdivisions: int,
        batch_size: int = None,
        channels_first: bool = False) -> Tuple[List[tuple], np.ndarray]:
    """split a big input image in tiles

    Args:
        image (torch.Tensor): input image, expected a 3D vector
        tile_size (int): size of each squared tile
        subdivisions (int): number of subdivisions over the single tile for overlaps
        batch_size (int, optional): size of each batch. Defaults to None.
        channels_first (bool, optional): whether the input image is channels-first or not. Defaults to False.

    Returns:
        Tuple[torch.Tensor, Tuple[List[tuple], np.ndarray]: padded image canvas + list of coordinates and respective patches as single batch array
    """

    if channels_first:
        image = np.transpose(image, axes=[1, 2, 0])
    padding, padded = pad_image(
        image=image, tile_size=tile_size, subdivisions=subdivisions)
    padded_width, padded_height, _ = padded.shape
    canvas = np.zeros((padded_width, padded_height, 11))

    return (canvas, padding,
            windowed_generator(
                padded_image=padded,
                window_size=tile_size,
                subdivisions=subdivisions,
                batch_size=batch_size))


def rotate_and_mirror(image: torch.Tensor) -> List[torch.Tensor]:
    """Duplicates an image with shape (h, w, channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations. https://en.wikipedia.org/wiki/Dihedral_group

    Args:
        image (torch.Tensor): input image, already padded.

    Returns:
        List[torch.Tensor]: list of images, rotated and mirrored.
    """
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    variants = []
    variants.append(image)
    variants.append(torch.rot90(image, k=1, dims=(0, 1)))
    variants.append(torch.rot90(image, k=2, dims=(0, 1)))
    variants.append(torch.rot90(image, k=3, dims=(0, 1)))
    image = torch.flip(image, dims=(0, 1))
    variants.append(image)
    variants.append(torch.rot90(image, k=1, dims=(0, 1)))
    variants.append(torch.rot90(image, k=2, dims=(0, 1)))
    variants.append(torch.rot90(image, k=3, dims=(0, 1)))
    return variants


def undo_rotate_and_mirror(variants: List[torch.Tensor]) -> torch.Tensor:
    """Reverts the 8 duplications provided by rotate and mirror.
    This restores the transformed inputs to the original position, then averages them.

    Args:
        variants (List[torch.Tensor]): D4 dihedral group of the same image

    Returns:
        torch.Tensor: averaged result over the given input.
    """
    origs = []
    origs.append(variants[0])
    origs.append(torch.rot90(variants[1], k=3, dims=(0, 1)))
    origs.append(torch.rot90(variants[2], k=2, dims=(0, 1)))
    origs.append(torch.rot90(variants[3], k=1, dims=(0, 1)))

    origs.append(torch.flip(variants[4], dims=(0, 1)))
    origs.append(
        torch.flip(torch.rot90(variants[5], k=3, dims=(0, 1)), dims=(0, 1)))
    origs.append(
        torch.flip(torch.rot90(variants[6], k=2, dims=(0, 1)), dims=(0, 1)))
    origs.append(
        torch.flip(torch.rot90(variants[7], k=1, dims=(0, 1)), dims=(0, 1)))
    return torch.mean(torch.stack(origs), axis=0)


def predict_smooth_windowing(image: torch.Tensor,
                             img_metas: dict,
                             tile_size: int,
                             subdivisions: int,
                             prediction_fn: Callable,
                             batch_size: int = None,
                             channels_first: bool = False,
                             mirrored: bool = False,
                             use_logits: bool = True) -> np.ndarray:
    """Allows to predict a large image in one go, dividing it in squared, fixed-size tiles and smoothly
    interpolating over them to produce a single, coherent output with the same dimensions.

    Args:
        image (np.ndarray): input image, expected a 3D vector
        tile_size (int): size of each squared tile
        subdivisions (int): number of subdivisions over the single tile for overlaps
        prediction_fn (Callable): callback that takes the input batch and returns an output tensor
        batch_size (int, optional): size of each batch. Defaults to None.
        channels_first (int, optional): whether the input image is channels-first or not
        mirrored (bool, optional): whether to use dihedral predictions (every simmetry). Defaults to False.

    Returns:
        np.ndarray: numpy array with dimensions (w, h), containing smooth predictions
    """
    if channels_first:
        image = image.permute(1, 2, 0)
    width, height, _ = image.shape
    padding, padded = pad_image(
        image=image, tile_size=tile_size, subdivisions=subdivisions)
    padded_width, padded_height, _ = padded.shape
    padded_variants = rotate_and_mirror(padded) if mirrored else [padded]
    spline = _spline_2d(window_size=tile_size, power=2).squeeze(-1)

    results = []
    for img in padded_variants:
        if use_logits:
            canvas = torch.zeros((padded_width, padded_height, 11),
                                 device=image.device)
        else:
            canvas = torch.zeros((padded_width, padded_height),
                                 device=image.device)
        for coords, batch in windowed_generator(
                padded_image=img,
                window_size=tile_size,
                subdivisions=subdivisions,
                batch_size=batch_size):
            # returns batch of channels-first, return to channels-last
            pred_batch = prediction_fn(
                return_loss=False,
                img_metas=img_metas,
                img=[torch.tensor(batch)],
                return_logits=use_logits)
            if isinstance(pred_batch, torch.Tensor):
                pred_batch = pred_batch.detach().cpu().numpy()

            pred_batch = [tile * spline for tile in pred_batch]
            canvas = reconstruct(
                canvas,
                tile_size=tile_size,
                coords=coords,
                predictions=pred_batch,
                use_logits=use_logits)
        canvas /= (subdivisions**2)
        results.append(canvas)

    padded_result = undo_rotate_and_mirror(results) if mirrored else results[0]
    prediction = unpad_image(padded_result, padding)
    if use_logits:
        return prediction[:width, :height, :]
    else:
        return prediction[:width, :height]
