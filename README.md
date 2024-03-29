# SPADA
Dataset and code for the paper *Land Cover Segmentation with Sparse Annotations from Sentinel-2 Imagery* (IGARSS 2023).

[![arXiv](https://img.shields.io/badge/arXiv-2306.16252-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2306.16252)

> [!NOTE]  
> Dataset available at [hf.co/datasets/links-ads/spada-dataset](https://huggingface.co/datasets/links-ads/spada-dataset).

---------------

![SPADA architecture](/assets/SPADA.png)

![Qualitative example](/assets/example.jpg)


## Bring me to the important bits

The main components of the training loop are located in:
- [custom_dacs.py](/mmseg/models/custom_dacs.py)

## Installation

First, create a python environment. Here we used `python 3.9` and `torch 1.9`, with `CUDA 11.1`.
We suggest creating a python environment, using `venv` or `conda` first.

The repository is based on [`mmsegmentation`](https://github.com/open-mmlab/mmsegmentation). Follow their installation instructions, or launch the following commands:

```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install -e .
```

## Training

Once the dataset is downloaded, create a soft link inside the `data` directory, it should be named `FuelMap`. You can use a custom name if you prefer, in that case however you will need to update the configuration files.

Once ready, you can launch a training with the following commands:

```console
$ CUDA_VISIBLE_DEVICES=... python tools/train.py [CONFIG_PATH]
```

For instance:
```console
$ CUDA_VISIBLE_DEVICES=... python tools/train.py configs/landcover/da_segformer_b5_lucas_dacs_v2.py
```

## Inference

To produce inference maps, run something like the following:

```
$ CUDA_VISIBLE_DEVICES=... python tools/test.py <path_to_workdir_config.py> <path to the checkpoint> --show-dir <output path> [--eval mIoU]
```

If you have large images, use the [prepare_mosaic.ipynb](/tools/prepare_mosaic.ipynb) to tile the large files into 2048x2048 tiles.

## Citing this work
```bibtex
@inproceedings{galatola2023land,
  title={Land Cover Segmentation with Sparse Annotations from Sentinel-2 Imagery},
  author={Galatola, Marco and Arnaudo, Edoardo and Barco, Luca and Rossi, Claudio and Dominici, Fabrizio},
  booktitle={IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium},
  pages={6952--6955},
  year={2023},
  organization={IEEE}
}
```
