"""Helper functions.

Based and partially copied from the model registry from the
timm package ( https://github.com/rwightman/pytorch-image-models ).
"""
from copy import deepcopy

import torch
from torch.hub import load_state_dict_from_url

from typing import Callable, Optional, Dict


_DOWNLOAD_PROGRESS = False
_CHECK_HASH = False


def set_download_progress(enable=True):
    """ Set download progress for pretrained weights on/off (globally). """
    global _DOWNLOAD_PROGRESS
    _DOWNLOAD_PROGRESS = enable


def set_check_hash(enable=True):
    """ Set hash checking for pretrained weights on/off (globally). """
    global _CHECK_HASH
    _CHECK_HASH = enable


def build_model_with_cfg(
        model_cls: Callable,
        cfg: Optional[Dict] = None,
        weights: Optional[str] = None,
        **kwargs):
    """Builds a model with a given config and restores weights.

    Args:
        model_cls: The model class.
        cfg: Dictionary with config parameters.
        weights: Path to model weights.

    Returns:
        Model.
    """
    if cfg is not None:
        cfg = deepcopy(cfg)  # avoid changing the input cfg dict
        cfg.update(kwargs)
        kwargs = cfg
    model = model_cls(**kwargs)

    if weights is not None:
        load_from_url = weights.startswith('http')
        if not load_from_url:
            print(f'Using model weights from file {weights}.')
            state_dict = torch.load(weights, map_location='cpu')
        else:
            print(f'Using model weights from url {weights}.')
            state_dict = load_state_dict_from_url(weights, map_location='cpu', progress=_DOWNLOAD_PROGRESS,
                                                  check_hash=_CHECK_HASH)

        model.load_state_dict(state_dict, strict=True)
    return model
