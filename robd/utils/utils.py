import importlib

import numpy as np
import torch


def get_function(name):  # from https://github.com/aschampion/diluvian/blob/master/diluvian/util.py
    mod_name, func_name = name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func


def get_class(name):
    return get_function(name)


def module_exists(name):
    spec = importlib.util.find_spec(name)
    return (spec is not None)


def function_exists(name):
    mod_name, fct_name = name.rsplit('.', 1)
    spec = importlib.util.find_spec(mod_name)
    if spec is None:
        return False
    else:
        mod = importlib.import_module(mod_name)
        return hasattr(mod, fct_name)


def class_exists(name):
    return function_exists(name)


def get_full_class_name(obj):
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + obj.__class__.__name__


def transform_from_rot_trans(R, t):
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1])).astype(np.float32)


def invert_transform(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_inv = R.T
    t_inv = np.dot(-R.T, t)
    return transform_from_rot_trans(R_inv, t_inv)


def to_cuda(data, device=None):
    if isinstance(data, dict):
        return {k: to_cuda(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cuda(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_cuda(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.cuda(device=device)
    else:
        return data


def to_torch(data):
    return torch.utils.data._utils.collate.default_collate([data])
