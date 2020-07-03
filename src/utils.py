"""
Minor utilities
"""

import sys
from functools import reduce, partial
import warnings
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics
import traceback
import logging
import argparse
import optparse
import datetime
import sys
import pprint
import types
import time
import copy
import subprocess
import glob
from collections import OrderedDict
import os
import signal
import atexit
import json
import inspect

from logging import handlers

import argh
import gin
from gin.config import _OPERATIVE_CONFIG

import torch
from torch.nn.modules.module import _addindent

logger = logging.getLogger(__name__)

def layer_freezer(model):
    for name, parameter in model.state_dict().items():
        if 'output' not in name:
            parameter.requires_grad = False

def specific_parameters_generetor(model, recurse=True):
    for name, param in model.named_parameters(recurse=recurse):
        if 'fc2' in name:
            yield param

def x_to_tensor(data_batch_x):
    tensor_list = [torch.Tensor(convert_img(data_batch_x[i])) for i in range(4)]
    cpu_x = dict(zip(["L-CC", "R-CC", "L-MLO", "R-MLO"], tensor_list))
    return cpu_x

def x_to_device(tensor_x, run_container):
    # Weird hack - multi-GPU can be left on CPU, , but single-device must be on GPU
    if not run_container.is_multi_gpu:
        tensor_x = {k: v.to(run_container.base_device) for k, v in tensor_x.items()}
    return tensor_x

def convert_img(img):
    """
    Convert images from
        [batch_size, H, W, chan]
    to
        [batch_size, chan, H, W
    """
    # return np.expand_dims(np.squeeze(img, 3), 1)
    return np.moveaxis(img, 3, 1)

@gin.configurable
def x_to_tensor_joint(data_batch_x, removing=None):
    #todo: Here I use black removal but mean can be implemented.
    #torch.mean(x, dim=[1, 2, 3, 4]).reshape(shape[0], 1, 1, 1)*torch.ones(shape)

    if removing is None:
        tensor_list = [torch.Tensor(convert_img(data_batch_x[i])) for i in range(2)]
    elif removing == 'mlo':
        tensor_list = [torch.Tensor(convert_img(data_batch_x[0])), torch.zeros(convert_img(data_batch_x[1]).shape)]
    elif removing == 'cc':
        tensor_list = [torch.zeros(convert_img(data_batch_x[0]).shape), torch.Tensor(convert_img(data_batch_x[1]))]
    
    cpu_x = dict(zip(["cc", "mlo"], tensor_list))
    return cpu_x

def x_to_tensor_joint_combine(data_batch_x):
    cpu_x = torch.cat([torch.Tensor(convert_img(data_batch_x[i])) for i in range(2)], 0)
    return cpu_x

def x_to_tensor_single(data_batch_x, view):
    tensor_list = torch.Tensor(convert_img(data_batch_x[0]))
    cpu_x = {view: tensor_list}
    return cpu_x


def numpy_to_torch(obj):
    """
    Convert to tensors all Numpy arrays inside a Python object composed of the
    supported types.
    Args:
        obj: The Python object to convert.
    Returns:
        A new Python object with the same structure as `obj` but where the
        Numpy arrays are now tensors. Not supported type are left as reference
        in the new object.
    Example:
        .. code-block:: python
            >>> from poutyne import numpy_to_torch
            >>> numpy_to_torch({
            ...     'first': np.array([1, 2, 3]),
            ...     'second':[np.array([4,5,6]), np.array([7,8,9])],
            ...     'third': 34
            ... })
            {
                'first': tensor([1, 2, 3]),
                'second': [tensor([4, 5, 6]), tensor([7, 8, 9])],
                'third': 34
            }
    """
    fn = lambda a: torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    return _apply(obj, fn)

    
##SPECIAL LOSS
def gradpenalty(model, grad_norm_penalty = 0.1):
    loss_grad_penalty = grad_norm_penalty * sum([(p.grad ** 2).sum() / 2 for p in model.parameters()])
    loss_grad_penalty.backward()


##METRICS
def acc(y_pred, y_true):
    _, y_pred = y_pred.max(1)
    # _, y_true = y_true.max(1)
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100

def acc_chexnet_covid(y_pred, y_true):
    # round the number and compare with label.
    y_pred = y_pred.round()
    # _, y_true = y_true.max(1)
    acc_pred = (y_pred.float() == y_true).float().mean()
    return acc_pred * 100

def acc_fmnist(y_pred, y_true):
    if isinstance(y_pred, list): 
        y_pred = torch.mean(torch.stack([out.data for out in y_pred], 0), 0)
    _, y_pred = y_pred.max(1)
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100

def isnan(x):
    """
    isnan method that works for both numpy and torch values
    """
    return x != x

def get_loss_function_with_nan_suppression(loss_func):
    """
    ignore both prediction and labels when label value is nan
    """
    def _func(preds, labels):
        preds = torch.where(torch.isnan(labels), torch.zeros_like(preds), preds)
        labels = torch.where(torch.isnan(labels), torch.zeros_like(labels), labels)
        return loss_func(preds, labels)
    return _func 

def eliminate_nan_indices(labels, preds):
    """
    if label is nan for some exams, remove it with the corresponding predictions
    """
    filtered_labels, filtered_preds = [], []
    for i, label in enumerate(labels):
        if not isnan(label):
            filtered_labels.append(label)
            filtered_preds.append(preds[i])
    return np.array(filtered_labels), np.array(filtered_preds)
    
def get_mac_auc(labels, preds):
    """
    Get AUC score, but ignore nan labels from calculation
    """
    label_size = labels.shape[1]
    auc_scores = []
    for i in range(label_size):
        labels_i, preds_i = eliminate_nan_indices(labels[:,i], preds[:,i])
        auc_scores.append(roc_auc_score(labels_i, preds_i))
    return np.array(auc_scores).mean()

def auc_chexnet_covid(y_pred, y_true):
    return get_mac_auc(y_true, y_pred)

def acc_chexnet_covid_numpy(y_pred, y_true):
    # round the number and compare with label.
    y_pred = y_pred.round()
    # _, y_true = y_true.max(1)
    acc_pred = (y_pred == y_true).mean()
    return acc_pred * 100


def evaluation_multiclasses(test_predictions_single_epoch, data_list_save_file, *, label_tabel=None):
    '''
    args: test_predictions_single_epoch, predictions on testset, a numpy array with shape (*, 9) 
    data_list_save_file, '/gpfs/data/luilab/covid/data_list/data_split_20200408.pkl'
    '''
    from src.data.data_chexnet_covid import flatten_exams_list, get_exams
    if label_tabel is None:
        assert data_list_save_file is not None
        _, test_imagelists = pickle.load(open(data_list_save_file, 'rb'))
        test_exams, _ = flatten_exams_list(get_exams(test_imagelists, False, False))
        label_tabel = pd.DataFrame([exam['label'] for exam in test_exams])
        label_tabel['index'] = label_tabel.index
        label_tabel['Id'] = [int(exam['accession_number']) for exam in test_exams]
    
    y_score_dict = {}
    y_score_dict['label_24'] = test_predictions_single_epoch[:, 1].squeeze()
    y_score_dict['label_48'] = test_predictions_single_epoch[:, 1:3].sum(1).squeeze()
    y_score_dict['label_72'] = test_predictions_single_epoch[:, 1:4].sum(1).squeeze()
    y_score_dict['label_96'] = test_predictions_single_epoch[:, 1:5].sum(1).squeeze()
    
    
    auc_breakdown = {}
    prauc_breakdown = {}
    
    for task in ['label_24', 'label_48', 'label_72', 'label_96',]:
        labels, preds = eliminate_nan_indices(label_tabel[task].values,
                                              y_score_dict[task]
                                             )
        auc_breakdown[task] = roc_auc_score(labels, preds)
        prauc_breakdown[task] = average_precision_score(labels, preds)
            
    return pd.DataFrame([auc_breakdown, prauc_breakdown], index=['AUC', 'PRAUC'])
    

def naming_partial_wrap(func, cls):
    partial_func = partial(func, key=cls)
    partial_func.__name__ = '%s_%s'%(func.__name__, cls)
    return partial_func

@gin.configurable
def L2Loss(models, lambda_val):
    
    if isinstance(models, nn.Module):
        models = [models]
    losses = []
    for model in models:
        for param in model.parameters():
            losses.append(param.pow(2).sum())

    return 10**lambda_val* torch.stack(losses).sum() / 2

@gin.configurable
def loss_breast(y_hat, y, criterion, model, 
    key = 'both', #'benign', 'malignant'
    with_l2_regularizer=False):
    
    n_obs, n_label_sets, n_classes = y_hat.shape
    
    if key == 'malignant':
        y_loss = criterion(
            y_hat[:, 1, :].view(n_obs, n_classes), 
            y[:, 1].view(n_obs),
        )
    elif key == 'benign':
        y_loss = criterion(
            y_hat[:, 0, :].view(n_obs, n_classes), 
            y[:, 0].view(n_obs),
        )
    elif key == 'both':
        y_loss = criterion(
            y_hat.view(n_obs * n_label_sets, n_classes),
            y.view(n_obs * n_label_sets),
        )
    else:
        raise ValueError
    
    if with_l2_regularizer:
        regularizer = L2Loss(model)
        y_loss = y_loss + regularizer

    return y_loss


def save_weights(model, optimizer, filename):
    """
    Save all weights necessary to resume training
    """
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)

from contextlib import contextmanager


class Fork(object):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)

    def flush(self):
        self.file1.flush()
        self.file2.flush()


@contextmanager
def replace_logging_stream(file_):
    root = logging.getLogger()
    if len(root.handlers) != 1:
        print(root.handlers)
        raise ValueError("Don't know what to do with many handlers")

    # HANDLE ABSL
    if "_python_handler" in root.handlers[0].__dict__:
        # ABSL logging
        if not isinstance(root.handlers[0]._python_handler, logging.StreamHandler):
            raise ValueError
        stream = root.handlers[0]._python_handler.stream
        root.handlers[0]._python_handler.stream = file_
        try:
            yield
        finally:
            root.handlers[0]._python_handler.stream = stream
    else:
        # Python default logging
        if not isinstance(root.handlers[0], logging.StreamHandler):
            raise ValueError
        stream = root.handlers[0].stream
        root.handlers[0].stream = file_
        try:
            yield
        finally:
            root.handlers[0].stream = stream


@contextmanager
def replace_standard_stream(stream_name, file_):
    stream = getattr(sys, stream_name)
    setattr(sys, stream_name, file_)
    try:
        yield
    finally:
        setattr(sys, stream_name, stream)

def gin_wrap(fnc):
    def main(save_path, config, bindings=""):
        # You can pass many configs (think of them as mixins), and many bindings. Both ";" separated.
        gin.parse_config_files_and_bindings(config.split("#"), bindings.replace("#", "\n"))
        if not os.path.exists(save_path):
            logger.info("Creating folder " + save_path)
            os.system("mkdir -p " + save_path)
        run_with_redirection(os.path.join(save_path, "stdout.txt"),
                             os.path.join(save_path, "stderr.txt"),
                             fnc)(save_path)
    argh.dispatch_command(main)

def run_with_redirection(stdout_path, stderr_path, func):
    print(stdout_path, stderr_path)
    def func_wrapper(*args, **kwargs):
        with open(stdout_path, 'a', 1) as out_dst:
            with open(stderr_path, 'a', 1) as err_dst:
                print(stdout_path)
                print(stderr_path)
                out_fork = Fork(sys.stdout, out_dst)
                err_fork = Fork(sys.stderr, err_dst)
                with replace_standard_stream('stderr', err_fork):
                    with replace_standard_stream('stdout', out_fork):
                        with replace_logging_stream(err_fork):
                            func(*args, **kwargs)

    return func_wrapper

def configure_logger(name='',
        console_logging_level=logging.INFO,
        file_logging_level=None,
        log_file=None):
    """
    Configures logger
    :param name: logger name (default=module name, __name__)
    :param console_logging_level: level of logging to console (stdout), None = no logging
    :param file_logging_level: level of logging to log file, None = no logging
    :param log_file: path to log file (required if file_logging_level not None)
    :return instance of Logger class
    """

    if file_logging_level is None and log_file is not None:
        print("Didnt you want to pass file_logging_level?")

    if len(logging.getLogger(name).handlers) != 0:
        print("Already configured logger '{}'".format(name))
        return

    if console_logging_level is None and file_logging_level is None:
        return  # no logging

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if console_logging_level is not None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(format)
        ch.setLevel(console_logging_level)
        logger.addHandler(ch)

    if file_logging_level is not None:
        if log_file is None:
            raise ValueError("If file logging enabled, log_file path is required")
        fh = handlers.RotatingFileHandler(log_file, maxBytes=(1048576 * 5), backupCount=7)
        fh.setFormatter(format)
        logger.addHandler(fh)

    logger.info("Logging configured!")

    return logger

def quarter_match(target_quater_ind, save_path):
    view_order = ['upper_left', 'upper_right', 'lower_left', 'lower_right']
    if view_order[target_quater_ind] in save_path:
        return True
    else:
        return False

def view_match(view, save_path):
    if view in save_path: 
        return True
    else:
        return False

def summary(model, file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if p is not None:
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        print(string, file=file)
    return count

def create_optimizer(optimizer, 
    model, 
    view_names = ['upper_left', 'upper_right', 'lower_left', 'lower_right'],
    lrs = None,
    ):
    
    lr = optimizer.param_groups[0]['lr']
    para_lr_list = []#[0 for view in view_names] + [0]

    basic_para = []
    ind = 0
    for name, parameter in model.named_parameters():
        flag = True
        for view_name in view_names:
            if view_name in name:
                flag = False
        if flag:
            basic_para.append(parameter)
            para_lr_list.extend([0 for x in range(list(parameter.view(-1).size())[0])])

        for ind_view, view_name in enumerate(view_names):
            if view_name in name:
                para_lr_list.extend([ind_view+1 for x in range(list(parameter.view(-1).size())[0])])

        ind+=1

    for group in optimizer.param_groups:
        group['params'] = basic_para
        group['lr'] = lr

    for ind, view_name in enumerate(view_names):
        if lrs is None: 
            param_group = {'lr': lr, 'params': []}
        else:
            param_group = {'lr': lrs[ind], 'params': []}
        for name, parameter in model.named_parameters():
            if view_name in name: 
                param_group['params'].append(parameter)

        optimizer.add_param_group(param_group)

    return optimizer, para_lr_list
