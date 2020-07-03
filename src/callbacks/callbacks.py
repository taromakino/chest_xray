# -*- coding: utf-8 -*-
"""
Callbacks implementation. Inspired by Keras.
"""

# NOTE(kudkudak): There is no (yet) standalone tensorboard, and I don't think it makes sense to use tensorboardX
import tensorflow

import timeit
import gin
import sys
import numpy as np
import pandas as pd
import os
import pickle
import logging
import time
import datetime
import json
import copy
from collections import defaultdict, OrderedDict

import torch


from gin.config import _OPERATIVE_CONFIG

from src.utils import save_weights
from src.utils import acc_chexnet_covid, auc_chexnet_covid, acc_chexnet_covid_numpy

types_of_instance_to_save_in_csv = (int, float, complex, np.int64, np.int32, np.float32, np.float64, np.float128, str)
logger = logging.getLogger(__name__)

class CallbackList:
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_forward_begin(self, batch, data):
        for callback in self.callbacks:
            callback.on_forward_begin(batch, data)

    def on_backward_end(self, batch):
        for callback in self.callbacks:
            callback.on_backward_end(batch)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_train_epoch_begin(self, epoch, logs):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_epoch_begin'):
                callback.on_train_epoch_begin(epoch, logs)

    def on_val_epoch_begin(self, epoch, logs):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_val_epoch_begin'):
                callback.on_val_epoch_begin(epoch, logs)

    def on_test_epoch_begin(self, epoch, logs):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_test_epoch_begin'):
                callback.on_test_epoch_begin(epoch, logs)

    def on_val_batch_end(self, batch, logs):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_val_batch_end(batch, logs)

    def __iter__(self):
        return iter(self.callbacks)

class Callback(object):
    def __init__(self):
        pass

    def set_config(self, config):
        self.config = config

    def set_meta_data(self, meta_data):
        self.meta_data = meta_data

    def set_save_path(self, save_path):
        self.save_path = save_path

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model(self, model, ignore=True):
        if ignore:
            return
        self.model = model

    def set_params(self, params):
        self.params = params

    def set_dataloader(self, data):
        self.data = data

    def get_dataloader(self):
        return self.data

    def get_config(self):
        return self.config

    def get_meta_data(self):
        return self.meta_data

    def get_optimizer(self):
        return self.optimizer

    def get_params(self):
        return self.params

    def get_model(self):
        return self.model

    def get_save_path(self):
        return self.save_path

    def on_epoch_begin(self, epoch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        pass

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        pass

    def on_forward_begin(self, batch, data):
        pass

    def on_backward_end(self, batch):
        pass

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        pass

    def on_train_epoch_begin(self, epoch, logs):
        pass

    def on_val_epoch_begin(self, epoch, logs):
        pass

    def on_test_epoch_begin(self, epoch, logs):
        pass

    def on_val_batch_end(self, batch, logs):
        pass
    
    
    
class BaseLogger(Callback):
    """Callback that accumulates epoch averages."""
    def __init__(self):
        super(BaseLogger, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = defaultdict(float)

    def on_batch_end(self, batch, logs=None):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        if logs is not None:
            for k, v in logs.items():
                self.totals[k] += v * batch_size


    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.totals:
                logs[k] = self.totals[k] / self.seen
 

@gin.configurable
class GradualUnfreezing(Callback):
    """
    Gradually unfreeze layers from last to first every unfreeze_every epochs.
    Assume layers are being progressively unfrozeon from the last layer.
    """
    def __init__(self, unfreeze_every=1, level='lowest'):
        self.unfreeze_every = unfreeze_every
        self.layers_info_init = None
        self.level = level
        super(GradualUnfreezing, self).__init__()
        
    @staticmethod
    def pop_last_item(k):
        keywords = k.split('.')
        return '.'.join(keywords[:-1]), keywords[-1]
    
    def get_layers_info(self):
        """
        group layers according to self.level
        This assumes layer names follow syntax of the following:
            'densenet121.features.denseblock4.denselayer15.conv1.weight'
        If not, recommended to upgrade pytorch version.
        
        'lowest' groups weight and bias etc. from each norm or conv layer
        'layer' groups norm and conv from each denselayer
        'block' groups all layers that belong to the same block
        """
        layer_names_dict = OrderedDict()
        for name, parameter in self.model.named_parameters():
            layer_name, _ = self.pop_last_item(name)
            if self.level == 'layer' or self.level == 'block':
                layer_name_candidate, last_item = self.pop_last_item(layer_name)
                if last_item.startswith('conv') or last_item.startswith('norm'):
                    layer_name = layer_name_candidate
                if self.level == 'block':
                    layer_name_candidate, last_item = self.pop_last_item(layer_name)
                    if 'layer' in last_item:
                        layer_name = layer_name_candidate
            if layer_name not in layer_names_dict:
                layer_names_dict[layer_name] = parameter.requires_grad
            else:
                # consider a layer is unfrozen when all of its params have requires_grad=True
                layer_names_dict[layer_name] &= parameter.requires_grad
                
        #total_num_layers = len(layer_names_dict)
        num_unfrozen_layers = sum(layer_names_dict.values())
        return list(layer_names_dict.keys()), num_unfrozen_layers

    def unfreeze_additional_layers(self, epoch):
        num_layers_gradual_unfreeze = epoch // self.unfreeze_every
        num_layers_to_be_unfrozen_total = num_layers_gradual_unfreeze + self.num_unfrozen_layers_init
        layer_names_to_be_unfrozen = self.layers_info_init[-num_layers_to_be_unfrozen_total:]
        for layer_name in layer_names_to_be_unfrozen:
            for name, parameter in self.model.named_parameters():
                if name.startswith(layer_name):
                    if not parameter.requires_grad:
                        logger.info(f'Unfreezing {name}')
                        parameter.requires_grad = True
                    else:
                        logger.info(f'Already unfrozen: {name}')
                        
        
    def on_epoch_begin(self, epoch, logs):
        # 1. Get layer names and how many are unfrozen
        if self.layers_info_init is None:
            self.layers_info_init, self.num_unfrozen_layers_init = self.get_layers_info()
        # 2. Unfreeze subsequent epoch % unfreeze_every layers
        #    This must be able to handle continuing to train from a saved checkpoint
        #    If picking up from epoch 34, for example, we must unfreeze layers accordingly.
        self.unfreeze_additional_layers(epoch)
        

@gin.configurable
class EarlyStopping(Callback):
    """
    The source code of this class is under the MIT License and was copied from the Keras project,
    and has been modified.
    Stop training when a monitored quantity has stopped improving.
    Args:
        monitor (int): Quantity to be monitored.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement. 
            (Default value = 0)
        patience (int): Number of epochs with no improvement after which training will be stopped.
            (Default value = 0)
        verbose (bool): Whether to print when early stopping is done.
            (Default value = False)
        mode (string): One of {'min', 'max'}. In `min` mode, training will stop when the quantity
            monitored has stopped decreasing; in `max` mode it will stop when the quantity monitored has
            stopped increasing. 
            (Default value = 'min')
    """

    def __init__(self, *, monitor='val_loss', min_delta=0, patience=0, verbose=False, mode='min'):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['min', 'max']:
            raise ValueError("Invalid mode '%s'" % mode)
        self.mode = mode

        if mode == 'min':
            self.min_delta *= -1
            self.monitor_op = np.less
        elif mode == 'max':
            self.min_delta *= 1
            self.monitor_op = np.greater

    def on_train_begin(self, logs):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.mode == 'min' else -np.Inf

    def on_epoch_end(self, epoch, logs):
        current = logs[self.monitor]
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs):
        if self.stopped_epoch > 0 and self.verbose:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


@gin.configurable
class CompletedStopping(Callback):

    def __init__(self, *, monitor='acc_fmnist', patience=5, verbose=True):
        super(CompletedStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience

        self.verbose = verbose
        
        self.stopped_epoch = 0

    def on_train_begin(self, logs):
        # Allow instances to be re-used
        self.stopped_epoch = 0
        self.counter = 0

    def on_epoch_end(self, epoch, logs):
        current = logs[self.monitor]
        if current == 100:
            self.counter +=1
            
        if self.counter>=self.patience:
            
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs):
        if self.stopped_epoch > 0 and self.verbose:
            print('Epoch %05d: completed stopping' % (self.stopped_epoch + 1))

@gin.configurable
class LRSchedule(Callback):
    def __init__(self, base_lr, schedule):
        self.schedule = schedule
        self.base_lr = base_lr
        super(LRSchedule, self).__init__()

    def on_epoch_begin(self, epoch, logs):
        # Epochs starts from 0
        for e, v in self.schedule:
            if epoch < e:
                break
        for group in self.optimizer.param_groups:
            group['lr'] = v * self.base_lr
        logger.info("Fix learning rate to {}".format(v * self.base_lr))

@gin.configurable
class ReduceLROnPlateau_PyTorch(Callback):
    def __init__(self, metric):
        self.metric = metric 
    
    def on_train_begin(self, logs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
          mode='min', 
          factor=0.3, 
          patience=5, 
          verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-4, eps=1e-08)

    def on_epoch_end(self, epoch, logs):
        '''Check for end of current cycle, apply restarts when necessary.'''
        self.scheduler.step(logs[self.metric])

@gin.configurable
class ReduceLROnPlateau(Callback):
    def __init__(self, base_lr, factor=0.5, patience=5, threshold=0.1, starting_loss = 0.05, new=False):
        self.factor=factor
        self.patience=patience
        self.threshold=threshold
        self.base_lr = base_lr
        self.best_loss = None
        self.bad_counter = 0
        self.starting_loss = starting_loss
        self.new=new
        self.lowerest = 1e-5 if self.new else 1e-6

        super(ReduceLROnPlateau, self).__init__()

    def on_batch_end(self, batch, logs):
        if self.best_loss is None:
            self.best_loss = logs['loss']
        # Epochs starts from 0
        if logs['loss']<self.best_loss:
            self.best_loss = logs['loss']
            self.bad_counter = 0
        elif (logs['loss']>self.threshold*self.best_loss + self.best_loss):
            if (logs['loss']>self.starting_loss) and self.new:
                pass
            else:
                self.bad_counter +=1
        else:
            pass

        if self.bad_counter>self.patience and self.base_lr>self.lowerest and logs['loss']<self.starting_loss: 

            self.bad_counter = 0
            self.best_loss = logs['loss']
            
            self.base_lr = self.factor * self.base_lr    
            for group in self.optimizer.param_groups:
                if group['lr']>self.base_lr:
                    group['lr'] = self.base_lr

            logger.info("Fix learning rate to {}".format( self.base_lr))

@gin.configurable
class CycleScheduler(Callback):
    
    def __init__(self,
                 starting_condition_epoch = 100,
                 starting_condition_loss = 0.1,
                 factor = 0.3,
                 step_size = 59,
                 ):

        self.starting_condition_epoch = starting_condition_epoch
        self.starting_condition_loss = starting_condition_loss
        self.factor = factor
        self.start_flag = False
        self.step_size = step_size

    def on_train_begin(self, logs):
        for group in self.optimizer.param_groups:
            self.base_lr = group['lr']
            break

    def on_epoch_begin(self, epoch, logs):
        self.step_counter = 0

    def on_batch_begin(self, batch, logs):
        if self.start_flag:
            self.step_counter +=1 
            if self.step_counter<=self.step_size:
                
                lr = (self.max_lr - self.min_lr)/self.step_size * self.step_counter + self.min_lr
            else:
                lr = self.max_lr - (self.max_lr - self.min_lr)/(self.step_counter - self.step_size ) * self.step_counter

            if self.step_counter>2*self.step_size:
                self.step_counter = 0

            for group in self.optimizer.param_groups:
                group['lr'] = lr
            logger.info("Fix learning rate to {}".format(lr))

    def on_epoch_end(self, epoch, logs):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch>self.starting_condition_epoch and logs['loss']>self.starting_condition_loss and not self.start_flag:
            self.min_lr = (1-self.factor)*self.base_lr
            self.max_lr = self.base_lr*(1+self.factor)
            
            self.start_flag = True
            self.step_counter = 0
            
class History(Callback):
    """
    History callback.

    By default saves history every epoch, can be configured to save also every k examples
    """
    def __init__(self, save_every_k_examples=-1, mode='train'):
        self.examples_seen = 0
        self.save_every_k_examples = save_every_k_examples
        self.examples_seen_since_last_population = 0
        self.mode = mode
        super(History, self).__init__()

    def on_train_begin(self, logs=None):
        # self.epoch = []
        self.history = {}
        self.history_batch = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

            # if k.endswith("labels"):# and (k not in self.history):
            #     # we don't need to save labels every epoch.
            #     #self.history[k] = v
            #     pass
            # else:
            #     self.history.setdefault(k, []).append(v)

        if self.save_path is not None:
            base_filename = 'history.pkl' if self.mode == 'train' else 'eval_history.pkl'
            pickle.dump(self.history, open(os.path.join(self.save_path, base_filename), "wb"))
            if self.save_every_k_examples != -1:
                pickle.dump(self.history_batch, open(os.path.join(self.save_path, "history_batch.pkl"), "wb"))

    def on_batch_end(self, batch, logs=None):
        # Batches starts from 1
        if self.save_every_k_examples != -1:
            if getattr(self.model, "history_batch", None) is None:
                setattr(self.model, "history_batch", self)
            assert "size" in logs
            self.examples_seen += logs['size']
            logs['examples_seen'] = self.examples_seen
            self.examples_seen_since_last_population += logs['size']

            if self.examples_seen_since_last_population > self.save_every_k_examples:
                for k, v in logs.items():
                    self.history_batch.setdefault(k, []).append(v)
                self.examples_seen_since_last_population = 0


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        del state['optimizer']
        return state

    def __setstate__(self, newstate):
        newstate['model'] = self.model
        newstate['optimizer'] = self.optimizer
        self.__dict__.update(newstate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, self.filepath))
                        self.best = current
                        save_weights(self.model, self.optimizer, self.filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, self.filepath))
                    save_weights(self.model, self.optimizer, self.filepath)


class LambdaCallback(Callback):
    def __init__(self,
                 on_epoch_begin=None,
                 on_epoch_end=None,
                 on_batch_begin=None,
                 on_batch_end=None,
                 on_train_begin=None,
                 on_train_end=None):
        super(LambdaCallback, self).__init__()
        if on_epoch_begin is not None:
            self.on_epoch_begin = on_epoch_begin
        else:
            self.on_epoch_begin = lambda epoch, logs: None
        if on_epoch_end is not None:
            self.on_epoch_end = on_epoch_end
        else:
            self.on_epoch_end = lambda epoch, logs: None
        if on_batch_begin is not None:
            self.on_batch_begin = on_batch_begin
        else:
            self.on_batch_begin = lambda batch, logs: None
        if on_batch_end is not None:
            self.on_batch_end = on_batch_end
        else:
            self.on_batch_end = lambda batch, logs: None
        if on_train_begin is not None:
            self.on_train_begin = on_train_begin
        else:
            self.on_train_begin = lambda logs: None
        if on_train_end is not None:
            self.on_train_end = on_train_end
        else:
            self.on_train_end = lambda logs: None


class LambdaCallbackPickableEveryKExamples(LambdaCallback):
    """
    Runs lambda every K examples.

    Note: Assumes 'size' key in batch logs denoting size of the current minibatch
    """
    def __init__(self,
                 on_k_examples=None,
                 k=45000,
                 call_after_first_batch=False,
                 **kwargs):
        super(LambdaCallback, self).__init__()
        self.__dict__.update(kwargs)
        self.examples_seen = 0
        self.call_after_first_batch = call_after_first_batch
        self.examples_seen_since_last_call = 0
        self.k = k
        self.on_k_examples = on_k_examples
        self.calls = 0

    def on_batch_end(self, batch, logs=None):
        # Batches starts from 1
        assert "size" in logs
        self.examples_seen += logs['size']
        self.examples_seen_since_last_call += logs['size']

        if (self.call_after_first_batch and batch == 1) \
                or self.examples_seen_since_last_call > self.k:
            logger.info("Batch " + str(batch))
            logger.info("Firing on K examples, ex seen = " + str(self.examples_seen))
            logger.info("Firing on K examples, ex seen last call = " + str(self.examples_seen_since_last_call))
            self.on_k_examples(logs) # self.calls, self.examples_seen,
            self.examples_seen_since_last_call = 0
            self.calls += 1

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['on_k_examples']
        return state


class DumpTensorboardSummaries(Callback):
    def __init__(self):
        super(DumpTensorboardSummaries, self).__init__()

    @property
    def file_writer(self):
        if not hasattr(self, '_file_writer'):
            self._file_writer = tensorflow.compat.v1.summary.FileWriter(
                self.save_path, flush_secs=10.)
        return self._file_writer

    def on_epoch_end(self, epoch, logs=None):
        summary = tensorflow.compat.v1.Summary()
        for key, value in logs.items():
            try:
                float_value = float(value)
                value = summary.value.add()
                value.tag = key
                value.simple_value = float_value
            except:
                pass
        self.file_writer.add_summary(
            summary, epoch)

@gin.configurable
class EvaluateEpoch(Callback):
    def __init__(self, metrics):
        '''
        '''
        super(EvaluateEpoch, self).__init__()
        self.metrics = metrics
        self.metric_func_dict = {'acc': acc_chexnet_covid_numpy, 'auc': auc_chexnet_covid}
            
    def on_epoch_end(self, epoch, logs=None):
        '''appending to log in callback'''
        
        if 'train_predictions' in logs:
            train_preds = logs['train_predictions']
            train_labels = logs['train_labels']
        if 'val_predictions' in logs:
            val_preds = logs['val_predictions']
            val_labels = logs['val_labels']
        if 'test_predictions' in logs:
            test_preds = logs['test_predictions']
            test_labels = logs['test_labels']

        for metric in self.metrics:
            
            func = metric.split('_')[0]

            if 'train_predictions' in logs:
                logs['{}'.format(metric)] = self.metric_func_dict[func](train_preds, train_labels)

            if 'val_predictions' in logs:
                logs['val_{}'.format(metric)] = self.metric_func_dict[func](val_preds, val_labels) 
            
            if 'test_predictions' in logs:
                logs['test_{}'.format(metric)] = self.metric_func_dict[func](test_preds, test_labels) 
                    

@gin.configurable
class MetaSaver(Callback):
    def __init__(self):
        super(MetaSaver, self).__init__()

    def on_train_begin(self, logs=None):
        logger.info("Saving meta data information from the beginning of training")

        assert os.system("cp {} {}".format(sys.argv[0], self.save_path)) == 0, "Failed to execute cp of source script"

        utc_date = datetime.datetime.utcnow().strftime("%Y_%m_%d")

        time_start = time.time()
        cmd = "python " + " ".join(sys.argv)
        self.meta = {"cmd": cmd,
                     "save_path": self.save_path,
                     "most_recent_train_start_date": utc_date,
                     "execution_time": -time_start}

        json.dump(self.meta, open(os.path.join(self.save_path, "meta.json"), "w"), indent=4)

        # Copy gin configs used, for reference, to the save folder
        os.system("rm " + os.path.join(self.save_path, "*gin"))
        for gin_config in sys.argv[2].split(";"):
            os.system("cp {} {}".format(gin_config, self.save_path))

    def on_train_end(self, logs=None):
        self.meta['execution_time'] += time.time()
        json.dump(self.meta, open(os.path.join(self.save_path, "meta.json"), "w"), indent=4)
        os.system("touch " + os.path.join(self.save_path, "FINISHED"))

@gin.configurable
class BreastDataLoader(Callback):
    def __init__(self, 
                 mode="multiclass_cancer_sides",
                 ):
        #self.view_weights = view_weights
        
        super(BreastDataLoader, self).__init__()
        self.mode =  mode

    def on_train_epoch_begin(self, epoch, logs):
        current_random_seed = self.data.seed_shifter.get_seed(phase='training', epoch_number=epoch)
        self.data.start_training_epoch(random_seed=current_random_seed, mode=self.mode)

    def on_val_epoch_begin(self, epoch, logs):
        current_random_seed = self.data.seed_shifter.get_seed(phase='validation', epoch_number=epoch)
        self.data.start_validation_epoch(random_seed=current_random_seed, mode=self.mode)

    def on_test_epoch_begin(self, epoch, logs):
        current_random_seed = self.data.seed_shifter.get_seed(phase='test', epoch_number=epoch)
        self.data.start_test_epoch(random_seed=current_random_seed, mode=self.mode)
