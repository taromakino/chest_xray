import sys

import itertools
import timeit
import gin

from .callbacks import Callback
from src import config
from src import models

@gin.configurable
class ProgressionCallback(Callback):
    def __init__(self,
        key=None,
        other_metrics = [],
        include_fc = False,
        view_names = None
        ):
        """
        Deprecated arguments: key, include_fc, view_names
        """

        super(ProgressionCallback, self).__init__()
        self.metrics = ['loss']
        self.other_metrics = other_metrics
    

    def on_train_begin(self, logs):
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']

    def on_train_end(self, logs):
        pass

    def on_epoch_begin(self, epoch, logs):
        self.step_times_sum = 0.
        self.epoch = epoch
        sys.stdout.write("\rEpoch %d/%d" % (self.epoch, self.epochs))
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs):
        epoch_total_time = logs['time']

        metrics_str = self._get_metrics_string(logs)
        iol_str = self._get_iol_string(logs)
        if self.steps is not None:
            print("\rEpoch %d/%d %.2fs/%.2fs: Step %d/%d: %s. %s" %
                  (self.epoch, self.epochs, epoch_total_time, timeit.default_timer()-logs['epoch_begin_time'], self.steps, self.steps, metrics_str, iol_str))

        else:
            print("\rEpoch %d/%d %.2fs/%.2fs: Step %d/%d: %s. %s" %
                  (self.epoch, self.epochs, epoch_total_time, timeit.default_timer()-logs['epoch_begin_time'], self.last_step, self.last_step, metrics_str, iol_str))

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        batch = batch + 1 # poutyne uses 1-based batch number, but we don't
        
        self.step_times_sum += timeit.default_timer()-logs['batch_begin_time']

        metrics_str = self._get_metrics_string(logs)
        iol_str = self._get_iol_string(logs)
        #print(iol_str)
        times_mean = self.step_times_sum / batch
        if self.steps is not None:
            remaining_time = times_mean * (self.steps - batch)
            
            sys.stdout.write("\rEpoch %d/%d ETA %.2fs Step %d/%d: %s. %s" %
                             (self.epoch, self.epochs, remaining_time, batch, self.steps, metrics_str, iol_str))
            if 'cumsum_iol' in iol_str: sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("\rEpoch %d/%d %.2fs/step Step %d: %s. %s" %
                             (self.epoch, self.epochs, times_mean, batch, metrics_str, iol_str))
            sys.stdout.flush()
            self.last_step = batch

    def _get_metrics_string(self, logs):
        train_metrics_str_gen = ('{}: {:f}'.format(k, logs[k]) for k in self.metrics if logs.get(k) is not None)
        val_metrics_str_gen = ('{}: {:f}'.format('val_' + k, logs['val_' + k]) for k in self.metrics
                               if logs.get('val_' + k) is not None)
        return ', '.join(itertools.chain(train_metrics_str_gen, val_metrics_str_gen))

    def _get_iol_string(self, logs):
        str_gen = ['{}: {:f}'.format(k, logs[k]) for k in self.other_metrics if logs.get(k) is not None]
        #print(str_gen, '\n',[(k, logs[k]) for k in ['average_iol_current_epoch', 'average_iol']])
        return  ', '.join(str_gen)

class ValidationProgressionCallback(Callback):
    def __init__(self, 
                 phase,
                 metrics_names,
                 steps=None):
        self.params = {}
        self.params['steps'] = steps
        self.params['phase'] = phase 
        self.metrics = metrics_names

        super(ValidationProgressionCallback, self).__init__()

    def _get_metrics_string(self, logs):
        metrics_str_gen = ('{}: {:f}'.format(self.params['phase'] + '_' + k, logs[k]) for k in self.metrics
                               if logs.get(k) is not None)
        return ', '.join(metrics_str_gen)

    def on_batch_begin(self, batch, logs):
        if batch==1:
            self.step_times_sum = 0.
        
        self.steps = self.params['steps']

    def on_batch_end(self, batch, logs):
        self.step_times_sum += timeit.default_timer()-logs['batch_begin_time']

        metrics_str = self._get_metrics_string(logs)
        times_mean = self.step_times_sum / batch
        if self.steps is not None:
            remaining_time = times_mean * (self.steps - batch)
            
            sys.stdout.write("\r%s ETA %.2fs Step %d/%d: %s." %
                             (self.params['phase'], remaining_time, batch, self.steps, metrics_str))
            sys.stdout.flush()
        else:
            sys.stdout.write("\r%s %.2fs/step Step %d: %s." %
                             (self.params['phase'], times_mean, batch, metrics_str))
            sys.stdout.flush()
            self.last_step = batch

