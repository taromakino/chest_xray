# -*- coding: utf-8 -*-
"""
A gorgeous, self-contained, training loop. Uses Poutyne implementation, but this can be swapped later.
"""

import logging
import os
import tqdm
import pickle
from functools import partial
import re
import timeit

import numpy as np
import pandas as pd
import torch
import gin
import h5py
from contextlib import contextmanager
from collections import defaultdict

from src.callbacks.callbacks import ModelCheckpoint, LambdaCallback, History, DumpTensorboardSummaries, BaseLogger
from src.callbacks.progress import ProgressionCallback
from src.utils import save_weights, layer_freezer, get_loss_function_with_nan_suppression

logger = logging.getLogger(__name__)

types_of_instance_to_save_in_csv = (int, float, complex, np.int64, np.int32, np.float32, np.float64, np.float128, str)
types_of_instance_to_save_in_history = (int, float, complex, np.int64, np.int32, np.float32, np.float64, np.ndarray, np.float128,str)

def _construct_default_callbacks(model, optimizer, H, H_batch, save_path, checkpoint_monitor, save_freq, custom_callbacks,
                                 use_tb, save_history_every_k_examples):
    
    history_batch = os.path.join(save_path, "history_batch")
    if not os.path.exists(history_batch):
        os.mkdir(history_batch)

    callbacks = []
    callbacks.append(LambdaCallback(on_epoch_end=partial(_append_to_history_csv, H=H), 
                                    on_batch_end=partial(_append_to_history_csv_batch, H=H_batch)
                               )
    )

    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_history_csv, save_path=save_path, H=H)))
    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_history_csv_batch, save_path=history_batch, H=H_batch)))
    callbacks.append(History(save_every_k_examples=save_history_every_k_examples))
    
    if save_freq > 0:
        callbacks.append(ModelCheckpoint(monitor=checkpoint_monitor,
                                     save_best_only=True,
                                     mode='max',
                                     filepath=os.path.join(save_path, "model_best_val.pt")))
        
        def save_weights_fnc(epoch, logs):
            if epoch % save_freq == 0:
                logger.info("Saving model from epoch " + str(epoch))
                save_weights(model, optimizer, os.path.join(save_path, "model_last_epoch.pt"))

        callbacks.append(LambdaCallback(on_epoch_end=save_weights_fnc))

        callbacks.append(LambdaCallback(on_epoch_end=partial(_save_loop_state, 
            save_callbacks=custom_callbacks,
            save_path=save_path)))
    
    if use_tb:
        callbacks.append(DumpTensorboardSummaries())

    #TODO: debug later TypeError: cannot serialize '_io.TextIOWrapper' object    
        
    return callbacks

def _save_loop_state(epoch, logs, save_path, save_callbacks):
    logger.info("Saving loop_state.pkl")  # TODO: Debug?

    loop_state = {"epochs_done": epoch, "callbacks": save_callbacks}  # 0 index
    
    ## A small hack to pickle callbacks ##
    data_batch_loader_flag = False
    if len(save_callbacks):
        m, opt, md, dloader = save_callbacks[0].get_model(), save_callbacks[0].get_optimizer(), save_callbacks[0].get_meta_data(), save_callbacks[0].get_dataloader()
        for c in save_callbacks:
            c.set_model(None, ignore=False)  # TODO: Remove
            c.set_optimizer(None)
            c.set_params(None)  # TODO: Remove
            c.set_meta_data(None)
            c.set_dataloader(None)
            try:
                data_batch_loader = c.data_batch_loader
                data_observation = c.data_observation
                c.data_batch_loader = None
                c.data_observation = None
                data_batch_loader_flag=True
            except AttributeError:
                pass

    pickle.dump(loop_state, open(os.path.join(save_path, "loop_state.pkl"), "wb"))
    if len(save_callbacks):
        for c in save_callbacks:
            c.set_model(m, ignore=False)
            c.set_optimizer(opt)
            c.set_meta_data(md)
            c.set_dataloader(dloader)
            try:
                if data_batch_loader_flag:
                    c.data_batch_loader = data_batch_loader
                    c.data_observation = data_observation
            except AttributeError:
                pass

    logger.info("Saved loop_state.pkl")  # TODO: Debug?

def _saving_by_h5py(file, H):
    h = h5py.File(file)
    for key, data in H.items():
        data = np.array(data)

        if key in h:
            
            h[key].resize(data.shape)
            h[key][...] = data

        else:
            maxshape = [None]
            maxshape.extend(data.shape[1:])
            h.create_dataset(key, data=np.array(data), maxshape=maxshape)

    h.close()

def _saving_by_pickle(file, H):

    with open(file, 'wb') as f:
    
         pickle.dump(H, f, pickle.HIGHEST_PROTOCOL)

def _save_history_csv_batch(epoch, logs, save_path, H):
    # out = ""
    # for key, value in logs.items():
    #     if isinstance(value, (int, float, complex, np.float32, np.float64)):
    #         out += "{key}={value}\t".format(key=key, value=value)

    '''
    saving seperately in files with epoch as names in folder - history_batch 
    '''

    H_tosave = {}
    for key, value in H.items():
        
        if isinstance(value[-1], types_of_instance_to_save_in_csv):
            H_tosave[key] = value
    
    pd.DataFrame(H_tosave).to_csv(os.path.join(save_path, "epoch_%d.csv"%epoch), index=False)

    for key in H.keys():
        H[key] = []

def _save_history_csv(epoch, logs, save_path, H, mode='train'):
    base_filename = 'history.csv' if mode == 'train' else 'eval_history.csv'
    out = ""
    for key, value in logs.items():
        if isinstance(value, types_of_instance_to_save_in_csv):
            out += "{key}={value}\t".format(key=key, value=value)
    logger.info(out)
    
    
    logger.info("Saving history to " + os.path.join(save_path, base_filename))
    H_tosave = {}
    for key, value in H.items():
        if isinstance(value[-1], types_of_instance_to_save_in_csv):
            
            H_tosave[key] = value
    print(H_tosave)
    pd.DataFrame(H_tosave).to_csv(os.path.join(save_path, base_filename), index=False)


def _append_to_history_csv_batch(batch, logs, H):
    if len(logs)>len(H) or len(H)==0:
        for key, value in logs.items():
            if isinstance(value, (types_of_instance_to_save_in_history)):
                H[key] = [value]
    elif len(H)==len(logs) :
        for key, value in logs.items():

            if isinstance(value, (types_of_instance_to_save_in_history)):
                if key not in H:
                    print(key)
                    H[key] = [value]
                else:
                    H[key].append(value)

            else:
                pass
    #print(H)

def _append_to_history_csv(epoch, logs, H):
    for key, value in logs.items():
        if isinstance(value, types_of_instance_to_save_in_history):
            if key not in H:
                H[key] = [value]
            else:
                H[key].append(value)

            # Epoch is 0 first, so 1 key. Etc
            # assert len(H[key]) == epoch + 1, "Len H[{}] is {}, expected {} ".format(key, len(H[key]), epoch + 1)
        else:
            pass

def _load_pretrained_model(model, save_path, model_to_load='model_best_val.pt'):
    checkpoint = torch.load(os.path.join(save_path, model_to_load))
    model.load_state_dict(checkpoint['model'])
    logger.info("Done reloading!")

def handle_old_layer_names(state_dict):
    # Code modified from torchvision densenet source for loading from pre .4 densenet weights.

    remove_data_parallel = False # Change if you don't want to use nn.DataParallel(model)

    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        match = pattern.match(key)
        new_key = match.group(1) + match.group(2) if match else key
        new_key = new_key[7:] if remove_data_parallel else new_key
        state_dict[new_key] = state_dict[key]
        # Delete old key only if modified.
        if match or remove_data_parallel: 
            del state_dict[key]
            
    return state_dict

def _load_BIRADS_pretrained(model, save_path):
    checkpoint = torch.load(save_path)
    
    model_dict = model.state_dict()
    
    checkpoint = handle_old_layer_names(checkpoint)

    # 2. overwrite entries in the existing state dict
    model_dict.update(checkpoint) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    logger.info("Done reloading!")


def _reload(model, optimizer, save_path, callbacks):
    model_last_epoch_path = os.path.join(save_path, "model_last_epoch.pt")
    loop_state_path = os.path.join(save_path, "loop_state.pkl")
    history_csv_path = os.path.join(save_path, "history.csv")

    if not os.path.exists(model_last_epoch_path) or not os.path.exists(loop_state_path):
        logger.warning("Failed to find last epoch model or loop state")
        return {}, 0

    # Reload everything (model, optimizer, loop state)
    logger.warning("Reloading weights!")
    checkpoint = torch.load(model_last_epoch_path)
    model.load_state_dict(checkpoint['model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("Reloading loop state!")
    loop_state = pickle.load(open(loop_state_path, 'rb'))
    logger.info("Reloading history!")
    H = pd.read_csv(history_csv_path)
    if 'epoch.1' in H:
        H = H.drop(columns=['epoch.1'])
    H = {col: list(H[col].values) for col in H.columns}
    print(H)

    logger.info("Done reloading!")

    # Small back-up
    os.system("cp " + os.path.join(save_path, "history.pkl") + " " + os.path.join(save_path, "history.pkl.bckp"))

    # Setup the rest
    epoch_start = loop_state['epochs_done'] + 1
    if not len(H[next(iter(H))]) == loop_state['epochs_done'] + 1:
        raise IOError("Mismatch between saved history and epochs recorded. "
                      "Found len(H)={0} and epoch_start={1} "
                      "Run was likely interrupted incorrectly and cannot be rerun.".format(len(H[next(iter(H))]),
                                                                                           epoch_start))

    # Load all callbacks from the loop_state
    for e, e_loaded in zip(callbacks, loop_state['callbacks']):
        assert type(e) == type(e_loaded)
        if hasattr(e, "__setstate__"):
            e.__setstate__(e_loaded.__dict__)
        else:
            e.__dict__.update(e_loaded.__dict__)

    # Some diagnostics
    logger.info(loop_state)
    for k in H:
        logger.info((k, len(H)))
        break
    if 'epoch_begin_time' not in H:
        H['epoch_begin_time'] = [0]*len(H[k])
    print(H)

    logger.info("epoch_start={}".format(epoch_start))

    return H, epoch_start

@contextmanager
def _set_training_mode(model, training):
    old_training = model.training
    model.train(training)
    with torch.set_grad_enabled(training):
        yield
    model.train(old_training)
    
def _loop(
    model, test_generator, valid_generator, train_generator, optimizer, scheduler, loss_function, initial_epoch, epochs, callbacks, steps_per_epoch=None, metrics=[], device="cuda", suppress_nan_labels_in_loss=True
):
    """
    Internal implementation of the training loop.
    Notes
    -----
    https://github.com/gmum/toolkit/blob/master/pytorch_project_template/src/training_loop.py
    """
    if suppress_nan_labels_in_loss:
        loss_function = get_loss_function_with_nan_suppression(loss_function)
    
    if train_generator is not None:
        progress_callback = ProgressionCallback()
        params = {'epochs': epochs, 'steps': steps_per_epoch}
        progress_callback.set_params(params)
        callbacks = [progress_callback] + callbacks

    for c in callbacks:
        c.on_train_begin(model)

    for epoch in range(initial_epoch, epochs):
        if model.__dict__.get('stop_training', False):
            break
        epoch_logs = {}
        epoch_logs['epoch'] = epoch
        
        epoch_begin_time = timeit.default_timer()

        for c in callbacks:
            c.on_epoch_begin(epoch, epoch_logs)

        # Train an epoch
        if train_generator is not None:
            with _set_training_mode(model, True):
                train_predictions_list = []
                train_labels_list = []
                train_indices_list = []

                for batch_id, (x_train, y_train, x_indices) in enumerate(train_generator):
                    batch_begin_time = timeit.default_timer()

                    x_train, y_train = x_train.to(device), y_train.to(device)
                    shape = x_train.shape
                    is_n_crop = len(shape) == 5
                    if is_n_crop:
                        batch_size, num_crops, c, h, w = shape
                        x_train = x_train.view(-1, c, h, w)
                    else:
                        batch_size, c, h, w = shape
                    batch_logs = {'size': len(x_train), 'batch': batch_id}

                    for c in callbacks:
                        c.on_batch_begin(batch=batch_id, logs=batch_logs)

                    optimizer.zero_grad()

                    outputs = model(x_train)
                    if is_n_crop:
                        outputs = outputs.view(batch_size, num_crops, -1).mean(1)
                    train_predictions_list.append(outputs.cpu().detach().numpy())
                    train_labels_list.append(y_train.cpu().detach().numpy())
                    train_indices_list.append(x_indices.cpu().detach().numpy())
                    loss = loss_function(outputs, y_train)
                    loss.backward()
                    optimizer.step()
                    batch_total_time = timeit.default_timer() - batch_begin_time

                    # Update logs
                    for m in metrics:
                        batch_logs[m.__name__] = float(m(outputs, y_train))
                    batch_logs['loss'] = loss.item()
                    batch_logs['batch_begin_time'] = batch_begin_time
                    batch_logs['batch_total_time'] = batch_total_time

                    for c in callbacks:
                        c.on_batch_end(batch=batch_id, logs=batch_logs)
                epoch_logs['train_predictions'] = np.concatenate(train_predictions_list, axis=0)
                epoch_logs['train_labels'] = np.concatenate(train_labels_list, axis=0)
                epoch_logs['train_indices'] = np.concatenate(train_indices_list, axis=0)


        # Validate
        if valid_generator is not None:
            with _set_training_mode(model, False):
                val_predictions_list = []
                val_labels_list = []
                val_indices_list = []
                val = defaultdict(float)
                seen = 0
                for x_valid, y_valid, x_indices in valid_generator:
                    shape = x_valid.shape
                    is_n_crop = len(shape) == 5
                    if is_n_crop:
                        batch_size, num_crops, c, h, w = shape
                        x_valid = x_valid.view(-1, c, h, w)
                    else:
                        batch_size, c, h, w = shape
                    x_valid, y_valid = x_valid.to(device), y_valid.to(device)
                    seen += batch_size
                    outputs = model(x_valid)
                    if is_n_crop:
                        outputs = outputs.view(batch_size, num_crops, -1).mean(1)
                    val_predictions_list.append(outputs.cpu().detach().numpy())
                    val_labels_list.append(y_valid.cpu().detach().numpy())
                    val_indices_list.append(x_indices.cpu().detach().numpy())
                    val['loss'] += loss_function(outputs, y_valid).item() * len(x_valid)
                    for m in metrics:
                        val[m.__name__] += float(m(outputs, y_valid)) * len(x_valid)
                for k in val:
                    epoch_logs['val_' + k] = val[k] / seen
                epoch_logs['val_predictions'] = np.concatenate(val_predictions_list, axis=0)
                epoch_logs['val_labels'] = np.concatenate(val_labels_list, axis=0)
                epoch_logs['val_indices'] = np.concatenate(val_indices_list, axis=0)
                
        # test
        if test_generator is not None:
            with _set_training_mode(model, False):
                test_predictions_list = []
                test_labels_list = []
                test_indices_list = []
                test = defaultdict(float)
                seen = 0
                for x_test, y_test, x_indices in test_generator:
                    shape = x_test.shape
                    is_n_crop = len(shape) == 5
                    if is_n_crop:
                        batch_size, num_crops, c, h, w = shape
                        x_test = x_test.view(-1, c, h, w)
                    else:
                        batch_size, c, h, w = shape
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    seen += batch_size
                    outputs = model(x_test)
                    if is_n_crop:
                        outputs = outputs.view(batch_size, num_crops, -1).mean(1)
                    test_predictions_list.append(outputs.cpu().detach().numpy())
                    test_labels_list.append(y_test.cpu().detach().numpy())
                    test_indices_list.append(x_indices.cpu().detach().numpy())
                    test['loss'] += loss_function(outputs, y_test).item() * len(x_test)
                    for m in metrics:
                        test[m.__name__] += float(m(outputs, y_test)) * len(x_test)
                for k in test:
                    epoch_logs['test_' + k] = test[k] / seen
                epoch_logs['test_predictions'] = np.concatenate(test_predictions_list, axis=0)
                epoch_logs['test_labels'] = np.concatenate(test_labels_list, axis=0)
                epoch_logs['test_indices'] = np.concatenate(test_indices_list, axis=0)

                
        epoch_total_time = timeit.default_timer() - epoch_begin_time
        epoch_logs['time'] = epoch_total_time
        epoch_logs['epoch_begin_time'] = epoch_begin_time
                
        for c in callbacks:
            c.on_epoch_end(epoch, epoch_logs)

        
        epoch_summary_string = 'End of epoch {}'.format(epoch)
        if train_generator is not None:
            epoch_summary_string += ', loss={}'.format(epoch_logs['loss'])
        if valid_generator is not None:
            epoch_summary_string += ', val_loss={}'.format(epoch_logs['val_loss'])
        if test_generator is not None:
            epoch_summary_string += ', test_loss={}'.format(epoch_logs['test_loss'])

        if scheduler is not None:
            scheduler.step()
        logger.info(epoch_summary_string)

    for c in callbacks:
        c.on_train_end(model)
        
@gin.configurable
def training_loop(model, loss_function, metrics, optimizer, scheduler, meta_data, config,
                  save_path, steps_per_epoch, n_epochs,
                  hyper_optim=None,
                  train=None, valid=None, test=None, validation_per_epoch=1,
                  data_loader=None, meta_data_loader=None,
                  test_steps=None, validation_steps=None,
                  use_gpu = False, device_numbers = [0], pretrained = False, 
                  BIRADS_pretrained_weights_path=None,
                  pretrained_weight_paths = None, 
                  custom_callbacks=[], checkpoint_monitor="val_acc", use_tb=False, reload=True,
                  save_freq=1, save_history_every_k_examples=-1,
                  fb_method=False,
                  target_indice=None,
                  grad_norm_penalty=None,
                  penalty_on_columns=False,
                  task_name=None,
                  grad_norm_penalty_var='vanilla',
                  num_of_view_to_change=1,
                  suppress_nan_labels_in_loss=True
                  ):

        
    callbacks = [BaseLogger()] + list(custom_callbacks)

    if pretrained and pretrained_weight_paths is not None:
        _load_pretrained_branch(model, pretrained_weight_paths)
    elif BIRADS_pretrained_weights_path is not None:
        _load_BIRADS_pretrained(model, BIRADS_pretrained_weights_path)
        
    if reload:
        H, epoch_start = _reload(model, optimizer, save_path, custom_callbacks)
    else:
        save_weights(model, optimizer, os.path.join(save_path, "init_weights.pt"))

        history_csv_path, history_pkl_path = os.path.join(save_path, "history.csv"), os.path.join(save_path,
                                                                                                  "history.pkl")
        logger.info("Removing {} and {}".format(history_pkl_path, history_csv_path))
        os.system("rm " + history_pkl_path)
        os.system("rm " + history_csv_path)
        H, epoch_start = {}, 0

    H_batch = {}
    
    if train is not None:
        default_callbacks = _construct_default_callbacks(model, optimizer, H, H_batch, save_path, checkpoint_monitor,
                                              save_freq, custom_callbacks, use_tb,
                                              save_history_every_k_examples)
        
    else:
        # If train is None, then evaluation. However, training_loop is not designed for this purpose.
        # You should use evaluation_loop when setting train to None
        default_callbacks = _construct_default_eval_callbacks(H, H_batch, save_path,
                                              save_history_every_k_examples)
    
        
    callbacks = callbacks + default_callbacks
    
    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(save_path)
        clbk.set_model(model, ignore=False)  # TODO: Remove this trick
        clbk.set_optimizer(optimizer)
        clbk.set_meta_data(meta_data)
        clbk.set_config(config)
        clbk.set_dataloader(None)

    is_multi_gpu = False

    if use_gpu and torch.cuda.is_available(): 
        is_multi_gpu = len(device_numbers) > 1
        base_device = torch.device("cuda:{}".format(device_numbers[0]))

        if is_multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=device_numbers)

        logger.info("Sending model to {}".format(base_device))
        model.to(base_device)
        optimizer.load_state_dict(optimizer.state_dict()) # Hack to use right device for optimizer, according to https://github.com/pytorch/pytorch/issues/8741
    
        
    _loop(
        model, test, valid, train, optimizer, scheduler, loss_function,
        initial_epoch=epoch_start,
        epochs=n_epochs,
        callbacks=callbacks,
        metrics=metrics,
        device=base_device,
        steps_per_epoch=steps_per_epoch,
        suppress_nan_labels_in_loss=suppress_nan_labels_in_loss
    )


@gin.configurable
def _construct_default_eval_callbacks(H, H_batch, save_path, 
                                      save_history_every_k_examples,
                                      extended_foldername = 'eval_history_batch'
                                      ):
    
    history_batch = os.path.join(save_path, extended_foldername)
    if not os.path.exists(history_batch):
        os.mkdir(history_batch)

    callbacks = []
    callbacks.append(LambdaCallback(on_epoch_end=partial(_append_to_history_csv, H=H), 
                                    on_batch_end=partial(_append_to_history_csv_batch, H=H_batch)
                               )
    )

    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_history_csv, 
                                                        save_path=save_path, 
                                                        H=H, mode='eval')))

    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_history_csv_batch, 
                                                        save_path=history_batch, 
                                                        H=H_batch)))

    callbacks.append(History(save_every_k_examples=save_history_every_k_examples, mode='eval'))

    return callbacks

@gin.configurable
def evaluation_loop(model, loss_function, metrics, optimizer, 
                   meta_data, config, 
                   save_path, pretrained_model_name,
                   test=None,  test_steps=None,
                   data_loader=None, 
                   use_gpu = False, device_numbers = [0], 
                   custom_callbacks=[], 
                   n_epochs=1, 
                   return_test_pred=False,
                   target_indice=None,
                   save_history_every_k_examples=-1, 
                   suppress_nan_labels_in_loss=True
                  ):

    callbacks = [BaseLogger()] + list(custom_callbacks)
    
    _load_pretrained_model(model, save_path, pretrained_model_name)

    history_csv_path, history_pkl_path = os.path.join(save_path, "eval_history.csv"), \
        os.path.join(save_path, "eval_history.pkl")

    logger.info("Removing {} and {}".format(history_pkl_path, history_csv_path))
    os.system("rm " + history_pkl_path)
    os.system("rm " + history_csv_path)
    H, epoch_start = {}, 0

    H_batch = {}

    callbacks += _construct_default_eval_callbacks(H, 
        H_batch, 
        save_path, 
        save_history_every_k_examples)

    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(save_path)
        clbk.set_model(model, ignore=False)  # TODO: Remove this trick
        clbk.set_meta_data(meta_data)
        clbk.set_config(config)
        clbk.set_dataloader(None)

    if use_gpu and torch.cuda.is_available(): 
        is_multi_gpu = len(device_numbers) > 1
        base_device = torch.device("cuda:{}".format(device_numbers[0]))

        if is_multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=device_numbers)

        logger.info("Sending model to {}".format(base_device))
        model.to(base_device)

    _loop(
        model, test, None, None, None, loss_function, 
        initial_epoch=epoch_start,
        epochs=n_epochs,
        callbacks=callbacks,
        metrics=metrics,
        device=base_device,
        suppress_nan_labels_in_loss=suppress_nan_labels_in_loss
    )
