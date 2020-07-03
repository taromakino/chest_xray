#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer script. Example run command: bin/train.py save_to_folder configs/cnn.gin.
"""
# A part of hack to mitigate cv2 error on skynet
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras

import gin
from gin.config import _CONFIG
import torch
import logging
from functools import partial
logger = logging.getLogger(__name__)

from src.data.data_chexnet_covid import get_chexnet_covid
from src import models
from src.training_loop import training_loop, evaluation_loop
from src.callbacks import get_callback
from src.utils import summary, acc_fmnist, gin_wrap, create_optimizer, acc_chexnet_covid

@gin.configurable
def train(save_path,
          model,
          lr_splitting_by=None,
          lrs=None, wd=0, lr=0.1,
          batch_size=128,
          n_epochs=100,
          weights=None,
          fb_method=False,
          callbacks=[],
          optimizer='sgd',
          scheduler=None,
          freeze_all_but_this_layer=None,
          mode='train'):
    # Create dynamically dataset generators
    train, valid, test, meta_data = get_chexnet_covid(batch_size=batch_size)

    # Create dynamically model
    model = models.__dict__[model]()
    summary(model)
    
    loss_function = torch.nn.BCELoss()
    
    if freeze_all_but_this_layer is not None:
        # First freeze all layers
        logger.info("Freezing all layers")
        for i, parameter in enumerate(model.parameters()):
            parameter.requires_grad = False
            
        # Unfreeze layers that matches
        
        for i, (name, parameter) in enumerate(model.named_parameters()):
            if name.startswith(freeze_all_but_this_layer):
                parameter.requires_grad = True
                logger.info("Unfreezing {}: {}".format(name, parameter.shape))
    
    if optimizer=='sgd':            
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    if lr_splitting_by is not None:
        optimizer, _ = create_optimizer(optimizer, model, lr_splitting_by, lrs)

    # Create dynamically callbacks
    callbacks_constructed = []
    for name in callbacks:
        clbk = get_callback(name, verbose=0)
        if clbk is not None:
            print(name)
            callbacks_constructed.append(clbk)

    # Pass everything to the training loop
    if train is not None:
        steps_per_epoch = len(train) 
    else:
        steps_per_epoch = None

    target_indice = None
    if fb_method:
        target_indice = weights.index(1) if 1 in weights else 0
    elif weights is not None:
        target_indice = 0

    if mode == 'train':
        assert train is not None, "please provide train data"
        assert valid is not None, "please provide validation data"
        training_loop(model=model, optimizer=optimizer, scheduler=scheduler, loss_function=loss_function,
                      metrics=[acc_chexnet_covid], train=train, valid=valid, test=test, meta_data=meta_data,
                      steps_per_epoch=steps_per_epoch, n_epochs=n_epochs, save_path=save_path, config=_CONFIG,
                      use_tb=True, custom_callbacks=callbacks_constructed,
                      fb_method=fb_method,
                      target_indice=target_indice,
                      )
    else:
        assert test is not None, "please provide test data for evaluation"
        evaluation_loop(model=model, optimizer=optimizer, loss_function=loss_function, metrics=[acc_chexnet_covid],
                      test=test, meta_data=meta_data,
                      save_path=save_path, config=_CONFIG,
                      custom_callbacks=callbacks_constructed,
                      target_indice=target_indice,
                      )


if __name__ == "__main__":
    gin_wrap(train)
