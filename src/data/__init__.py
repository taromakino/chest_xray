# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid.
"""
import os
import gin
from functools import partial
import logging
import numpy as np

#comment out while using local machine  
#from .breast_data_utilities.breast_data import data_single_view, data_single_view_with_segmentation


logger = logging.getLogger(__name__)
