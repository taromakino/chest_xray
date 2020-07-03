# -*- coding: utf-8 -*-
"""
Models used in the project
"""
from .simple_cnn_fmnist import QuarterCNN, JointQuarterCNN
from .resnet_fmnist import QuarterResnet, JointQuarterResnet, JointQuarterResnet_Shared, BlendQuarterResnet
from .chexnet import CheXNet
from .classifier import Classifier
#comment out while using local machine  
# from .breast import (AllViewModel, SingleViewModel, EnsembleModel,
# BlendedModel, SmallerSingleViewModel,
# TiedViewModel, MidFusedModel,
# SmallerBlendedModel, ModDropModel,
# ViewPoolingModel)
# from .breast_view_pooling import GeneralViewPoolingModel, MeanViewPoolingModel, ViewMergeModel
# from .breast_deepfused import DeepFusionModel, DeepViewFusionModel

# from .mvcnn import SVCNN, MVCNN
