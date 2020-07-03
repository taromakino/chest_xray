# encoding: utf-8

"""
The main CheXNet model implementation.
"""

import torch
import torch.nn as nn
import torchvision
import gin

@gin.configurable
class CheXNet(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, imagenet_pretrained=True, multiclass=False):
        super(CheXNet, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=imagenet_pretrained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Softmax() if multiclass else nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x