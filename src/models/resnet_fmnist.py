import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import gin

from .layers import BasicBlock

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        #self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

@gin.configurable
class QuarterResnet(ResNet):
    def __init__(self, 
                 with_classifier, 
                 target_quarter_ind
                 ):
        super(QuarterResnet, self).__init__(BasicBlock, [2, 2, 2], num_classes=10)

        self.target_quarter_ind = target_quarter_ind
        self.with_classifier = with_classifier

        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)


        if self.with_classifier: 
            self.fc = self.nn.Linear(128 * BasicBlock.expansion, 10)

    def forward(self, x_, target_quarter = None ):
        if target_quarter is None:
            target_quarter = x_[:, self.target_quarter_ind, :]
        x = self.conv1(target_quarter)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.with_classifier:
            x = self.fc(x)

        return x
                
@gin.configurable
class JointQuarterResnet(nn.Module):
    def __init__(self, 
                 viewdropping,
                 inverse_dropping_rate,
                 device_numbers,
                 ):

        super(JointQuarterResnet,self).__init__()
        
        self.upper_left = QuarterResnet(False, target_quarter_ind = 0) 
        self.upper_right = QuarterResnet(False, target_quarter_ind = 1)
        self.lower_left = QuarterResnet(False, target_quarter_ind = 2)
        self.lower_right = QuarterResnet(False, target_quarter_ind = 3)

        self.fc1_upper_left= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc1_upper_right= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc1_lower_left= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc1_lower_right= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc2= nn.Linear(128*BasicBlock.expansion*4,  10)

        self.viewdropping = viewdropping
        self.inverse_dropping_rate = inverse_dropping_rate

        self.base_device = torch.device("cuda:{}".format(device_numbers[0]))

    def dropping_weights_generator(self):
        '''
        Generate a array with length == layer after fusion (cancatenation) : 
        In the current setting: 8*7*7*4
        '''
        #TODO: random seed for sampling dropout 
        if not self.training:
            weights = numpy.array([[1]*(128*BasicBlock.expansion) for rate in self.inverse_dropping_rate]).flatten()
        else:
            weights = numpy.array([[numpy.random.binomial(1, rate)]*128*BasicBlock.expansion for rate in self.inverse_dropping_rate]).flatten()
            inverse_probs = numpy.array([[1/rate]*(128*BasicBlock.expansion) for rate in self.inverse_dropping_rate]).flatten()
            weights = weights*inverse_probs

        return torch.from_numpy(weights).float().to(self.base_device)
             
        
    def forward(self, x):

        h_upper_left = self.upper_left(x)
        h_upper_right = self.upper_right(x)
        h_lower_left = self.lower_left(x)
        h_lower_right = self.lower_right(x)

        h = self.fc1_upper_left(h_upper_left) + self.fc1_upper_right(h_upper_right) + self.fc1_lower_left(h_lower_left) + self.fc1_lower_right(h_lower_right)

        #h = torch.cat([h_upper_left, h_upper_right, h_lower_left, h_lower_right], dim=1)
        if self.viewdropping:
            h = h*self.dropping_weights_generator()
        
        #h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)

        return h
      
@gin.configurable
class JointQuarterResnet_Shared(nn.Module):
    def __init__(self, 
                 viewdropping,
                 inverse_dropping_rate,
                 device_numbers,
                 ):

        super(JointQuarterResnet_Shared,self).__init__()
        
        self.sharedresnet = QuarterResnet(False, target_quarter_ind = 0) 

        self.fc1_upper_left= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc1_upper_right= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc1_lower_left= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc1_lower_right= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc2 = nn.Linear(128*BasicBlock.expansion*4,  10)

        self.viewdropping = viewdropping
        self.inverse_dropping_rate = inverse_dropping_rate

        self.base_device = torch.device("cuda:{}".format(device_numbers[0]))

    def dropping_weights_generator(self):
        '''
        Generate a array with length == layer after fusion (cancatenation) : 
        In the current setting: 8*7*7*4
        '''
        #TODO: random seed for sampling dropout 
        if not self.training:
            weights = numpy.array([[1]*(128*BasicBlock.expansion) for rate in self.inverse_dropping_rate]).flatten()
        else:
            weights = numpy.array([[numpy.random.binomial(1, rate)]*128*BasicBlock.expansion for rate in self.inverse_dropping_rate]).flatten()
            inverse_probs = numpy.array([[1/rate]*(128*BasicBlock.expansion) for rate in self.inverse_dropping_rate]).flatten()
            weights = weights*inverse_probs

        return torch.from_numpy(weights).float().to(self.base_device)
             
        
    def forward(self, x):

        h_upper_left = self.sharedresnet(None, x[:, 0, :])
        h_upper_right = self.sharedresnet(None, x[:, 1, :])
        h_lower_left = self.sharedresnet(None, x[:, 2, :])
        h_lower_right = self.sharedresnet(None, x[:, 3, :])

        h = self.fc1_upper_left(h_upper_left) + self.fc1_upper_right(h_upper_right) + self.fc1_lower_left(h_lower_left) + self.fc1_lower_right(h_lower_right)

        #h = torch.cat([h_upper_left, h_upper_right, h_lower_left, h_lower_right], dim=1)
        if self.viewdropping:
            h = h*self.dropping_weights_generator()
        
        #h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)

        return h

@gin.configurable
class BlendQuarterResnet(nn.Module):
    def __init__(self, 
                 viewdropping,
                 inverse_dropping_rate,
                 device_numbers,
                 ):

        super(BlendQuarterResnet,self).__init__()
        
        self.upper_left = QuarterResnet(False, target_quarter_ind = 0) 
        self.upper_right = QuarterResnet(False, target_quarter_ind = 1)
        self.lower_left = QuarterResnet(False, target_quarter_ind = 2)
        self.lower_right = QuarterResnet(False, target_quarter_ind = 3)

        self.fc1_upper_left= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc1_upper_right= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc1_lower_left= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc1_lower_right= nn.Linear(128*BasicBlock.expansion, 128*BasicBlock.expansion*4)
        self.fc2= nn.Linear(128*BasicBlock.expansion*4,  10)

        self.fc2_upper_left = nn.Linear(128*BasicBlock.expansion,  10)
        self.fc2_upper_right = nn.Linear(128*BasicBlock.expansion,  10)
        self.fc2_lower_left = nn.Linear(128*BasicBlock.expansion,  10)
        self.fc2_lower_right = nn.Linear(128*BasicBlock.expansion,  10)

        self.viewdropping = viewdropping
        self.inverse_dropping_rate = inverse_dropping_rate

        self.base_device = torch.device("cuda:{}".format(device_numbers[0]))
             
        
    def forward(self, x):

        h_upper_left = self.upper_left(x)
        h_upper_right = self.upper_right(x)
        h_lower_left = self.lower_left(x)
        h_lower_right = self.lower_right(x)

        h = self.fc1_upper_left(h_upper_left) + self.fc1_upper_right(h_upper_right) + self.fc1_lower_left(h_lower_left) + self.fc1_lower_right(h_lower_right)
        
        h = F.relu(h)
        h = self.fc2(h)

        h_upper_left = self.fc2_upper_left(h_upper_left)
        h_upper_right = self.fc2_upper_left(h_upper_right)
        h_lower_left = self.fc2_upper_left(h_lower_left)
        h_lower_right = self.fc2_upper_left(h_lower_right)


        return h, h_upper_left, h_upper_right, h_lower_left, h_lower_right 






        