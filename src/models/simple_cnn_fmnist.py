'''
Model created by @AbhirajHinge at https://github.com/AbhirajHinge/CNN-with-Fashion-MNIST-dataset
'''
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock

import gin

@gin.configurable
class QuarterCNN(nn.Module):
    def __init__(self, 
                 out_channels, 
                 kernel_size, 
                 with_classifier, 
                 target_quarter_ind
                 ):
        super(QuarterCNN,self).__init__()

        self.target_quarter_ind = target_quarter_ind
        self.with_classifier = with_classifier
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        padding = int((self.kernel_size-1)/2)

        self.conv1 = nn.Conv2d(in_channels=1,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=1,
                              padding=padding)

        self.conv2 = nn.Conv2d(in_channels=self.out_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=1,
                              padding=padding)

        self.conv3 = nn.Conv2d(in_channels=self.out_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=1,
                              padding=padding)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        if self.with_classifier: 
            self.fc1 = nn.Linear(self.out_channels*7*7,  self.out_channels*7*7)
            self.fc2 = nn.Linear(self.out_channels*7*7,  10)
        
        self.initialize()
        
    def forward(self, x):
        target_quarter = x[:, self.target_quarter_ind, :]
        out = F.elu(self.conv1(target_quarter))
        out = F.elu(self.conv2(out))
        out = F.elu(self.conv3(out))
        
        out = self.maxpool1(out)
        
        out = out.view(out.size(0),-1)

        if self.with_classifier:
            out = self.fc1(out)
            out = F.relu(out)
            out = self.fc2(out)

        return out
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_normal_(m.weight)

@gin.configurable
class QuarterResnet(nn.Module):
    def __init__(self, 
                 out_channels, 
                 kernel_size, 
                 with_classifier, 
                 target_quarter_ind
                 ):
        super(QuarterCNN,self).__init__()

        self.target_quarter_ind = target_quarter_ind
        self.with_classifier = with_classifier

        self.resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)


        if self.with_classifier: 
            self.fc1 = nn.Linear(self.out_channels*7*7,  self.out_channels*7*7)
            self.fc2 = nn.Linear(self.out_channels*7*7,  10)
        
        self.initialize()
        
    def forward(self, x):
        target_quarter = x[:, self.target_quarter_ind, :]
        out = F.elu(self.conv1(target_quarter))
        out = F.elu(self.conv2(out))
        out = F.elu(self.conv3(out))
        
        out = self.maxpool1(out)
        
        out = out.view(out.size(0),-1)

        if self.with_classifier:
            out = self.fc1(out)
            out = F.relu(out)
            out = self.fc2(out)

        return out
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_normal_(m.weight)

                


@gin.configurable
class JointQuarterCNN(nn.Module):
    def __init__(self, 
                 out_channels, 
                 kernel_size,  
                 viewdropping,
                 inverse_dropping_rate,
                 device_numbers,
                 ):

        super(JointQuarterCNN,self).__init__()

        self.out_channels = out_channels
        
        self.upper_left = QuarterCNN(out_channels, kernel_size, False, target_quarter_ind = 0) 
        self.upper_right = QuarterCNN(out_channels, kernel_size, False, target_quarter_ind = 1)
        self.lower_left = QuarterCNN(out_channels, kernel_size, False, target_quarter_ind = 2)
        self.lower_right = QuarterCNN(out_channels, kernel_size, False, target_quarter_ind = 3)

        self.fc1= nn.Linear(self.out_channels*4*7*7,  self.out_channels*4*7*7)
        self.fc2= nn.Linear(self.out_channels*4*7*7,  10)

        self.viewdropping = viewdropping
        self.inverse_dropping_rate = inverse_dropping_rate

        self.base_device = torch.device("cuda:{}".format(device_numbers[0]))

        self.initialize()

    def dropping_weights_generator(self):
        '''
        Generate a array with length == layer after fusion (cancatenation) : 
        In the current setting: 8*7*7*4
        '''
        #TODO: random seed for sampling dropout 
        if not self.training:
            weights = numpy.array([[1]*(self.out_channels*7*7) for rate in self.inverse_dropping_rate]).flatten()
        else:
            weights = numpy.array([[numpy.random.binomial(1, rate)]*self.out_channels *7*7 for rate in self.inverse_dropping_rate]).flatten()
            inverse_probs = numpy.array([[1/rate]*(self.out_channels*7*7) for rate in self.inverse_dropping_rate]).flatten()
            weights = weights*inverse_probs

        return torch.from_numpy(weights).float().to(self.base_device)
             
        
    def forward(self, x):

        h_upper_left = self.upper_left(x)
        h_upper_right = self.upper_right(x)
        h_lower_left = self.lower_left(x)
        h_lower_right = self.lower_right(x)

        h = torch.cat([h_upper_left, h_upper_right, h_lower_left, h_lower_right], dim=1)
        if self.viewdropping:
            h = h*self.dropping_weights_generator()
        
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)

        return h
        
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_normal_(m.weight)
    











        