"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*list(vgg16.features.children()))
        self.conv1AF = nn.Conv2d(512,4096,1)
        self.conv2AF = nn.Conv2d(4096,4096,1)
        self.conv3AF = nn.Conv2d(4096,num_classes,1)
        self.upSample = nn.Upsample(size=(240,240),mode='bilinear')

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        x = self.features(x)
        x = self.conv3AF(self.conv2AF(self.conv1AF(x)))
        x = self.upSample(x)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
