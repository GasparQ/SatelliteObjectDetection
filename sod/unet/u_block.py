import torch
from torch.nn import Conv2d
from torch.nn import Module
from torch.nn import ReLU

class UBlock(Module):
    """Convolution block of the UNet model"""
    def __init__(self, inChannels: int, outChannels: int):
        """Create convolution block from in and out channels

        Args:
            inChannels (int): Number of input channels
            outChannels (int): Number of output channels
        """
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computing formula of the layer

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Formula result
        """
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))
