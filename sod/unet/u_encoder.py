from typing import List, Tuple
import torch
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList

from .u_block import UBlock

class UEncoder(Module):
    """Represent the encoding part of the UNet model"""
    def __init__(self, input_channel=3, channels=(16, 32, 64)):
        """Create an encoder from set of successive channels

        Args:
            channels (tuple, optional): Set of successive channel sizes for the UNet encoding
                architecture. Defaults to (16, 32, 64).
        """
        super().__init__()
        # concatenante input channels to encoding channels
        channels = (input_channel,) + channels
        # store the encoder blocks and maxpooling layer
        self.encoding_u_blocks = ModuleList([
            UBlock(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])
        self.pool = MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Computing formula of the layer

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Formula result
            List[torch.Tensor]: Encoded features for each block
        """
        # initialize an empty list to store the intermediate outputs
        encoded_features = []
        # loop through the encoder blocks
        for block in self.encoding_u_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            encoded_features.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return encoded_features[-1], encoded_features[:-1]
