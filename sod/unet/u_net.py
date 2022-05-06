from typing import Tuple
import torch
from torch.nn import Conv2d
from torch.nn import Module
from torch.nn import functional as F

from ..config import INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH

from .u_encoder import UEncoder
from .u_decoder import UDecoder

class UNet(Module):
    """UNet neural network architecture"""
    def __init__(self,
                 input_channel: int=3,
                 encoding_channels: Tuple[int]=(16, 32, 64),
                 output_channel=1,
                 retain_dim=True,
                 out_size=(INPUT_IMAGE_HEIGHT,  INPUT_IMAGE_WIDTH)):
        """Create a Unet from input, output and encoding channels
        Note: decoding channels will be the mirror of encoding channels

        Args:
            input_channel (int, optional): Number of channels as input. Defaults to 3 (RGB).
            encoding_channels (Tuple[int], optional): Set of successive channels for encoding.
                Defaults to (16, 32, 64).
            output_channel (int, optional): Number of channels as output. Defaults to 1.
            retain_dim (bool, optional): Weather to resize or not output dimension to out_size.
                Defaults to True.
            out_size (tuple, optional): Size of the output if retain_dim specified.
                Defaults to (INPUT_IMAGE_HEIGHT,  INPUT_IMAGE_WIDTH).
        """
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = UEncoder(input_channel, encoding_channels)
        self.decoder = UDecoder(encoding_channels[::-1])
        # initialize the regression head and store the class variables
        self.head = Conv2d(encoding_channels[0], output_channel, 1)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computing formula of the layer

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Formula result
        """
        # grab the features from the encoder
        x, encoded_features = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        x = self.decoder(x, encoded_features[::-1])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        output_map = self.head(x)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retain_dim:
            output_map = F.interpolate(output_map, self.out_size)
        # return the segmentation map
        return output_map