from typing import List, Tuple
from torch.nn import ConvTranspose2d
from torch.nn import Module
from torch.nn import ModuleList
from torchvision.transforms import CenterCrop
import torch

from .u_block import UBlock

class UDecoder(Module):
    """Decoder part of the UNet architecture"""
    def __init__(self, channels: Tuple[int]=(64, 32, 16)):
        """Create a decoder from a set of successive channels
        Note: The number of decoding channels must be equals
            to the number of encoding channels

        Args:
            channels (Tuple[int], optional): Set of successive channels
                for decoding (must be the same than for encoding). Defaults to (64, 32, 16).
        """
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList([
            ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
            for i in range(len(channels) - 1)
        ])
        self.dec_blocks = ModuleList([
            UBlock(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x: torch.Tensor, encoded_features: List[torch.Tensor]) -> torch.Tensor:
        """Computing formula of the layer

        Args:
            x (torch.Tensor): Input data
            encoded_features (List[torch.Tensor]): All the other encoded features from the encoder

        Returns:
            torch.Tensor: Formula result
        """
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            enc_feat = self.crop(encoded_features[i], x)
            x = torch.cat([x, enc_feat], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x

    def crop(self, encoded_features: List[torch.Tensor], x: torch.Tensor) -> List[torch.Tensor]:
        """Perform a Center Crop of encoded features to fit the input shape

        Args:
            encoded_features (List[torch.Tensor]): All the other encoded features from the encoder
            x (torch.Tensor): Input data

        Returns:
            List[torch.Tensor]: Center Cropped encoded features
        """
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encoded_features = CenterCrop([H, W])(encoded_features)
        # return the cropped features
        return encoded_features
