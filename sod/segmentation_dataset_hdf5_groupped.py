import random
from turtle import pos

import h5py

import torch
from torchvision import transforms as trt
from torchvision.transforms import functional as trf
from torch.utils.data import Dataset

class SegmentationDatasetHDF5Groupped(Dataset):
    def __init__(self, dataset_path: str, crop_width: int, crop_height: int, augment: bool = False) -> None:
        super().__init__()
        self._dataset_path = dataset_path
        with h5py.File(self._dataset_path, mode='r') as file:
            self._size, _, self._width, self._height = file['/images'].shape
        self._crop_width = crop_width
        self._crop_height = crop_height
        self._augment = augment
        self._color_augment = trt.RandomChoice([
            trt.ColorJitter(),
            trt.GaussianBlur(kernel_size=5),
            trt.Grayscale(num_output_channels=3),
            trt.RandomInvert(),
            trt.RandomSolarize(threshold=100),
            trt.RandomPosterize(bits=2),
            trt.RandomEqualize(),
            trt.RandomAutocontrast()
        ])

    def __len__(self):
        return self._size

    def _crop(self):
        if self._augment:
            pos_x = random.randrange(0, self._width - self._crop_width)
            pos_y = random.randrange(0, self._height - self._crop_height)
        else:
            pos_x = self._width // 2 - self._crop_width // 2
            pos_y = self._height // 2 - self._crop_height // 2
        return pos_x, pos_y, self._crop_width, self._crop_height

    def __getitem__(self, index: int) -> torch.Tensor:
        # get random crop position and size
        pos_x, pos_y, width, height = self._crop()

        # get image and mask from hdf5
        with h5py.File(self._dataset_path, mode='r') as file:
            img = torch.as_tensor(file['/images'][index, :, pos_x:pos_x+width, pos_y:pos_y+height], dtype=torch.uint8)
            msk = torch.as_tensor(file['/masks'][index, :, pos_x:pos_x+width, pos_y:pos_y+height], dtype=torch.float32)
        
        # perform data augmentation on image
        if self._augment:
            img = self._color_augment(img)
        
        # normalize image between 0 and 1
        img = img / 255.0

        # synchronous random rotation
        if self._augment:
            angle = trt.RandomRotation.get_params([45, 90, 135, 180, 225, 270, 315])
            img = trf.rotate(img, angle)
            msk = trf.rotate(msk, angle)

        return img, msk