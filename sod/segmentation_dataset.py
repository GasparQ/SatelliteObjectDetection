from typing import List

import cv2

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as trf

from .config import INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH

class SegmentationDataset(Dataset):
    """
    Allow to load an image segmentation dataset with random crop
    """
    def __init__(self, image_paths: List[str], mask_paths: List[str]):
        """Image Segmentation datasets are formed with images and masks

        Args:
            imagePaths (List[str]): List of the pathes to the images of the dataset
            maskPaths (List[str]): List of the pathes to the masks of the dataset
        """
        # store the image and mask filepaths, and augmentation
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        """Compute the length of the dataset

        Returns:
            int: Number of images in the dataset
        """
        # return the number of total samples contained in the dataset
        return len(self.image_paths)
        
    def __getitem__(self, idx: int):
        """Pick one item at a specific index in the dataset

        Args:
            idx (int): Item index to pick

        Returns:
            torch.Tensor: Random crop of the image of the dataset at index idx
            torch.Tensor: Corresponding mask of the cropped image
        """
        # grab the image path from the current index
        imagePath = self.image_paths[idx]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0)

        # convert into PIL Image
        image = trf.to_pil_image(image)
        mask = trf.to_pil_image(mask)

        # perform synchronous random crop
        i, j, height, width = transforms.RandomCrop.get_params(image, (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH))
        image = trf.crop(image, i, j, height, width)
        mask = trf.crop(mask, i, j, height, width)

        # convert into tensor
        image = trf.to_tensor(image)
        mask = trf.to_tensor(mask)

        return (image, mask)