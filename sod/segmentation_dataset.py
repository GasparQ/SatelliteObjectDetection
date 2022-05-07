from functools import partial
from typing import List, Tuple

import tqdm

import numpy as np

import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as trf
from torch.multiprocessing import Pool, Lock

# mutex to perform the send to device operation synchronously
# device_send_mutex = Lock()

def sample_image_random_crops(img_mask_pathes: Tuple[str, str],
                              crop_size: Tuple[int, int],
                              sample_factor: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Perform successive random crop on image and mask

    Args:
        img_mask_pathes (Tuple[str, str]): Pathes to both image and mask (in that order)
        crop_size (Tuple[int, int]): Height and width (in that order) for the crop size
        sample_factor (int): Number of successive crop to perform

    Returns:
        List[Tuple[torch.Tensor, torch.Tensor]]: List of crop pairs on image and mask (in that order)
    """ 
    image_path, mask_path = img_mask_pathes

    # load the image from disk, swap its channels from BGR to RGB,
    # and read the associated mask from disk in grayscale mode
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path)

    # convert into PIL Image
    image = trf.to_pil_image(image)
    # mask = trf.to_pil_image(mask)

    # convert into binary mask
    #   - convert rgb into greyscale image (max applied)
    #   - clip values between 0 and 1
    #   - add channel as first dimension
    mask = np.expand_dims(mask.max(-1).clip(0, 1), 0)

    # convert to uint8 tensor
    mask = torch.as_tensor(mask, dtype=torch.uint8)

    random_crops = []
    for i in range(sample_factor):
        # get random crop indices, height and width
        i, j, height, width = transforms.RandomCrop.get_params(image, crop_size)

        # perform synchronous crop on image and mask
        cropped_img = trf.to_tensor(trf.crop(image, i, j, height, width))
        cropped_mask = trf.crop(mask, i, j, height, width).float()

        # append to sample list
        random_crops.append((cropped_img, cropped_mask))

    del image
    del mask

    return random_crops

class SegmentationDataset(Dataset):
    """
    Allow to load an image segmentation dataset with random crop
    """
    def __init__(self,
                 image_paths: List[str],
                 mask_paths: List[str],
                 crop_size: Tuple[int, int],
                 sample_factor: int=1,
                 device: str='cpu',
                 nworkers: int=1,
                 auto_resample: bool=True):
        """Create a segmentation dataset

        Args:
            image_paths (List[str]): List of all pathes to the images
            mask_paths (List[str]): List of all pathes to the corresponding masks
                Note: must be same length as image_paths
            crop_size (Tuple[int, int]): Height and Width (in that order) of the image crop size
            auto_resample (bool, optional): Weather resample the dataset all the time or not. Defaults to True.
            sample_factor (int, optional): Number of random crop performed per image. Defaults to 1.
            device (str, optional): Where load the dataset. Defaults to 'cpu'.
            nworkers (int, optional): Number of image to process in parallel. Defaults to 1.
        """
        self._image_paths = image_paths
        self._mask_paths = mask_paths
        self._auto_resample = auto_resample
        self._sample_factor = sample_factor if auto_resample else 1
        self._nworkers = nworkers
        self._crop_size = crop_size
        self._device = device
        

        self._samples_chunks: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(len(self._image_paths))
        ]

        self._preload_mutex = Lock()

    def __len__(self):
        """Compute the length of the dataset

        Returns:
            int: Number of images in the dataset
        """
        # return the number of total samples contained in the dataset
        return len(self._samples_chunks)

    def _preload(self):
        """Preload samples of the current dataset"""
        # zip images and masks to iterate them by pairs
        img_msk_zip = zip(self._image_paths, self._mask_paths)

        # create a progress bar with the following parameters
        progress_bar = partial(
            tqdm.tqdm,
            desc='Preload segmentation dataset: ',
            total=len(self._samples_chunks),
            unit='image',
            position=2,
            leave=False
        )

        # create a partial function to send crop size and sample factor
        sample_function = partial(
            sample_image_random_crops,
            crop_size=self._crop_size,
            sample_factor=self._sample_factor,
        )

        if self._nworkers > 1:
            # create a multiprocessing pool
            with Pool(self._nworkers) as pool:
                # create a parallel iterator with function and image/mask pairs
                parallel_iterator = pool.imap(sample_function, img_msk_zip)

                # get all the chunks
                chunks = list(progress_bar(parallel_iterator))
        else:
            # create empty chunks
            chunks = []
            
            # for each image/mask pair (with progress bar)
            for img_msk_paths in progress_bar(img_msk_zip):
                # sample random crops and append them
                chunks.append(sample_function(img_msk_paths))

        # create progress bar for device sending
        chunk_progress = partial(
            tqdm.tqdm,
            desc=f'Sending crops to device {self._device}: ',
            unit='chunk',
            position=2,
            leave=False
        )

        # send all samples to the device
        for i, chunk in enumerate(chunk_progress(chunks)):
            for img, msk in chunk:
                # send image and mask to the selected device
                img, msk = (img.to(self._device), msk.to(self._device))

                # append them to current chunk
                self._samples_chunks[i].append((img, msk))
        del chunks

    def __getitem__(self, idx: int):
        """Pick one item at a specific index in the dataset

        Args:
            idx (int): Item index to pick

        Returns:
            torch.Tensor: Random crop of the image of the dataset at index idx
            torch.Tensor: Corresponding mask of the cropped image
        """
        # get index must be synchronous to avoid multiple preloading
        with self._preload_mutex:
            # get chunk of the selected index
            sample_chunk = self._samples_chunks[idx]

            # if chunk is empty, preload dataset and reselect the chunk
            if len(sample_chunk) == 0:
                self._preload()
                sample_chunk = self._samples_chunks[idx]
        
        # if auto resample needed
        if self._auto_resample:
            # pop first image/mask pair from the selected chunk 
            image, mask = sample_chunk.pop(0)
        else:
            # either just pick sample
            image, mask = sample_chunk[0]
        return (image, mask)
