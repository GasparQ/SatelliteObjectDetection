from typing import Iterator, Tuple, List

import cv2

import numpy as np

import torch
from torchvision import transforms
from torchvision.transforms import functional as trf

def rotate_generator(image: torch.Tensor, mask: torch.Tensor) -> Iterator[Tuple[str, Tuple[torch.Tensor, torch.Tensor]]]:
    yield 'no_rotate', (image, mask)
    yield 'rotate_90', (trf.rotate(image, 90), trf.rotate(mask, 90))
    yield 'rotate_180', (trf.rotate(image, 180), trf.rotate(mask, 180))
    yield 'rotate_270', (trf.rotate(image, 270), trf.rotate(mask, 270))
    yield 'vflip', (trf.vflip(image), trf.vflip(mask))
    yield 'hflip', (trf.hflip(image), trf.hflip(mask))

def transform_generator(image: torch.Tensor) -> Iterator[Tuple[str, torch.Tensor]]:
    trs: List[Tuple[str, torch.Module]] = [
        ('original', None),
        ('color_jit', transforms.ColorJitter()),
        ('grayscale', transforms.Grayscale(num_output_channels=3)),
        ('perspective_d0.1', transforms.RandomPerspective(distortion_scale=0.1, p=1)),
        ('inverted', transforms.RandomInvert(p=1)),
        ('solarized_t200', transforms.RandomSolarize(threshold=200, p=1)),
        ('posterized_b2', transforms.RandomPosterize(bits=2, p=1)),
        ('blur_k5_s0.1+2', transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2))),
        ('equalize', transforms.RandomEqualize(p=1)),
        ('contrast', transforms.RandomAutocontrast(p=1)),
    ]
    for name, tr in trs:
        if tr is None:
            yield name, image
        else:
            yield name, tr(image)

def sync_random_crop(image, mask, crop_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform synchronous random crop between image and mask

    Args:
        image (np.array): Image to crop
        mask (np.array): Corresponding mask
        crop_size (Tuple[int, int]): Height and Width (in that order) for the crop size

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Pair of synchronized cropped image and mask
    """
    # get random crop indices, height and width
    i, j, height, width = transforms.RandomCrop.get_params(image, crop_size)

    # perform synchronous crop on image and mask
    cropped_img = trf.to_tensor(trf.crop(image, i, j, height, width))
    cropped_mask = trf.crop(mask, i, j, height, width).float()
    return cropped_img, cropped_mask

def load_image_mask_pair(image_path: str, mask_path: str)  -> Tuple[np.array, np.array]:
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

    mask = torch.as_tensor(mask, dtype=torch.uint8)

    return image, mask

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

    image, mask = load_image_mask_pair(*img_mask_pathes)

    # convert to uint8 tensor
    mask = torch.as_tensor(mask, dtype=torch.uint8)

    random_crops = [sync_random_crop(image, mask, crop_size) for _ in range(sample_factor)]

    del image
    del mask

    return random_crops
