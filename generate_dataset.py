"""Script to generate an augmented image segmentation dataset into an hdf5 file"""

from ast import parse
import os
from typing import Tuple

from multiprocessing import Queue, Pool, Manager

import torch

from sod.processing import image_augmentation

class ImageMaskSampler:
    """Perform image sampling asynchronously"""

    def __init__(self,
                 samples_queue: Queue,
                 crop_size: Tuple[int, int],
                 crop_number: int,
                 with_augmentation: bool=True) -> None:
        """Create an image sampler

        Args:
            samples_queue (Queue): Multiprocessing queue to feed with generated samples
            crop_size (Tuple[int, int]): Size used to random crop samples
            crop_number (int): Number of random crops to perform per augmented image
            with_augmentation (bool, optional): Weather perform data augmentation or not
        """
        self._samples_queue = samples_queue
        self._crop_size = crop_size
        self._crop_number = crop_number
        self._with_augmentation = with_augmentation
        if with_augmentation:
            self.n_transforms = (image_augmentation.N_COLOR_TRANSFORMS *
                                  image_augmentation.N_SHAPE_TRANSFORMS)
        else:
            self.n_transforms = 1

    def __call__(self, image_path: str, mask_path: str) -> int:
        """Generate samples of image and mask
        This function will generate both image color transformations and images/masks
        shape transformations.
        For each transformation, it will perform several random crops.
        Each samples will the added into multiprocessing queue

        Args:
            img_mask (Tuple[str, str]): Pathes to the image and masks file (in that order) in PNG format
            crop_size (Tuple[int, int]): Height and width of random crop on transformed images
            crop_number (int): Number of crops to perform on transformed image

        Returns:
            int: The number of samples generated into queue
        """
        # print('on demarre le call')
        # get the filename of the image
        img_basename = os.path.basename(image_path)
        # load the image and masks file content
        img, mask = image_augmentation.load_image_mask_pair(image_path, mask_path)
        # peform sampling
        size = 0

        # create color samples on which iterate with or without data augmentation
        if self._with_augmentation:
            color_samples = image_augmentation.color_transformer(img)
        else:
            color_samples = (('original', img),)

        # perform color transformations on images only (not masks)
        for color_tr, tr_image in color_samples:
            
            # create shape samples on which iterate with or without data augmentation
            if self._with_augmentation:
                shape_samples = image_augmentation.shape_transformer(tr_image, mask)
            else:
                shape_samples = (('original', (tr_image, mask)),)

            # perform shape transformations on images and masks to keep synchronisation
            for shape_tr, (rt_img, rt_msk) in shape_samples:
                # print(shape_tr)
                # perform random crop on transformed images/masks
                s_imgs, s_msks = [], []
                for _ in range(self._crop_number):
                    c_img, c_mask = image_augmentation.sync_random_crop(rt_img, rt_msk, self._crop_size)
                    s_imgs.append(c_img)
                    s_msks.append(c_mask)
                # append stack of all the crops to the samples queue
                self._samples_queue.put((
                    img_basename,
                    color_tr,
                    shape_tr,
                    torch.stack(s_imgs),
                    torch.stack(s_msks)
                ))
                size += 1
        return size

if __name__ == '__main__':

    # Define the command line argument parser

    import argparse

    parser = argparse.ArgumentParser()

    # Definition of 4 positional arguments

    parser.add_argument(
        "images_folder",
        help="Folder containing all the dataset images",
        type=str
    )
    parser.add_argument(
        "masks_folder",
        help="Folder containing all the masks",
        type=str
    )
    parser.add_argument(
        "output_folder",
        help="Folder where to store the dataset (will create a folder inside)",
        type=str
    )
    parser.add_argument(
        'name',
        help='Name of the dataset generated',
        type=str
    )

    # Definition of 3 optional arguments

    parser.add_argument(
        "-s", "--size",
        help="Sample dimensions (ex: 128x128)",
        type=str, default='128x128'
    )
    parser.add_argument(
        "-c", "--crop",
        help="Number of crops to perform per image",
        type=int, default=50
    )
    parser.add_argument(
        '-p', '--parallel',
        help="Number of CPU on which parallelise processing",
        type=int, default=os.cpu_count()
    )
    parser.add_argument(
        '-a', '--augmentation',
        help='Perform data augmentation over samples',
        action='store_true'
    )

    # Parse arguments from command line

    args = parser.parse_args()

    # Check if provided folders are actually directories

    if not os.path.isdir(args.images_folder):
        raise NotADirectoryError(f'Argument {args.image_folder} (image_folder) must be a directory')

    if not os.path.isdir(args.masks_folder):
        raise NotADirectoryError(f'Argument {args.masks_folder} (masks_folder) must be a directory')

    # Search for all png files in the provided folders

    import glob

    images = glob.glob(os.path.join(args.images_folder, '*.png'))
    masks = glob.glob(os.path.join(args.masks_folder, '*.png'))

    # Check if there is at least one file in both folders

    if len(images) == 0 or len(masks) == 0:
        raise FileNotFoundError("No such images or masks in provided folders: "
                                f"image={args.images_folder} and masks={args.masks_folder}")

    # Check if the number of files corresponds

    if len(images) != len(masks):
        raise OSError("The number of images and masks must be identic:"
                      f" got images={len(images)} and masks={len(masks)}")

    # Check if the output folder exists

    if not os.path.isdir(args.output_folder):
        raise NotADirectoryError(f'Argument {args.output_folder} (output_folder) '
                                 'must be a directory')

    # Check if the --size option respects the format [NUMBER]x[NUMBER]

    import re

    if not re.match('[0-9]+x[0-9]+', args.size):
        raise ValueError(f"Argument --size (-s) must be formatted as [WIDTH]x[HEIGHT]"
                         f" (ex: 128x128) but got {args.size}")

    # Parse the --size option to extract width & height

    width, height = (int(v) for v in args.size.split('x'))

    # Check if the --crop option is greater than 0

    if args.crop < 1:
        raise ValueError(f"Argument --crop (-n) must be positive integer: got {args.crop}")

    # Check if the --parallel option is between 1 and os.cpu_count()

    if args.parallel < 1 or args.parallel > os.cpu_count():
        raise ValueError(f"Argument --parallel (-p) must be between 1 and {os.cpu_count()}"
                         f"(included): got {args.parallel}")

    # Generate the filename to generate in the output folder

    from datetime import datetime

    filepath = os.path.join(
        args.output_folder,
        f'{args.name}.{args.size}.{args.crop}.{datetime.now():%Y_%m_%d.%H_%M_%S}.hdf5'
    )

    # Check if filepath already exists

    if os.path.exists(filepath):
        raise FileExistsError(f"Unable to generate dataset into {filepath}: Already exists")

    # Perform dataset generation in multiprocessing

    import tqdm
    import h5py

    # Open the hdf5 file that will contain the final dataset
    with h5py.File(filepath, mode='w') as output_file:

        # Start the multiprocessing pool with the right number of processes
        with Pool(args.parallel) as p:

            # Create multiprocessing manager and queue
            manager = Manager()
            maxsize = int(6e9 / (width * height * args.crop * 4))  # limit to 3Go RAM
            samples_q = manager.Queue(maxsize=maxsize)

            # Create a sample with the queue to feed and augmentation parameters
            sampler = ImageMaskSampler(samples_q, (height, width), args.crop,
                                       with_augmentation=args.augmentation)

            # Compute sample size in input and output

            n_samples_in = len(images)
            n_samples_out = n_samples_in * sampler.n_transforms

            # Map image/mask pair to pool workers
            p.starmap_async(sampler, zip(images, masks))

            # For each output sample
            for _ in tqdm.trange(n_samples_out, desc='Image Augmentation', unit='sample'):

                # Retreive next sample from queue
                im_name, col_tr, shap_tr, imgs, masks = samples_q.get()

                # Write into hdf5 file in the group /image_name/color_transform/shape_transform
                group = output_file.create_group(f'/{im_name}/{col_tr}/{shap_tr}')

                # 2 datasets will be writen: transformed image crops, transformed mask crops
                group.create_dataset('images', data=imgs.numpy(), compression='gzip')
                group.create_dataset('masks', data=masks.numpy(), compression='gzip')
