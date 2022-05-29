"""Script to generate an augmented image segmentation dataset into an hdf5 file"""

import os
from typing import Iterator, List, Tuple

from multiprocessing import Queue, Pool, Manager

import numpy as np
import tqdm
import h5py

import torch
from torchvision.transforms import functional as trf

from sod.processing import image_augmentation

class ImageMaskSampler:
    """Perform image sampling asynchronously"""
    def __init__(self,
                 sampling_strategy: str,
                 **sampling_configuration) -> None:
        """Create sampler from several options

        Args:
            sampling_strategy (str): Sampling strategy to use for each image (archive, group, augment)
            **sampling_configuration: Configuration for each strategy
                archive: None
                group: 'crop_size'
                augment: 'crop_number' (default=1), 'crop_height', 'crop_width'
        """
        self._samples_queue = None
        if sampling_strategy.lower() == 'archive':
            self._sampling_strategy = self._archive
            self.n_samples_per_image = 1
        elif sampling_strategy.lower() == 'group':
            self._sampling_strategy = self._group
            self.n_samples_per_image = 1
            self._crop_size = (
                sampling_configuration['crop_height'],
                sampling_configuration['crop_width']
            )
        elif sampling_strategy.lower() == 'augment':
            self._sampling_strategy = self._augment
            self._crop_size = (
                sampling_configuration['crop_height'],
                sampling_configuration['crop_width']
            )
            self._crop_number = sampling_configuration.get('crop_number', 1)
            self.n_samples_per_image = (image_augmentation.N_COLOR_TRANSFORMS *
                                        image_augmentation.N_SHAPE_TRANSFORMS)

    def set_sample_queue(self, samples_queue: Queue):
        self._samples_queue = samples_queue

    def _archive(self, image_path: str, mask_path: str) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
        # get the filename of the image
        img_basename = os.path.basename(image_path)
        # load the image and masks file content
        img, mask = image_augmentation.load_image_mask_pair(image_path, mask_path)

        img = np.transpose(img, axes=(2, 0, 1))

        yield (img_basename, torch.as_tensor(img, dtype=torch.uint8), torch.as_tensor(mask, dtype=torch.uint8))

    def _group(self, image_path: str, mask_path: str) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
        # get the filename of the image
        img_basename = os.path.basename(image_path)
        # load the image and masks file content
        img, mask = image_augmentation.load_image_mask_pair(image_path, mask_path)

        # convert to pil image
        img = torch.as_tensor(np.transpose(img, axes=(2, 0, 1)), dtype=torch.uint8)
        mask = torch.as_tensor(np.transpose(mask, axes=(2, 0, 1)), dtype=torch.uint8)

        # perform synchronous center crop
        img = trf.center_crop(img, output_size=self._crop_size)
        mask = trf.center_crop(mask, output_size=self._crop_size)

        yield (img_basename, img, mask)

    def _augment(self, image_path: str, mask_path: str) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
        # get the filename of the image
        img_basename = os.path.basename(image_path)
        # load the image and masks file content
        img, mask = image_augmentation.load_image_mask_pair(image_path, mask_path)

        # convert np array to PIL image for augmentation
        img = trf.to_pil_image(img)

        # convert mask as float32 tensor
        mask = torch.as_tensor(mask, dtype=torch.float32)

        # perform color transformations on images only (not masks)
        for color_tr, tr_image in image_augmentation.color_transformer(img):
            # perform shape transformations on images and masks to keep synchronisation
            for shape_tr, (rt_img, rt_msk) in image_augmentation.shape_transformer(tr_image, mask):
                # print(shape_tr)
                # perform random crop on transformed images/masks
                s_imgs, s_msks = [], []
                for _ in range(self._crop_number):
                    c_img, c_mask = image_augmentation.sync_random_crop(rt_img, rt_msk, self._crop_size)
                    s_imgs.append(c_img)
                    s_msks.append(c_mask)
                # yield crop stack
                yield (
                    f'{img_basename}/{color_tr}/{shape_tr}',
                    torch.stack(s_imgs),
                    torch.stack(s_msks)
                )

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
        # peform sampling
        size = 0
        for name, img, msk in self._sampling_strategy(image_path, mask_path):
            self._samples_queue.put((name, img, msk))
            size += 1
        return size

def generate_groupped_hdf5(images: List[str], masks: List[str], output_path: str, sampler: ImageMaskSampler, parallel: int):
    # Open the hdf5 file that will contain the final dataset
    with h5py.File(output_path, mode='w') as output_file:

        with Manager() as manager:
            # Create multiprocessing manager and queue
            samples_q = manager.Queue(maxsize=parallel)

            # Create a sample with the queue to feed and augmentation parameters
            sampler.set_sample_queue(samples_q)

            # Compute sample size in input and output
            n_samples_in = len(images)
            n_samples_out = n_samples_in * sampler.n_samples_per_image

            # Start the multiprocessing pool with the right number of processes
            with Pool(parallel) as p:

                # Map image/mask pair to pool workers
                p.starmap_async(sampler, zip(images, masks))

                # For each output sample
                for i in tqdm.trange(n_samples_out, desc='Image sampling', unit='sample'):

                    # Retreive next sample from queue
                    _, imgs, masks = samples_q.get()

                    img_shape = imgs.shape
                    msk_shape = masks.shape

                    imgs = imgs[None, :]
                    masks = masks[None, :]

                    if i == 0:
                        output_file.create_dataset('/images', data=imgs,
                                                compression='gzip', chunks=True,
                                                maxshape=(None,) + img_shape)

                        output_file.create_dataset('/masks', data=masks,
                                                compression='gzip', chunks=True,
                                                maxshape=(None,) + msk_shape)
                    else:
                        output_file['/images'].resize(output_file['/images'].shape[0] + 1, axis=0)
                        output_file['/images'][-1:] = imgs

                        output_file['/masks'].resize(output_file['/masks'].shape[0] + 1, axis=0)
                        output_file['/masks'][-1:] = masks

def generate_named_hdf5(images: List[str], masks: List[str], output_path: str, sampler: ImageMaskSampler, parallel: int):
    # Open the hdf5 file that will contain the final dataset
    with h5py.File(output_path, mode='w') as output_file:

        with Manager() as manager:
            # Create multiprocessing manager and queue
            samples_q = manager.Queue(maxsize=parallel)

            # Create a sample with the queue to feed and augmentation parameters
            sampler.set_sample_queue(samples_q)

            # Compute sample size in input and output
            n_samples_in = len(images)
            n_samples_out = n_samples_in * sampler.n_samples_per_image

            # Start the multiprocessing pool with the right number of processes
            with Pool(parallel) as pool:

                # Map image/mask pair to pool workers
                pool.starmap_async(sampler, zip(images, masks))

                # For each output sample
                for _ in tqdm.trange(n_samples_out, desc='Image sampling', unit='sample'):

                    # Retreive next sample from queue
                    name, imgs, masks = samples_q.get()

                    # Write into hdf5 file in the group /image_name/color_transform/shape_transform
                    group = output_file.create_group(f'/{name}')

                    # 2 datasets will be writen: transformed image crops, transformed mask crops
                    group.create_dataset('images', data=imgs.numpy(), compression='gzip')
                    group.create_dataset('masks', data=masks.numpy(), compression='gzip')

def parse_size(size: str) -> Tuple[int, int]:
    if not re.match('[0-9]+x[0-9]+', size):
        raise ValueError(f"Argument --size (-s) must be formatted as [WIDTH]x[HEIGHT]"
                        f" (ex: 128x128) but got {size}")

    # Parse the --size option to extract width & height

    width, height = (int(v) for v in size.split('x'))

    return width, height

if __name__ == '__main__':

    # Define the command line argument parser
    import re
    import glob
    import argparse

    from datetime import datetime

    parser = argparse.ArgumentParser()

    # Definition of 4 positional arguments
    parser.add_argument(
        'name',
        help='Name of the dataset generated',
        type=str
    )
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

    # Definition of 3 optional arguments
    parser.add_argument(
        '-p', '--parallel',
        help="Number of CPU on which parallelism processing",
        type=int, default=os.cpu_count()
    )
    parser.add_argument(
        '-o', '--output',
        help="Folder where to store the dataset (will create a folder inside)",
        type=str,
        default=os.getcwd()
    )

    # sub commands
    subparsers = parser.add_subparsers(help='Dataset generation mode')

    # archive
    archive_parser = subparsers.add_parser('archive', help='Generate a dataset with original images stored in different dataset')
    def get_archive_sampler(archive_args):
        return ImageMaskSampler('archive')
    archive_parser.set_defaults(get_sampler=get_archive_sampler, group=False)

    # group images into 1 center cropped images dataset
    groupped_parser = subparsers.add_parser('group', help='Generate a dataset of center cropped images stored in 1 unique dataset')
    groupped_parser.add_argument(
        '-s', '--size',
        help='Crop dimensions of the group (ex: 128x128). Set to 0x0 for the maximum allowed size.',
        type=str,
        default='0x0'
    )
    def parse_groupped_subcommand(groupped_args):
        width, height = parse_size(groupped_args.size)

        return ImageMaskSampler(
            'group',
            crop_height=height,
            crop_width=width,
        )
    groupped_parser.set_defaults(get_sampler=parse_groupped_subcommand, group=True)

    # augment subcommand
    augment_parser = subparsers.add_parser('augment', help='Generate a dataset with fixed data augmentation')
    augment_parser.add_argument(
        "-s", "--size",
        help="Crop dimensions (ex: 128x128)",
        type=str, default='128x128'
    )
    augment_parser.add_argument(
        "-c", "--crop",
        help="Number of crops to perform per image",
        type=int, default=50
    )
    def get_augment_sampler(augmented_args):
        width, height = parse_size(augmented_args.size)

        return ImageMaskSampler(
            'augment',
            crop_number=augmented_args.crop,
            crop_height=height,
            crop_width=width
        )
    # For augment subcommand => call generate_augmented_dataset function
    augment_parser.set_defaults(get_sampler=get_augment_sampler, group=False)

    # Parse arguments from command line
    args = parser.parse_args()

    # Check if provided folders are actually directories
    if not os.path.isdir(args.images_folder):
        raise NotADirectoryError(f'Argument {args.image_folder} (image_folder) must be a directory')

    if not os.path.isdir(args.masks_folder):
        raise NotADirectoryError(f'Argument {args.masks_folder} (masks_folder) must be a directory')

    # Search for all png files in the provided folders
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

    # Check if the --parallel option is between 1 and os.cpu_count()
    if args.parallel < 1 or args.parallel > os.cpu_count():
        raise ValueError(f"Argument --parallel (-p) must be between 1 and {os.cpu_count()}"
                         f"(included): got {args.parallel}")

    # Check if output folder exists
    if not os.path.isdir(args.output):
        raise NotADirectoryError(f"Output folder is not a directory: {args.output}")

    # parse args
    args = parser.parse_args()

    # Resolve subcommands
    sampler = args.get_sampler(args)

    output_path = os.path.join(
        args.output,
        f'{args.name}.{datetime.now():%Y_%m_%d.%H_%M_%S}.hdf5'
    )

    if os.path.exists(output_path):
        raise FileExistsError(f"Dataset file already exists: {output_path}. Delete it first.")

    if args.group:
        generate_groupped_hdf5(images, masks, output_path, sampler, args.parallel)
    else:
        generate_named_hdf5(images, masks, output_path, sampler, args.parallel)
