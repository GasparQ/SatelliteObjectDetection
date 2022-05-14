import os
from typing import List, Tuple
import torch
from sod.processing import image_augmentation

def sample_image_mask(img_mask: Tuple[str, str], crop_size: Tuple[int, int], sample_size: int) -> List[Tuple[str, str, str, torch.Tensor, torch.Tensor]]:
    img, mask = img_mask
    img_basename = os.path.basename(img)
    img, mask = image_augmentation.load_image_mask_pair(img, mask)
    augment_samples = []
    for transform_name, tr_image in image_augmentation.transform_generator(img):
        for rotate_name, (rt_img, rt_msk) in image_augmentation.rotate_generator(tr_image, mask):
            s_imgs, s_msks = [], []
            for _ in range(sample_size):
                c_img, c_mask = image_augmentation.sync_random_crop(rt_img, rt_msk, crop_size)
                s_imgs.append(c_img)
                s_msks.append(c_mask)
            augment_samples.append((
                img_basename,
                transform_name,
                rotate_name,
                torch.stack(s_imgs),
                torch.stack(s_msks)
            ))
    return augment_samples

if __name__ == '__main__':
    import re
    import glob
    import argparse
    from functools import partial
    from datetime import datetime

    parser = argparse.ArgumentParser()
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

    args = parser.parse_args()

    if not os.path.isdir(args.images_folder):
        raise NotADirectoryError(f'Argument {args.image_folder} (image_folder) must be a directory')

    if not os.path.isdir(args.masks_folder):
        raise NotADirectoryError(f'Argument {args.masks_folder} (masks_folder) must be a directory')

    images = glob.glob(os.path.join(args.images_folder, '*.png'))
    masks = glob.glob(os.path.join(args.masks_folder, '*.png'))

    if len(images) != len(masks):
        raise OSError(f"The number of images and masks must be identic: got images={images} and masks={masks}")

    if not os.path.isdir(args.output_folder):
        raise NotADirectoryError(f'Argument {args.output_folder} (output_folder) must be a directory')

    if not re.match('[0-9]+x[0-9]+', args.size):
        raise ValueError(f"Argument --size (-s) must be formatted as [WIDTH]x[HEIGHT] (ex: 128x128) but got {args.size}")
    width, height = (int(v) for v in args.size.split('x'))

    if args.crop < 1:
        raise ValueError(f"Argument --crop (-n) must be positive integer: got {args.crop}")

    if args.parallel < 1 or args.parallel > os.cpu_count():
        raise ValueError(f"Argument --parallel (-p) must be between 1 and {os.cpu_count()} (included): got {args.parallel}")

    import h5py
    import tqdm
    import multiprocessing

    filepath = os.path.join(args.output_folder, f'{args.name}.{datetime.now():%Y_%m_%d.%H_%M_%S}.hdf5')
    with h5py.File(filepath, mode='w') as output_file:
        with multiprocessing.Pool(args.parallel) as p:
            sample_image_mask_job = partial(sample_image_mask, crop_size=(height, width), sample_size=args.crop)
            img_samples = p.imap(sample_image_mask_job, zip(images, masks))
            for samples in tqdm.tqdm(img_samples, desc='Augment image', total=len(images), position=0):
        # for img, mask in tqdm.tqdm(zip(images, masks), desc='Augment image', total=len(images), position=0):
        #     samples = sample_image_job((img, mask))
                for im_name, tr_name, rt_name, imgs, msks in tqdm.tqdm(samples, desc='Write samples', position=1, leave=False):
                    group = output_file.create_group(f'/{im_name}/{tr_name}/{rt_name}')
                    group.create_dataset('images', data=imgs.numpy(), compression='gzip')
                    group.create_dataset('masks', data=msks.numpy(), compression='gzip')
