from typing import Dict, List, Tuple

import random
import itertools

import h5py

import numpy as np

from torch.utils.data import Dataset

class SegmentationDatasetHDF5(Dataset):
    """Segmentation dataset from HDF5 file

    The hdf5 file hierarchy must follow:
        - sample name : ex : 'P0000.png'
            - color transformation : ex : 'posterize'
                - shape transformation : ex : 'rotate_90'
                    - 'images' : tensor of shape (NCROPS, CHANNELS, WIDTH, HEIGHT)
                    - 'masks' : tensor of shape (NCROPS, CHANNELS, WIDTH, HEIGHT)

    The index specified in operator [] correspond to the top level (aka sample name)
    Each time you access an item, it will pick a random augmented version of the sample
    A random version cannot be pick twice until all other version has been picked

    Note: Colors, shapes and number of crops must be identic for all sample
    """

    def __init__(self, h5file: str) -> None:
        """Create a dataset from the path to the hdf5 file

        Args:
            h5file (str): Path to the hdf5 file
        """
        self._h5file: str = h5file
        self._samples_name: List[str] = []
        self._samples: Dict[str, List[Tuple[str, str, int]]] = {}
        self._visited: Dict[str, List[Tuple[str, str, int]]] = {}
        self._preloaded: Dict[str, Tuple[np.array, np.array]] = {}

        self._init_cache_from_file()

    def _init_cache_from_file(self):
        """Create the inner caches from the samples and augmented data in the hdf5 file"""
        # open the file in readonly and swmr mode (multiprocessing)
        with h5py.File(self._h5file, mode='r', swmr=True) as file:
            # retreive samples name from the top level
            self._samples_name = list(file.keys())

            # retreive all possible color transform from first sample
            fsample = self._samples_name[0]
            colors = list(file[fsample].keys())

            # retreive all possible shape transform from first color
            fcolor = colors[0]
            shapes = list(file[f'{fsample}/{fcolor}'].keys())

            # retreive all possible indices from first shape
            fshape = shapes[0]
            indices = list(range(file[f'{fsample}/{fcolor}/{fshape}/images'].shape[0]))

            # all possible augmented samples are the combination of all above possibilities
            augmented_samples = list(itertools.product(colors, shapes, indices))

            # for each sample
            for sample in self._samples_name:

                # create the list of augmented data (cache and visited)
                self._samples[sample] = list(augmented_samples)
                self._visited[sample] = []
                self._preloaded[sample] = (None, None)

    def _get_random_sample_version(self, file, sample):
        # choose a random version of the sample from the augmented data
        random_version = random.choice(self._samples[sample])

        # append the choosen version to the visited versions
        self._visited[sample].append(random_version)

        # remove the choosen version from the cache
        self._samples[sample].remove(random_version)

        # unpack random sample data to retreive them from the hdf5 file
        color, shape, index = random_version

        # retreive the image and mask from the hdf5 file
        image = file[f'{sample}/{color}/{shape}/images'][index]
        mask = file[f'{sample}/{color}/{shape}/masks'][index]

        return image, mask

    def _preload_samples(self):
        """Preload random samples from the hdf5 file"""
        with h5py.File(self._h5file, mode='r', swmr=True) as file:
            for sample in self._samples_name:
                # save image and mask
                self._preloaded[sample] = self._get_random_sample_version(file, sample)


    def __len__(self) -> int:
        """Get the number of samples to retreive in the object
        Used to know the range of values that can be passed in the __getitem__ (operator [])

        Returns:
            int: Number of samples in the dataset
        """
        return len(self._samples_name)

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        """Get a random version of a specific sample in the dataset
        Will be called with operator []

        Args:
            index (int): Integer index of a sample in the dataset

        Returns:
            Tuple[np.array, np.array]: Image and associated mask in a random augmented version
        """

        # find the name of the sample from its integer index
        sample = self._samples_name[index]

        # if the cache is empty, reset it
        if len(self._samples[sample]) == 0:
            # swap samples and visited
            (
                self._samples[sample],
                self._visited[sample]
            ) = (
                self._visited[sample],
                self._samples[sample]
            )

        with h5py.File(self._h5file, mode='r', swmr=True) as file:
            return self._get_random_sample_version(file, sample)

        # image, mask = self._preloaded[sample]
        # if image is None or mask is None:
        #     self._preload_samples()
        #     image, mask = self._preloaded[sample]
        #     self._preloaded[sample] = (None, None)

        # return image, mask
