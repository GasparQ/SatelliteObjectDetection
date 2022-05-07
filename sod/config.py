import os
import torch

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
INPUT_CHANNEL = 3
NUM_CLASSES = 1
OUTPUT_CHANNEL = 1
ENCODING_CHANNELS = (16, 32, 64)

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
# BATCH_SIZE = 64
BATCH_SIZE = 32

# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
OUTPUT_PATH = "output"