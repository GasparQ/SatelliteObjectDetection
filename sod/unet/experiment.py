import os
import time
import logging
from typing import List, Tuple

import tqdm
import matplotlib.pyplot as plt

import torch
from torch.nn import Module, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from .u_net import UNet

class Experiment:
    """Perform UNet experiment from scratch, handle saving and reloading"""
    def __init__(self, model: Module, config: dict) -> None:
        self._model = model.to(config.DEVICE)
        self._config = config

        self._loss = BCEWithLogitsLoss()
        self._opt = Adam(self._model.parameters(), lr=config.INIT_LR)

    @staticmethod
    def create(config):
        """Create an new Unet experiment from configuration

        Args:
            config (dict): Configuration of the experiment

        Returns:
            Experiment: newly created experiment
        """
        os.makedirs(config.OUTPUT_PATH)
        return Experiment(UNet(), config)

    @staticmethod
    def load(config):
        """Load an experiment from a configuration

        Args:
            config (dict): Configuration file of the experiment

        Returns:
            Experiment: Loaded experiment from configuration
        """
        model = torch.load(os.path.join(config.OUTPUT_PATH, 'model.pth'))
        return Experiment(model, config)

    def train_step(self, data_loader: DataLoader) -> torch.Tensor:
        """Perform a training step of the experiment

        Args:
            data_loader (DataLoader): Dataset on which perform the training step

        Returns:
            torch.Tensor: Sum of all losses over the dataset
        """
        # set the model in training mode
        self._model.train()
        # initialize the total training and validation loss
        total_loss = 0
        # loop over the training set
        for (x, y) in tqdm.tqdm(data_loader, desc='Training: ',
                                unit='batch', position=1, leave=False):
            # send the input to the device
            # (x, y) = (x.to(self._config.DEVICE), y.to(self._config.DEVICE))
            # perform a forward pass and calculate the training loss
            pred = self._model(x)
            loss = self._loss(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
            # add the loss to the total training loss so far
            total_loss += loss
        return total_loss

    def test_step(self, data_loader: DataLoader) -> torch.Tensor:
        """Perform a test step of the experiment over a dataset

        Args:
            data_loader (DataLoader): Dataset to use for model evaluation

        Returns:
            torch.Tensor: Sum of all losses over the dataset
        """
        total_loss = 0
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            self._model.eval()
            # loop over the validation set
            for (x, y) in tqdm.tqdm(data_loader, desc='Testing: ',
                                    unit='batch', position=1, leave=False):
                # # send the input to the device
                # (x, y) = (x.to(self._config.DEVICE), y.to(self._config.DEVICE))
                # make the predictions and calculate the validation loss
                pred = self._model(x)
                total_loss += self._loss(pred, y)
        return total_loss

    def train(self, train_dataset: DataLoader,
                    test_dataset: DataLoader,
                    plot_loss_history: bool=False) -> Tuple[List[float], List[float]]:
        """Perform a full training + test on the current experiment model

        Args:
            train_dataset (DataLoader): Training dataset on which optimize the model
            test_dataset (DataLoader): Test dataset on which evaluate the model
            save_history (bool): Save the loss history plot into the output path. Default: False.

        Returns:
            List[float]: List of average train losses at each epoch
            List[float]: List of average test losses at each epoch
        """
        # calculate steps per epoch for training and test set
        train_steps = len(train_dataset)
        test_steps = len(test_dataset)

        # Store training history
        train_losses = []
        test_losses = []

        # loop over epochs
        logging.info("Training the network...")
        start_time = time.time()
        for _ in tqdm.trange(self._config.NUM_EPOCHS, desc='Epoch: ', unit='epoch', position=0):

            # perform training step and get sum of train losses
            total_train_loss = self.train_step(train_dataset)

            # perform test step and get sum of test losses
            total_test_loss = self.test_step(test_dataset)

            # calculate the average training and validation loss
            avg_train_loss = total_train_loss / train_steps
            avg_test_loss = total_test_loss / test_steps

            # update our training history
            train_losses.append(avg_train_loss.cpu().detach().numpy())
            test_losses.append(avg_test_loss.cpu().detach().numpy())

            # print the model training and validation information
            logging.info(f"Train loss: {avg_train_loss:.6f}, Test loss: {avg_test_loss:.4f}")

        # display the total time needed to perform the training
        end_time = time.time()
        logging.info(f"Total time taken to train the model: {end_time - start_time:.2f}s")

        if plot_loss_history:
            # plot the training loss
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(train_losses, label="train_loss")
            plt.plot(test_losses, label="test_loss")
            plt.title("Training Loss on Dataset")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend(loc="lower left")
            path = os.path.join(self._config.OUTPUT_PATH, 'loss_history.png')
            plt.savefig(path)
            logging.info(f'Loss history plot saved into {path}')

        return train_losses, test_losses

    def save(self):
        """Save the inner model of the experiment to the log path"""
        path = os.path.join(self._config.OUTPUT_PATH, 'model.pth')
        torch.save(self._model, path)
        logging.info(f'Model saved into {path}')