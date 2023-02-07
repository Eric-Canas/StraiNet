"""
Performs training and evaluation of the model.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train.loss import RotationLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from loguru import logger
from utils import letterbox

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
NUM_WORKERS = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINTS_FOLDER = os.path.join('.', 'checkpoints')

class Trainer:
    def __init__(self, model: nn.Module, train_dataset: Dataset, val_dataset: Dataset, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
        self.model = model
        self.dataset = train_dataset
        self.val_dataset = val_dataset
        # Set the number of workers to 0 if you are using Windows
        self.num_workers = num_workers if os.name != 'nt' else 0
        self.dataset = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_dataset = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.device = DEVICE

        # TODO: A custom loss function could improve the performance of the model
        self.criterion = RotationLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.model.to(self.device)
        self.writer = SummaryWriter()
        self.global_step = 0

    def train(self, epochs: int = EPOCHS):
        """
        Trains the model for the specified number of epochs.
        :param epochs: int. The number of epochs to train the model for.
        """
        for epoch in range(epochs):
            logger.info(f'Epoch {epoch + 1}/{epochs}')
            logger.info('-' * len(f'Epoch {epoch + 1}/{epochs}'))
            self.train_one_epoch()
            self.evaluate()
            self.global_step += 1
            if epoch % 10 == 0:
                self.save_checkpoint(path=os.path.join(CHECKPOINTS_FOLDER, f'checkpoint_{epoch}.pt'))

    def train_one_epoch(self):
        """
        Trains the model for one epoch.
        """
        # Set the model to training mode
        self.model.train()
        avg_loss = []
        # Iterate over the dataset (batch by batch)
        for img, label in tqdm(self.dataset, total=len(self.dataset), desc='Training'):
            # Move the data to the device
            img, label = img.to(self.device), label.to(self.device)
            # Forward pass
            output = self.model(img)
            # Compute the loss
            loss = self.criterion(output, label)
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Log the loss
            self.writer.add_scalar('Training loss', loss.item(), self.global_step)
            avg_loss.append(loss.item())
        logger.info(f'Average training loss: {np.mean(avg_loss)}')

    def evaluate(self):
        """
        Evaluates the model on the validation set.
        """

        # Set the model to evaluation mode
        self.model.eval()
        # Disable gradient computation
        avg_loss = []
        with torch.no_grad():
            # Iterate over the dataset (batch by batch)
            for img, label in tqdm(self.val_dataset, total=len(self.val_dataset), desc='Evaluating'):
                # Move the data to the device
                img, label = img.to(self.device), label.to(self.device)
                # Forward pass
                output = self.model(img)
                # Compute the loss
                loss = self.criterion(output, label)
                # Log the loss
                self.writer.add_scalar('Validation loss', loss.item(), self.global_step)
                avg_loss.append(loss.item())
        logger.info(f'Average validation loss: {np.mean(avg_loss)}')

    def save_checkpoint(self, path: str):
        """
        Saves the model's state dictionary to the specified path.
        :param path: str. The path to save the model's state dictionary to.
        """
        folder = os.path.dirname(path)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        torch.save(self.model.state_dict(), path)
        logger.info(f'Saved checkpoint to {path}')

    def load_checkpoint(self, path: str):
        """
        Loads the model's state dictionary from the specified path.
        :param path: str. The path to load the model's state dictionary from.
        """
        self.model.load_state_dict(torch.load(path))
        logger.info(f'Loaded checkpoint from {path}')

    def predict(self, img):

        # Set the model to evaluation mode
        self.model.eval()
        # Disable gradient computation
        with torch.no_grad():
            # Move the data to the device
            img = img.to(self.device)
            # Forward pass
            output = self.model(img)
            # Compute the loss
            return output