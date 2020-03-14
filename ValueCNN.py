# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.autonotebook import trange
from utils import one_hot, rotate_cell
from game import WIDTH,HEIGHT

class FCModule(nn.Module):
    def __init__(self, config):
        super(FCModule, self).__init__()

        self.config = config

        # Architecture

        # 8x10 grid and one hot encoded cell state
        #obs_size = 8*10*5

        self.input_channel = 5

        # output is just a scalar value
        self.output_dim = 1
        self.n_hidden = config.n_hidden
        self.hidden_dim = config.hidden_dim

        act_fn = nn.ReLU

        self.min_val = self.config.min_val
        self.max_val = self.config.max_val

        self.dropout = self.config.get("dropout", 0.25)

        layers = []
        # (B,5,10,8)
        layers.append(
            torch.nn.Conv2d(self.input_channel, self.config.get("conv_features", 16)*self.input_channel,
                            kernel_size=3, stride=1, padding=0, groups=self.input_channel)
        )
        layers.append(act_fn())
        # (B,5*ft,8,6)
        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        # (B,5*ft,4,3)
        layers.append(
            torch.nn.Conv2d(self.config.get("conv_features", 16)*self.input_channel, 1, kernel_size=1, stride=1, padding=0, groups=1)
        )
        layers.append(act_fn())
        # (B,1,4,3)
        layers.append(torch.nn.Flatten())
        # (B,12)
        for _ in range(self.n_hidden):
            layers.append(nn.Linear(12, self.hidden_dim))
            layers.append(nn.Dropout(p=self.dropout))
            layers.append(act_fn())
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.model = nn.Sequential(*layers)

        # Training params:
        self.lr = config.lr
        self.loss_fn = F.mse_loss

    def forward(self, inputs):
        return torch.clamp(self.model(inputs),min=self.min_val,max=self.max_val)

    def compute_cost(self, inputs, targets):
        pred = self.forward(inputs)
        return self.loss_fn(pred, targets.unsqueeze(-1))


class ValueModel:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = FCModule(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.epochs = self.config.epochs   
        self.train_losses,self.valid_losses = [],[]

    def train(self, replay_buffer):
        """Train the forward model

        Inputs:
        replay_buffer (list): List of lists of Tensors of the format:
            [[states_i, moves_i, values_i]]
            Where i is the episode index

        Returns:
        mean_training_loss (float): Mean training loss
        """
        self.model.train(True)

        states, _, values = zip(*replay_buffer)
        
        val_ind = int(np.floor(len(states)/10))
        inputs = torch.cat(states[val_ind:], 0)
        targets = torch.cat(values[val_ind:], 0)

        val_inputs = torch.cat(states[:val_ind], 0)
        val_targets = torch.cat(values[:val_ind], 0)

        dataset = TensorDataset(inputs, targets)
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        epoch_range = trange(self.epochs, desc="training model", leave=False)
        for _ in epoch_range:
            self.model.train(True)
            running_loss = 0
            for states_batch, targets_batch in data_loader:
                # for convenience, later we might want transformations (e.g. for state-action value)
                inp = states_batch 
                loss = self.model.compute_cost(inp.to(self.device), targets_batch.to(self.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.train_losses.append(running_loss)
            epoch_range.set_postfix({"Loss": np.mean(np.array(self.train_losses))})

            #compute validation loss
            self.model.eval()
            vloss = self.model.compute_cost(val_inputs.to(self.device), val_targets.to(self.device))
            self.valid_losses.append(vloss.detach().item())
        # Optionally, decay the number of epochs
        self.epochs = max(
            int(self.config.get("epoch_decay_rate", 1.0) * self.epochs),
            self.config.get("epoch_lower_limit", 5),
        )

        mean_training_loss = np.mean(np.array(self.train_losses))
        return mean_training_loss

    @staticmethod
    def grid2input(grid, rot=0):
        res = np.zeros((5,HEIGHT, WIDTH))
        for i in range(HEIGHT):
            for j in range(WIDTH):
                res[:,i,j] = one_hot(rotate_cell(grid[i][j], rot=rot))

        return res

    def reset(self):
        """Reset the model

        Deletes the model, optimizer instances and creates new ones
        """
        del self.model
        del self.optimizer
        self.model = FCModule(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.training.lr)

    def save(self, save_dir, prefix = ""):
        """Save the forward model

        Saves the parameters of the network and the optimizer

        Inputs:
        save_dir (str): Path to the directory to save the parameters to.

        Returns:
        None

        NOTE: Throws an `AssertionError` if `save_dir` does not exist
        """
        assert os.path.exists(save_dir), "save_dir: {} does not exist!".format(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, f"{prefix}weights.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, f"{prefix}optimizer.pth"))

    def load(self, load_dir, prefix = ""):
        """Load a saved forward model

        Load the state of the forward model and the optimizer from a directory
        NOTE: Assumes that the files `scaler.pth`, 'weights.pth` and `optimizer.pth` can all be
            found in `load_dir`

        Inputs:
        load_dir (str): Path to the directory containing the parameters to be loaded

        Returns:
        None

        NOTE: Throws an `AssertionError` if:
            1. `save_dir` does not exist
            2. Atleast one of: `weights.pth` or 'optimimzer.pth` does not exist
        """
        assert os.path.exists(load_dir), "load_dir: {} does not exist!".format(load_dir)
        assert os.path.exists(
            os.path.join(load_dir, f"{prefix}weights.pth")
        ), "file: {} does not exist!".format(os.path.join(load_dir, f"{prefix}weights.pth"))
        assert os.path.exists(
            os.path.join(load_dir, f"{prefix}optimizer.pth")
        ), "file: {} does not exist!".format(os.path.join(load_dir, f"{prefix}optimizer.pth"))

        self.model.load_state_dict(torch.load(os.path.join(load_dir, f"{prefix}weights.pth")))
        self.optimizer.load_state_dict(torch.load(os.path.join(load_dir, f"{prefix}optimizer.pth")))
