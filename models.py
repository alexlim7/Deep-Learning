from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


class OneLayerNN(nn.Module):
    def __init__(self, input_features: int = 11):
        """
        Initializes a linear layer.

        Parameters
        ----------
        input_features : int, default=11
            The number of features of each sample.
        """
        super().__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear layer defined in __init__() to input features X.

        Parameters
        ----------
        X : torch.Tensor
            2D torch tensor of shape [n, 11], where n is batch size.
            Represents features of a batch of data.

        Returns
        -------
        torch.Tensor
            2D torch tensor of shape [n, 1], where n is batch size.
            Represents prediction of wine quality.
        """
        return self.linear(X)


class TwoLayerNN(nn.Module):
    def __init__(self, input_features: int = 11):
        """
        Initializes model layers.

        Parameters
        ----------
        input_features : int, default=11
            The number of features of each sample.
        """
        super().__init__()
        hidden_size = 32
        self.linear1 = nn.Linear(input_features, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size, 1)
        pass

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Applies the layers defined in __init__() to input features X.

        Parameters
        ----------
        X : torch.Tensor
            2D torch tensor of shape [n, 11], where n is batch size.
            Represents features of a batch of data.

        Returns
        -------
        torch.Tensor
            2D torch tensor of shape [n, 1], where n is batch size.
            Represents prediction of wine quality.
        """
        x = self.linear1(X)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_channels: int = 1, class_num: int = 10):
        """
        Initializes model layers.

        Parameters
        ----------
        input_channels : int, default=1
            The number of features of each sample.
        class_num : int, default=10
            The number of categories.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 4 * 4, class_num)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Applies the layers defined in __init__() to input features X.

        Parameters
        ----------
        X : torch.Tensor
            4D torch tensor of shape [n, 1, 8, 8], where n is batch size.
            Represents a batch of 8 * 8 gray scale images.

        Returns
        -------
        torch.Tensor
            2D torch tensor of shape [n, 10], where n is batch size.
            Represents logits of different categories.
        """
        x = self.conv1(X)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x



def train(
    model: nn.Module,
    dataloader: DataLoader,
    loss_func: nn.MSELoss,
    optimizer: torch.optim,
    num_epoch: int,
    correct_num_func: Callable = None,
    print_info: bool = True,
) -> list[float] | tuple[list[float], list[float]]:
    """
    Trains the model for `num_epoch` epochs.

    Parameters
    ----------
    model : torch.nn.Module
        A deep model.
    dataloader : torch.utils.data.DataLoader
        Dataloader of the training set. Contains the training data equivalent to ((Xi, Yi)),
        where (Xi, Yi) is a batch of data.
        X: 2D torch tensor for UCI wine and 4D torch tensor for MNIST.
        Y: 2D torch tensor for UCI wine and 1D torch tensor for MNIST, containing the corresponding labels
            for each example.
    loss_func : torch.nn.MSELoss
        An MSE loss function for UCI wine and a cross entropy loss for MNIST.
    optimizer : torch.optim
        An optimizer instance from torch.optim.
    num_epoch : int
        The number of epochs we train our network.
    correct_num_func : Callable, default=None
        A function to calculate how many samples are correctly classified.
        To train the CNN model, we also want to calculate the classification accuracy in addition to loss.
    print_info : bool, default=True
        If True, print the average loss (and accuracy, if applicable) after each epoch.

    Returns
    -------
    epoch_average_losses : list[float]
        A list of average loss after each epoch.
        Note: different from HW10, we will return average losses instead of total losses.
    epoch_accuracies : list[float]
        A list of accuracy values after each epoch. This is applicable when training on MNIST.
    """
    epoch_average_losses = []
    epoch_accuracies = []
    for epoch in range(num_epoch):
        # Sum of losses in an epoch. 
        epoch_loss_sum = 0
        # Sum of the number of correct predictions.
        epoch_correct_num = 0
        # Iterate through batches.
        for X, Y in dataloader:
            output = model(X)
            optimizer.zero_grad()
            loss = loss_func(output, Y)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * X.size(0)

            if correct_num_func:
                epoch_correct_num += correct_num_func(output, Y)

        epoch_average_losses.append(epoch_loss_sum / len(dataloader.dataset))

        if correct_num_func:
            epoch_accuracies.append(epoch_correct_num / len(dataloader.dataset) * 100)

        if print_info:
            print(
                "Epoch: {} | Loss: {:.4f} ".format(
                    epoch, epoch_loss_sum / len(dataloader.dataset)
                ),
                end="",
            )
            if correct_num_func:
                print(
                    "Accuracy: {:.4f}%".format(
                        epoch_correct_num / len(dataloader.dataset) * 100
                    ),
                    end="",
                )
            print()

    if correct_num_func:
        return epoch_average_losses, epoch_accuracies
    else:
        return epoch_average_losses


def test(
    model: nn.Module,
    dataloader: DataLoader,
    loss_func: nn.MSELoss,
    correct_num_func: Callable = None,
) -> float | tuple[float, float]:
    """
    Tests the model.

    Parameters
    ----------
    model : torch.nn.Module
        A deep model.
    dataloader : torch.utils.data.DataLoader
        Dataloader of the testing set. Contains the testing data equivalent to ((Xi, Yi)),
        where (Xi, Yi) is a batch of data.
        X: 2D torch tensor for UCI wine and 4D torch tensor for MNIST.
        Y: 2D torch tensor for UCI wine and 1D torch tensor for MNIST, containing the corresponding labels
            for each example.
    loss_func : torch.nn.MSELoss
        An MSE loss function for UCI wine and a cross entropy loss for MNIST.
    correct_num_func : Callable, default=None
        A function to calculate how many samples are correctly classified.
        To test the CNN model, we also want to calculate the classification accuracy in addition to loss.

    Returns
    -------
    float
        Average loss.
    float
        Average accuracy. This is applicable when testing on MNIST.
    """
    with torch.no_grad():
        model.eval()
        epoch_loss_sum = 0
        epoch_correct_num = 0
        for X, Y in dataloader:
            output = model(X)
            loss = loss_func(output, Y)
            epoch_loss_sum += loss.item() * X.size(0)

            if correct_num_func:
                epoch_correct_num += correct_num_func(output, Y)

    avg_loss = epoch_loss_sum / len(dataloader.dataset)

    if correct_num_func:
        avg_acc = epoch_correct_num / len(dataloader.dataset) * 100
        return avg_loss, avg_acc
    else:
        return avg_loss


def correct_predict_num(logit: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns the number of correct predictions.

    Parameters
    ----------
    logit : torch.Tensor
        2D torch tensor of shape [n, class_num], where
        n is the number of samples, and class_num is the number of classes (10 for MNIST).
        Represents the output of CNN model.
    target : torch.Tensor
        1D torch tensor of shape [n],  where n is the number of samples.
        Represents the ground truth categories of images.

    Returns
    -------
    float
        A python scalar. The number of correct predictions.
    """
    pred = torch.argmax(logit, dim=1)
    correct = (pred == target.long()).sum()
    return correct.item()

