from __future__ import annotations

import random

import numpy as np
import torch
from models import CNN, OneLayerNN, TwoLayerNN, correct_predict_num, test, train
from torch import nn, optim
from utils import (
    get_mnist_loader,
    get_wine_loader,
    visualize_accuracy,
    visualize_confusion_matrix,
    visualize_image,
    visualize_loss,
    visualize_misclassified_image,
)


def test_linear_nn(test_size: float = 0.2, nn_type: str = "one_layer") -> float:
    """
    Tests OneLayerNN, TwoLayerNN on the wine dataset.

    Parameters
    ----------
    test_size : float, default=0.2
        The ratio of test set w.r.t. the whole dataset.
    nn_type : str, default='one_layer'
        'one_layer': test OneLayerNN
        'two_layer': test TwoLayerNN

    Returns
    -------
    float
        Loss on test set.
    """
    assert nn_type in ["one_layer", "two_layer"]
    if nn_type == "one_layer":
        # Hyper-parameters of OneLayerNN
        batch_size = 64  # batch size
        num_epoch = 25  # number of training epochs
        learning_rate = 0.01  # learning rate
    else:
        batch_size = 64  # batch size
        num_epoch = 25  # number of training epochs
        learning_rate = 0.01  # learning rate
    pass

    # Load data
    dataloader_train, dataloader_test = get_wine_loader(
        batch_size=batch_size, test_size=test_size
    )

    # Initialize model
    if nn_type == "one_layer":
        model = OneLayerNN(input_features=11)
        # Initialize optimizer
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        model = TwoLayerNN(input_features=11)
        # Initialize optimizer
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize the MSE (i.e., L2) loss function
    loss_func = nn.MSELoss()

    losses = train(model, dataloader_train, loss_func, optimizer, num_epoch)

    # Uncomment to visualize training losses
    # visualize_loss(losses)

    # Average training/testing loss
    loss_train = test(model, dataloader_train, loss_func)
    loss_test = test(model, dataloader_test, loss_func)
    print("Average Training Loss:", loss_train)
    print("Average Testing Loss:", loss_test)

    return loss_test


def test_cnn(test_size: float = 0.2) -> float:
    """
    Tests CNN on the MNIST dataset.

    Parameters
    ----------
    test_size : float, default=0.2
        The ratio of test set w.r.t. the whole dataset.

    Returns
    -------
    float
        Accuracy on test set.
    """
    # Hyper-parameters of CNN
    batch_size = 64  # batch size
    num_epoch = 280  # number of training epochs
    learning_rate = 0.01  # learning rate

    # Load data
    shuffle_train_label = True
    dataloader_train, dataloader_test = get_mnist_loader(
        batch_size=batch_size,
        test_size=test_size,
        shuffle_train_label=shuffle_train_label,
    )

    # Initialize model
    model = CNN()

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize the cross entropy loss function
    loss_func = nn.CrossEntropyLoss()

    losses, accuracies = train(
        model,
        dataloader_train,
        loss_func,
        optimizer,
        num_epoch,
        correct_num_func=correct_predict_num,
    )

    # Average training/testing loss/accuracy
    loss_train, accuracy_train = test(
        model, dataloader_train, loss_func, correct_num_func=correct_predict_num
    )
    loss_test, accuracy_test = test(
        model, dataloader_test, loss_func, correct_num_func=correct_predict_num
    )
    print(
        "Average Training Loss: {:.4f} | Average Training Accuracy: {:.4f}%".format(
            loss_train, accuracy_train * 100
        )
    )
    print(
        "Average Testing Loss: {:.4f} | Average Testing Accuracy: {:.4f}%".format(
            loss_test, accuracy_test * 100
        )
    )

    # Uncomment to visualize training losses, training accuracies, test set images,
    # misclassified test set images, and confusion matrix
    # visualize_loss(losses)
    # visualize_accuracy(accuracies)
    # visualize_image(model, dataloader_test, 4, 4)
    # visualize_misclassified_image(model, dataloader_test, 4, 4)
    # visualize_confusion_matrix(model, dataloader_test)

    return accuracy_test


def main():
    # Set random seeds.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Uncomment to test your models
    # test_linear_nn(nn_type='one_layer')
    # test_linear_nn(nn_type='two_layer')
    test_cnn()


if __name__ == "__main__":
    main()
