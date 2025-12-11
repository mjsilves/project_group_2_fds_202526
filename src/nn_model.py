import copy
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


def train_neural_network(
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        lr: float,
        n_hidden_layers: int = 1,
        n_epochs: int = 200,
        batch_size: int = 10,
        print_results: bool = True,
    ) -> tuple:
    """
    Train a NN regressor model with a pyramid structure,
    using ReLU activation function, MSE loss function,
    and ADAM optimizer.

    Args:
        X_train (Tensor): Tensor with training inputs.
        y_train (Tensor): Tensor with training targets.
        X_test (Tensor): Tensor with test inputs.
        y_test (Tensor): Tensor with test targets.
        lr (float): Learning rate.
        n_hidden_layers (int, optional): Number of hidden layers
            to include in the network structure (default: 1).
        n_epochs (int, optional): Total training epochs (default: 200).
        batch_size (int, optional): Batch size to process inputs during
            the training loop (default: 10).
        print_results (bool, optional): Whether to plot the model
            error metric scores across epochs (default: True).

    Returns:
        tuple: model, history
        The trained neural network model and the list of error metrics
        obtained throughout the training epochs.
    """

    n_input_features = X_train.shape[1]
    batch_start = torch.arange(0, X_train.shape[0], batch_size)

    hidden_layers = [
        (
            nn.Linear(n_input_features * (2 ** (i + 1)), n_input_features * (2 ** i)),
            nn.ReLU(),
        )
        for i in list(range(n_hidden_layers))[::-1]
    ]
    hidden_layers = [i for items in hidden_layers for i in items]

    model = nn.Sequential(
        nn.Linear(n_input_features, n_input_features * (2 ** n_hidden_layers)),
        nn.ReLU(),
        *hidden_layers,
        nn.Linear(n_input_features, 1)
    )

    # Loss function: Mean Square Error
    loss_fn = nn.MSELoss()

    # Optimizer: ADAM
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Variables to hold the best model
    best_rmse = np.inf
    best_weights = None
    history = []

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:

                # Pick data batch
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]

                # Forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Update weights
                optimizer.step()

                # Print progress
                bar.set_postfix(mse=float(loss))

        # Measure RMSE at the end of each epoch on the test set
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        rmse = float(mse) ** 0.5
        history.append(rmse)

        # In case of improvement, save model weights
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = copy.deepcopy(model.state_dict())

    # Restore best model weights found
    model.load_state_dict(best_weights)

    if print_results:
        print(f"RMSE: {best_rmse:.4f}")
        plt.plot(history)
        plt.title("Model metrics evolution")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.show()

    return model, history
