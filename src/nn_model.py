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
    best_rmse_train = np.inf
    best_weights = None

    history = []
    history_train = []

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

        y_pred_train = model(X_train)
        mse_train = loss_fn(y_pred_train, y_train)
        rmse_train = float(mse_train) ** 0.5
        history_train.append(rmse_train)

        # In case of improvement, save model weights
        if rmse < best_rmse:
            best_rmse = rmse
            best_rmse_train = rmse_train
            best_weights = copy.deepcopy(model.state_dict())

    # Restore best model weights found
    model.load_state_dict(best_weights)

    if print_results:
        print(f"RMSE (train): {best_rmse_train:.4f}\nRMSE (test): {best_rmse:.4f}")
        plt.plot(history)
        plt.title("Model metrics evolution")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.show()

    return model, history, history_train, best_rmse, best_rmse_train


def visualize_3_param_grid_search_results(
        param_1_int: Iterable,
        param_1_name: str,
        param_2_int: Iterable,
        param_2_name: str,
        param_3_float: Iterable,
        param_3_name: str,
        rmse: Iterable,
    ) -> None:
    """
    Visualize 3-parameter grid search results with:
      (1) 3D surface, where color corresponds to RMSE score
      (2) 2D heatmap of RMSE on int x int grid for fixed float parameter slices

    Args:
        param_1_int (Iterable): Values of the integer parameter 1.
        param_1_name (str): Name of the integer parameter 1.
        param_2_int (Iterable): Values of the integer parameter 2.
        param_2_name (str): Name of the integer parameter 2.
        param_3_float (Iterable): Values of the float parameter 3.
        param_3_name (str): Name of the float parameter 3.
        rmse (Iterable): RMSE values.
    """

    p1 = np.array(param_1_int)
    p2 = np.array(param_2_int)
    p3 = np.array(param_3_float)
    rmse = np.array(rmse)

    # 3D scatter plot colored by RMSE

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121, projection="3d")

    sc = ax.scatter(p1, p2, p3, c=rmse, cmap="viridis", s=50)

    ax.set_xlabel(param_1_name)
    ax.set_ylabel(param_2_name)
    ax.set_zlabel(param_3_name)
    ax.set_title("Parameter space (color = RMSE)")
    fig.colorbar(sc, ax=ax, shrink=0.5, label="RMSE")

    # 2D heatmap by averaged LR slices

    p1_grid = np.unique(p1)
    p2_grid = np.unique(p2)

    Z = np.full((len(p2_grid), len(p1_grid)), np.nan)

    for i, p2_g in enumerate(p2_grid):
        for j, p1_g in enumerate(p1_grid):
            mask = (p1 == p1_g) & (p2 == p2_g)
            if np.any(mask):
                Z[i, j] = np.mean(rmse[mask])   # respective avg LR

    ax2 = fig.add_subplot(122)

    hmap = ax2.imshow(Z, origin="lower", cmap="viridis", aspect="auto")

    ax2.set_xticks(range(len(p1_grid)))
    ax2.set_yticks(range(len(p2_grid)))
    ax2.set_xticklabels(p1_grid)
    ax2.set_yticklabels(p2_grid)
    ax2.set_xlabel(param_1_name)
    ax2.set_ylabel(param_2_name)
    ax2.set_title(f"2D Heatmap (RMSE averaged over {param_3_name})")
    fig.colorbar(hmap, ax=ax2, shrink=0.5, label="RMSE")

    plt.tight_layout()
    plt.show()
