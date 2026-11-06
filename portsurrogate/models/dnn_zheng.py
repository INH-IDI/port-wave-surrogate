"""Pointwise Deep Neural Network (DNN) surrogate based on Zheng et al. (2020).

MLP architecture [3-7-8-N] with tanh hidden activations and [-1, 1] min-max scaling.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim


class ZhengTorchMLP(nn.Module):
    """PyTorch implementation of the Zheng et al. MLP [input_dim - 7 - 8 - output_dim]."""

    def __init__(self, input_dim: int = 3, output_dim: int = 5):
        super(ZhengTorchMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 7),
            nn.Tanh(),
            nn.Linear(7, 8),
            nn.Tanh(),
            nn.Linear(8, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ZhengWaveModel:
    """Manages training and inference of the Zheng et al. (2020) model."""

    def __init__(self, input_dim: int = 3, output_dim: int = 5, lr: float = 1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        self.scalerX = MinMaxScaler(feature_range=(-1, 1))
        self.scalerY = MinMaxScaler(feature_range=(-1, 1))
        self.model: Optional[ZhengTorchMLP] = None

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        epochs: int = 500,
        batch_size: int = 32,
        patience: int = 50,
        verbose: bool = False,
    ):
        """Fit model with early stopping on validation MSE."""
        X_tr = self.scalerX.fit_transform(X_train)
        Y_tr = self.scalerY.fit_transform(Y_train)

        if X_val is not None and Y_val is not None:
            X_va = self.scalerX.transform(X_val)
            Y_va = self.scalerY.transform(Y_val)
            t_Xva = torch.tensor(X_va, dtype=torch.float32)
            t_Yva = torch.tensor(Y_va, dtype=torch.float32)
        else:
            t_Xva, t_Yva = None, None

        self.model = ZhengTorchMLP(self.input_dim, self.output_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        t_Xtr = torch.tensor(X_tr, dtype=torch.float32)
        t_Ytr = torch.tensor(Y_tr, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(t_Xtr, t_Ytr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            for b_x, b_y in loader:
                optimizer.zero_grad()
                pred = self.model(b_x)
                loss = criterion(pred, b_y)
                loss.backward()
                optimizer.step()

            # Validation step
            val_loss = loss.item()
            if t_Xva is not None and t_Yva is not None:
                self.model.eval()
                with torch.no_grad():
                    pred_va = self.model(t_Xva)
                    val_loss = criterion(pred_va, t_Yva).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Zheng early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X_eval: np.ndarray) -> np.ndarray:
        """Predict physical target values."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        X_norm = self.scalerX.transform(X_eval)
        t_X = torch.tensor(X_norm, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            t_pred = self.model(t_X)
            pred_norm = t_pred.numpy()

        return self.scalerY.inverse_transform(pred_norm)
