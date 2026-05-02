"""
MLP token-importance predictor and trainer.

Architecture: Linear(7→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1).
forward() returns logits; apply sigmoid for probabilities at inference.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from token_importance.features import FEATURE_COLS


class ImportanceModel(nn.Module):
    def __init__(self, input_dim: int = len(FEATURE_COLS)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Trainer:
    def __init__(
        self,
        *,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 2048,
        patience: int = 7,
        device: str | None = None,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """
        Fit StandardScaler + model on training data.

        Returns dict with keys: model, scaler, best_val_auc, history.
        The scaler is fit on x_train only; caller must save both artifacts together.
        """
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_train).astype(np.float32)
        x_va = scaler.transform(x_val).astype(np.float32)

        dev = self.device
        t_x = torch.from_numpy(x_tr).to(dev)
        t_y = torch.from_numpy(y_train.astype(np.float32)).to(dev)
        v_x = torch.from_numpy(x_va).to(dev)
        v_y_np = y_val.astype(np.float32)

        # Upweight minority positive class (~10.4% positive under per-head labels).
        # n_neg/n_pos is computed dynamically so any shift in label balance is handled.
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        loader = DataLoader(
            TensorDataset(t_x, t_y),
            batch_size=self.batch_size,
            shuffle=True,
        )

        model = ImportanceModel().to(dev)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_auc = 0.0
        best_state: dict | None = None
        no_improve = 0
        history: list[dict] = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            train_loss = epoch_loss / len(t_x)

            model.eval()
            with torch.no_grad():
                val_logits = model(v_x).cpu().numpy()
            val_probs = torch.sigmoid(torch.from_numpy(val_logits)).numpy()
            val_auc = float(roc_auc_score(v_y_np, val_probs))

            history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_auc={val_auc:.4f}")

            if val_auc > best_val_auc + 1e-4:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"early stop at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to("cpu")

        return {
            "model": model,
            "scaler": scaler,
            "best_val_auc": best_val_auc,
            "history": history,
        }
