"""Training pipelines and tracking for FNO, U-Net, and baseline surrogates."""
import csv
import json
import os
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models.fno import FNO2d, CustomNormalizer, MaskedRelL2


def train_fno(
    model: FNO2d,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    water_mask_train: torch.Tensor,
    water_mask_val: torch.Tensor,
    epochs: int = 600,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints",
    model_tag: str = "fno_v1",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train FNO2d model with water-masked relative L2 loss and Cosine Annealing scheduler."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    mask_tr = water_mask_train.to(device)
    mask_va = water_mask_val.to(device) if val_loader is not None else mask_tr

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MaskedRelL2()

    os.makedirs(checkpoint_dir, exist_ok=True)
    csv_path = os.path.join(checkpoint_dir, f"{model_tag}_hist.csv")
    bench_path = os.path.join(checkpoint_dir, f"{model_tag}_bench.json")
    best_ckpt_path = os.path.join(checkpoint_dir, f"{model_tag}_best.pth")
    last_ckpt_path = os.path.join(checkpoint_dir, f"{model_tag}_last.pth")

    history = []
    best_val_loss = float("inf")
    start_time = time.time()

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "epoch_sec"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            n_batches = 0

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by, mask_tr)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(1, n_batches)
            scheduler.step()

            # Validation
            val_loss = train_loss
            if val_loader is not None:
                model.eval()
                v_loss = 0.0
                v_batches = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        v_pred = model(vx)
                        v_loss += criterion(v_pred, vy, mask_va).item()
                        v_batches += 1
                val_loss = v_loss / max(1, v_batches)

            ep_sec = time.time() - t0
            current_lr = scheduler.get_last_lr()[0]
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.6e}", f"{ep_sec:.2f}"])
            f_csv.flush()

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "epoch_sec": ep_sec})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "modes1": model.modes1,
                        "modes2": model.modes2,
                        "width": model.width,
                        "padding": model.padding,
                    },
                    best_ckpt_path,
                )

            if verbose and (epoch % 50 == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ({ep_sec:.1f}s)")

    # Save last checkpoint
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "last_val_loss": val_loss,
            "modes1": model.modes1,
            "modes2": model.modes2,
            "width": model.width,
            "padding": model.padding,
        },
        last_ckpt_path,
    )

    total_time = time.time() - start_time
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    bench_info = {
        "model_tag": model_tag,
        "epochs": epochs,
        "best_val_loss": float(best_val_loss),
        "total_time_sec": float(total_time),
        "avg_sec_per_epoch": float(total_time / epochs),
        "peak_gpu_mem_gb": float(peak_mem_gb),
    }
    with open(bench_path, "w", encoding="utf-8") as f_json:
        json.dump(bench_info, f_json, indent=2)

    return bench_info
