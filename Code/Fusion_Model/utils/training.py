"""
Training utilities — train loop, validation, early stopping, class weights.

Optimised for Apple Silicon (MPS backend) and CUDA.
Key optimisations:
  - Automatic Mixed Precision (AMP) via torch.autocast (float16)
  - channels-last memory format for faster convolutions
  - non-blocking host→device transfers
  - gradient clipping for stable mixed-precision training
  - MPS synchronisation helpers for accurate timing
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# -------------------------------------------------------------------------
# Apple Silicon / MPS helpers
# -------------------------------------------------------------------------

def setup_apple_silicon() -> None:
    """Apply MPS-specific environment tweaks. Call *before* creating tensors."""
    if not torch.backends.mps.is_available():
        return
    # Let the MPS allocator use all available unified memory
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    # Gracefully fall back to CPU for any op not yet implemented on MPS
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def mps_sync(device: torch.device) -> None:
    """Synchronise MPS command queue (needed for accurate wall-clock timings)."""
    if device.type == "mps":
        torch.mps.synchronize()


# -------------------------------------------------------------------------
# Class-weight computation
# -------------------------------------------------------------------------

def compute_class_weights(labels, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Inverse-frequency class weights for ``CrossEntropyLoss(weight=...)``.

    Parameters
    ----------
    labels : array-like of int
        All training-set labels.
    device : torch.device or str

    Returns
    -------
    torch.Tensor of shape (num_classes,)
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n_samples = len(labels)
    weights = []
    for c in sorted(classes):
        count = (labels == c).sum()
        weights.append(n_samples / (len(classes) * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


# -------------------------------------------------------------------------
# Single-epoch helpers
# -------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    channels_last: bool = True,
) -> tuple[float, float]:
    """Run one training epoch with optional mixed-precision (AMP).

    Parameters
    ----------
    use_amp : bool
        Automatic Mixed Precision (float16) on MPS / CUDA.
    grad_clip_norm : float
        Max gradient norm for clipping (stabilises mixed-precision training).
        Set to 0 to disable.
    channels_last : bool
        Convert image tensors to channels-last memory format (faster convolutions).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    amp_dtype = torch.float16
    amp_enabled = use_amp and device.type in ("mps", "cuda")

    for images, cbc, labels in loader:
        # --- Host → device (non-blocking on MPS/CUDA, channels-last for conv perf) ---
        if channels_last and device.type in ("mps", "cuda"):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        cbc    = cbc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # saves memory vs zero-fill

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images, cbc)
            loss   = criterion(logits, labels)

        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    channels_last: bool = True,
) -> tuple[float, float]:
    """Run validation with optional mixed-precision. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    amp_dtype = torch.float16
    amp_enabled = use_amp and device.type in ("mps", "cuda")

    for images, cbc, labels in loader:
        if channels_last and device.type in ("mps", "cuda"):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        cbc    = cbc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images, cbc)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# -------------------------------------------------------------------------
# Early stopping
# -------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Parameters
    ----------
    patience : int
        How many epochs to wait after last improvement.
    checkpoint_path : str
        Where to save the best model state dict.
    min_delta : float
        Minimum decrease in val loss to qualify as improvement.
    """

    def __init__(self, patience: int = 5, checkpoint_path: str = "best_model.pth", min_delta: float = 0.0):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter = 0
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns True when training should stop.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            torch.save(self.best_state, self.checkpoint_path)
            return False

        self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model: nn.Module):
        """Restore best weights into *model*."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        else:
            model.load_state_dict(torch.load(self.checkpoint_path, weights_only=True))
