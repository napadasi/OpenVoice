from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def safe_torch_load(path: str | Path, map_location: Optional[torch.device | str] = None) -> Any:
    path = Path(path)
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        return torch.from_numpy(array)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        data = load_file(path, device=map_location)
        if len(data) == 1:
            return next(iter(data.values()))
        return data
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Safe checkpoint loading requires PyTorch >= 2.0 (weights_only=True)."
        ) from exc


def safe_torch_save(tensor: torch.Tensor, path: str | Path) -> str:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file({"tensor": tensor.detach().cpu()}, path)
        return str(path)

    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    np.save(path, tensor.detach().cpu().numpy(), allow_pickle=False)
    return str(path)
