from __future__ import annotations
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch


def plot_window(
    x: torch.Tensor,
    y: torch.Tensor,
    pred: torch.Tensor,
    idx: int,
    labels: List[str],
    run_id: str,
    save_dir: str = "../plots/",
    show: bool = False,
):
    x = x.detach().cpu().squeeze()

    y_i = int(y) if not isinstance(y, int) else y
    p_i = int(pred) if not isinstance(pred, int) else pred

    plt.figure()
    plt.plot(x, label="input window")
    plt.title(f"Class:{labels[y_i]} Pred_Class:{labels[p_i]}")
    plt.legend()

    if save_dir:
        out_dir = Path(save_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_dir / f"{idx}.png"), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()
