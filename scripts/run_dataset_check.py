import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from create_torch_dataset import PerClassH5Dataset

def main():
    ds = PerClassH5Dataset("../data/processed")
    print("classes:", ds.classes)
    print("len:", len(ds))

    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    xb, yb = next(iter(dl))
    print("batch x:", xb.shape, xb.dtype)
    print("batch y:", yb.shape, yb.dtype, "min/max:", int(yb.min()), int(yb.max()))

if __name__ == "__main__":
    main()
