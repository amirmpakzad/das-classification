import os
from glob import glob
import h5py
import numpy as np
from data_loader import fft  

def spec_cmp_1d(x_spec: np.ndarray) -> bool:
    split = int(len(x_spec) * 0.1)
    return (np.mean(x_spec[:split]) - np.mean(x_spec[split:])) > 0.05

def window_noise_ok(window_1d: np.ndarray, keep_if_regular: bool) -> bool:
    if keep_if_regular:
        return True
    x = np.fft.rfft(window_1d)[1:]
    x = np.log10(np.abs(x) + 1.0)
    return spec_cmp_1d(x)

def iter_windows(label_dir: str, fsize: int, shift: int):
    for ds_path in glob(os.path.join(label_dir, "*.h5")):
        with h5py.File(ds_path, "r") as f:
            data = f["Acquisition"]["Raw[0]"]["RawData"]
            bmp = np.load(ds_path[:-2] + "npy")

            idxs = np.transpose(np.where(bmp))

            for pos, ch in idxs:
                start = pos * shift
                end = start + fsize
                if end > data.shape[0]:
                    continue

                yield ds_path, int(pos), int(ch), data[start:end, ch].astype(np.float32)


def build_one_class(
    data_root: str,
    label: str,
    out_dir: str,
    sample_len: int,
    fsize: int,
    shift: int,
    drop_noise: bool,
    max_samples: int | None = None,
    seed: int = 42,
):
    import numpy as np
    rng = np.random.default_rng(seed)

    label_dir = os.path.join(data_root, label)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.h5")

    # -------- Pass 1: collect valid samples metadata --------
    candidates = []

    for src, pos, ch, w in iter_windows(label_dir, fsize, shift):
        if drop_noise:
            ok = window_noise_ok(w, keep_if_regular=(label == "regular"))
            if not ok:
                continue
        candidates.append((src, pos, ch, w))

    total_available = len(candidates)

    if total_available == 0:
        raise RuntimeError(f"No samples for label={label}")

    # -------- Apply cap if needed --------
    if max_samples is not None and total_available > max_samples:
        idx = rng.choice(total_available, size=max_samples, replace=False)
        candidates = [candidates[i] for i in idx]

    n = len(candidates)

    # -------- Feature dim --------
    dummy = np.zeros((1, fsize), dtype=np.float32)
    feat = fft(dummy)
    feat_dim = feat.shape[1]
    if sample_len:
        feat_dim = min(sample_len, feat_dim)

    str_dt = h5py.string_dtype(encoding="utf-8")

    # -------- Write HDF5 --------
    with h5py.File(out_path, "w") as hf:
        x_ds = hf.create_dataset(
            "x", shape=(n, feat_dim), dtype="float32",
            chunks=(min(1024, n), feat_dim)
        )
        pos_ds = hf.create_dataset("pos", shape=(n,), dtype="int32")
        ch_ds  = hf.create_dataset("ch",  shape=(n,), dtype="int32")
        src_ds = hf.create_dataset("src", shape=(n,), dtype=str_dt)

        idx = 0
        for src, pos, ch, w in candidates:
            x = fft(w.reshape(1, -1))
            if sample_len:
                x = x[:, :sample_len]

            x_ds[idx] = x[0]
            pos_ds[idx] = pos
            ch_ds[idx] = ch
            src_ds[idx] = os.path.basename(src)
            idx += 1

    print(f"[OK] {label}: wrote {n} samples (from {total_available}) -> {out_path}")


if __name__ == "__main__":
   
    data_root = os.path.abspath("../data")          
    out_dir   = os.path.abspath("../data/processed")   

    
    labels = ["construction", "manipulation", "fence"]

    # decimation per class
    #decim = {"regular":10}

    for label in labels:
        if label == "regular":
            cap = 25000
        else:
            cap = None
        build_one_class(
            data_root=data_root,
            label=label,
            out_dir=out_dir,
            sample_len=2048,
            fsize=8192,
            shift=2048,
            drop_noise=True,
            max_samples=cap,
        )

