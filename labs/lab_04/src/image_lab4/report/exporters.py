from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_ppm(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    height, width, _ = clipped.shape
    with path.open("wb") as stream:
        stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        stream.write(clipped.tobytes())


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image / 255.0, 0.0, 1.0))


def save_hdr(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image, dtype=np.float32)
    height, width, _ = image.shape
    header = (
        "#?RADIANCE\n"
        "FORMAT=32-bit_rle_rgbe\n\n"
        f"-Y {height} +X {width}\n"
    ).encode("ascii")
    rgbe = _float_to_rgbe(image)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(rgbe.tobytes())


def _float_to_rgbe(image: np.ndarray) -> np.ndarray:
    rgbe = np.zeros(image.shape[:2] + (4,), dtype=np.uint8)
    max_channel = np.max(image, axis=2)
    valid = max_channel > 1e-32
    if not np.any(valid):
        return rgbe
    mantissa, exponent = np.frexp(max_channel[valid])
    scale = mantissa * 256.0 / np.maximum(max_channel[valid], 1e-32)
    rgbe_valid = np.zeros((scale.shape[0], 4), dtype=np.uint8)
    rgbe_valid[:, 0:3] = np.clip(image[valid] * scale[:, None], 0, 255).astype(np.uint8)
    rgbe_valid[:, 3] = np.clip(exponent + 128, 0, 255).astype(np.uint8)
    rgbe[valid] = rgbe_valid
    return rgbe
