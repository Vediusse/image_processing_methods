from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class FilterGuide:
    depth: Optional[np.ndarray] = None
    object_ids: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None


@dataclass(frozen=True)
class FilterSettings:
    name: str = "bilateral"
    radius: int = 2
    sigma_spatial: float = 1.4
    sigma_color: float = 0.18
    sigma_depth: float = 0.08
    sigma_normal: float = 0.35
    strength: float = 1.0
    preserve_object_flux: bool = True
    median_rank: float = 0.5


class ImageFilterService:
    """Linear-light denoising filters for Monte-Carlo radiance buffers."""

    def __init__(self) -> None:
        self._filters: Dict[str, Callable[[np.ndarray, FilterSettings, Optional[FilterGuide]], np.ndarray]] = {
            "none": self._none,
            "box": self._box,
            "arithmetic": self._box,
            "gaussian": self._gaussian,
            "median": self._median,
            "bilateral": self._bilateral,
        }

    @property
    def names(self) -> Iterable[str]:
        return self._filters.keys()

    def apply(self, image: np.ndarray, settings: FilterSettings, guide: Optional[FilterGuide] = None) -> np.ndarray:
        name = settings.name.lower().strip()
        if name not in self._filters:
            raise ValueError("Unknown filter '{0}'. Available filters: {1}".format(name, ", ".join(sorted(self.names))))

        source = np.clip(np.asarray(image, dtype=np.float32), 0.0, None)
        if name == "none" or settings.strength <= 0.0 or settings.radius <= 0:
            return source.copy()

        filtered = self._filters[name](source, settings, guide)
        strength = float(np.clip(settings.strength, 0.0, 1.0))
        mixed = source * (1.0 - strength) + filtered * strength
        if settings.preserve_object_flux and name in {"median", "bilateral"}:
            object_ids = guide.object_ids if guide is not None else None
            mixed = _preserve_flux(source, mixed, object_ids)
        return np.clip(mixed, 0.0, None).astype(np.float32, copy=False)

    def _none(self, image: np.ndarray, settings: FilterSettings, guide: Optional[FilterGuide]) -> np.ndarray:
        return image.copy()

    def _box(self, image: np.ndarray, settings: FilterSettings, guide: Optional[FilterGuide]) -> np.ndarray:
        radius = _radius(settings.radius)
        kernel = np.full((2 * radius + 1, 2 * radius + 1), 1.0, dtype=np.float32)
        kernel /= float(kernel.sum())
        return _weighted_sum(image, kernel)

    def _gaussian(self, image: np.ndarray, settings: FilterSettings, guide: Optional[FilterGuide]) -> np.ndarray:
        radius = _radius(settings.radius)
        kernel = _gaussian_kernel(radius, settings.sigma_spatial)
        return _weighted_sum(image, kernel)

    def _median(self, image: np.ndarray, settings: FilterSettings, guide: Optional[FilterGuide]) -> np.ndarray:
        radius = _radius(settings.radius)
        padded = _pad_image(image, radius)
        rank = float(np.clip(settings.median_rank, 0.0, 1.0))
        object_ids = None if guide is None or guide.object_ids is None else np.asarray(guide.object_ids)
        padded_ids = None if object_ids is None else np.pad(object_ids, radius, mode="edge")
        height = image.shape[0]
        worker_count = min(max(1, os.cpu_count() or 1), 8, height)
        if worker_count <= 1 or height < 64:
            return _median_chunk(image, padded, padded_ids, radius, rank, 0, height)

        edges = np.linspace(0, height, worker_count + 1, dtype=int)
        chunks = [(int(edges[index]), int(edges[index + 1])) for index in range(worker_count) if edges[index] < edges[index + 1]]
        result = np.empty_like(image)
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(_median_chunk, image, padded, padded_ids, radius, rank, start, stop)
                for start, stop in chunks
            ]
            for (start, stop), future in zip(chunks, futures):
                result[start:stop] = future.result()
        return result

    def _bilateral(self, image: np.ndarray, settings: FilterSettings, guide: Optional[FilterGuide]) -> np.ndarray:
        radius = _radius(settings.radius)
        spatial = _gaussian_kernel(radius, settings.sigma_spatial)
        padded = _pad_image(image, radius)
        center = image
        center_luma = _luminance(center)
        depth = None if guide is None or guide.depth is None else np.asarray(guide.depth, dtype=np.float32)
        normals = None if guide is None or guide.normals is None else _normalize_normals(np.asarray(guide.normals, dtype=np.float32))
        object_ids = None if guide is None or guide.object_ids is None else np.asarray(guide.object_ids)
        padded_depth = None if depth is None else np.pad(depth, radius, mode="edge")
        padded_normals = None if normals is None else np.pad(normals, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        padded_ids = None if object_ids is None else np.pad(object_ids, radius, mode="edge")

        result = np.zeros_like(image)
        weights = np.zeros(image.shape[:2] + (1,), dtype=np.float32)
        guide_available = depth is not None or normals is not None or object_ids is not None
        luma_scale = max(float(np.percentile(center_luma, 95.0)), float(np.mean(center_luma) + np.std(center_luma)), 1e-4)
        color_sigma = max(float(settings.sigma_color) * luma_scale, float(settings.sigma_color) if guide_available else 1e-6)
        finite_depth = depth[np.isfinite(depth)] if depth is not None else np.array([], dtype=np.float32)
        depth_scale = max(float(np.percentile(finite_depth, 90.0)), 1.0) if finite_depth.size else 1.0
        depth_sigma = max(float(settings.sigma_depth) * depth_scale, 1e-6)
        normal_sigma = max(float(settings.sigma_normal), 1e-6)

        for ky in range(2 * radius + 1):
            for kx in range(2 * radius + 1):
                neighbor = padded[ky : ky + image.shape[0], kx : kx + image.shape[1]]
                neighbor_luma = _luminance(neighbor)
                color_delta = neighbor_luma - center_luma
                if guide_available:
                    color_weight = 1.0 / (1.0 + (color_delta / color_sigma) ** 2)
                else:
                    color_weight = np.exp(-0.5 * (color_delta / color_sigma) ** 2)
                weight = spatial[ky, kx] * color_weight

                if padded_depth is not None:
                    neighbor_depth = padded_depth[ky : ky + image.shape[0], kx : kx + image.shape[1]]
                    depth_delta = neighbor_depth - depth
                    valid_depth = np.isfinite(neighbor_depth) & np.isfinite(depth)
                    weight *= np.where(valid_depth, np.exp(-0.5 * (depth_delta / depth_sigma) ** 2), 0.0)

                if padded_normals is not None:
                    neighbor_normal = padded_normals[ky : ky + image.shape[0], kx : kx + image.shape[1]]
                    cosine = np.clip(np.sum(neighbor_normal * normals, axis=2), -1.0, 1.0)
                    weight *= np.exp(-0.5 * ((1.0 - cosine) / normal_sigma) ** 2)

                if padded_ids is not None:
                    neighbor_id = padded_ids[ky : ky + image.shape[0], kx : kx + image.shape[1]]
                    weight *= (neighbor_id == object_ids).astype(np.float32)

                weight3 = weight[:, :, None].astype(np.float32, copy=False)
                result += neighbor * weight3
                weights += weight3

        return result / np.maximum(weights, 1e-8)


def split_bilateral_denoise(
    direct: np.ndarray,
    secondary: np.ndarray,
    settings: FilterSettings,
    guide: Optional[FilterGuide],
) -> np.ndarray:
    service = ImageFilterService()
    if settings.name.lower().strip() == "bilateral":
        direct_settings = FilterSettings(
            name=settings.name,
            radius=settings.radius,
            sigma_spatial=settings.sigma_spatial,
            sigma_color=settings.sigma_color * 2.5,
            sigma_depth=settings.sigma_depth,
            sigma_normal=settings.sigma_normal,
            strength=settings.strength,
            preserve_object_flux=settings.preserve_object_flux,
            median_rank=settings.median_rank,
        )
    else:
        direct_settings = settings
    secondary_settings = FilterSettings(
        name=settings.name,
        radius=settings.radius,
        sigma_spatial=settings.sigma_spatial,
        sigma_color=settings.sigma_color * (8.0 if settings.name.lower().strip() == "bilateral" else 1.75),
        sigma_depth=settings.sigma_depth,
        sigma_normal=settings.sigma_normal,
        strength=settings.strength,
        preserve_object_flux=settings.preserve_object_flux,
        median_rank=settings.median_rank,
    )
    return service.apply(direct, direct_settings, guide) + service.apply(secondary, secondary_settings, guide)


def _radius(value: int) -> int:
    return max(1, int(value))


def _pad_image(image: np.ndarray, radius: int) -> np.ndarray:
    return np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")


def _gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    kernel = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma)).astype(np.float32)
    kernel /= float(kernel.sum())
    return kernel


def _weighted_sum(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = kernel.shape[0] // 2
    padded = _pad_image(image, radius)
    result = np.zeros_like(image)
    for ky in range(kernel.shape[0]):
        for kx in range(kernel.shape[1]):
            result += padded[ky : ky + image.shape[0], kx : kx + image.shape[1]] * kernel[ky, kx]
    return result


def _median_chunk(
    image: np.ndarray,
    padded: np.ndarray,
    padded_ids: Optional[np.ndarray],
    radius: int,
    rank: float,
    start: int,
    stop: int,
) -> np.ndarray:
    kernel_size = 2 * radius + 1
    window_count = kernel_size * kernel_size
    chunk = padded[start : stop + 2 * radius]
    windows = np.lib.stride_tricks.sliding_window_view(chunk, (kernel_size, kernel_size), axis=(0, 1))
    windows = np.moveaxis(windows, 2, -1).reshape(stop - start, image.shape[1], window_count, image.shape[2])

    luma = (
        windows[:, :, :, 0] * 0.2126
        + windows[:, :, :, 1] * 0.7152
        + windows[:, :, :, 2] * 0.0722
    )
    if padded_ids is not None:
        id_chunk = padded_ids[start : stop + 2 * radius]
        id_windows = np.lib.stride_tricks.sliding_window_view(id_chunk, (kernel_size, kernel_size))
        id_windows = id_windows.reshape(stop - start, image.shape[1], window_count)
        center_ids = padded_ids[start + radius : stop + radius, radius : radius + image.shape[1]][:, :, None]
        same_object = id_windows == center_ids
        center_rgb = image[start:stop, :, None, :]
        center_luma = _luminance(image[start:stop])[:, :, None]
        windows = np.where(same_object[:, :, :, None], windows, center_rgb)
        luma = np.where(same_object, luma, center_luma)

    rank_index = int(round(float(np.clip(rank, 0.0, 1.0)) * (window_count - 1)))
    order = np.argpartition(luma, rank_index, axis=2)[:, :, rank_index]
    selected = np.take_along_axis(windows, order[:, :, None, None], axis=2)
    return selected[:, :, 0, :].astype(np.float32, copy=False)


def _luminance(image: np.ndarray) -> np.ndarray:
    return image[:, :, 0] * 0.2126 + image[:, :, 1] * 0.7152 + image[:, :, 2] * 0.0722


def _normalize_normals(normals: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(normals, axis=2, keepdims=True)
    return normals / np.maximum(length, 1e-8)


def _preserve_flux(source: np.ndarray, filtered: np.ndarray, object_ids: Optional[np.ndarray]) -> np.ndarray:
    corrected = filtered.copy()
    if object_ids is None:
        source_sum = source.sum(axis=(0, 1), keepdims=True)
        filtered_sum = filtered.sum(axis=(0, 1), keepdims=True)
        return corrected * (source_sum / np.maximum(filtered_sum, 1e-8))

    for object_id in np.unique(object_ids):
        mask = object_ids == object_id
        if not np.any(mask):
            continue
        source_sum = source[mask].sum(axis=0)
        filtered_sum = filtered[mask].sum(axis=0)
        corrected[mask] *= source_sum / np.maximum(filtered_sum, 1e-8)
    return corrected
