import numpy as np

from image_lab4.services.image_filters import FilterGuide, FilterSettings, ImageFilterService


def test_box_filter_preserves_constant_linear_radiance() -> None:
    image = np.full((5, 5, 3), 2.5, dtype=np.float32)
    filtered = ImageFilterService().apply(image, FilterSettings(name="box", radius=1))
    assert np.allclose(filtered, image)


def test_bilateral_filter_respects_object_boundaries() -> None:
    image = np.zeros((3, 5, 3), dtype=np.float32)
    image[:, :2] = 1.0
    image[:, 2:] = 10.0
    object_ids = np.zeros((3, 5), dtype=np.int32)
    object_ids[:, 2:] = 1

    filtered = ImageFilterService().apply(
        image,
        FilterSettings(name="bilateral", radius=2, sigma_color=100.0, strength=1.0, preserve_object_flux=False),
        FilterGuide(object_ids=object_ids),
    )

    assert np.allclose(filtered[:, :2], 1.0)
    assert np.allclose(filtered[:, 2:], 10.0)


def test_median_flux_correction_preserves_object_energy() -> None:
    image = np.ones((3, 3, 3), dtype=np.float32)
    image[1, 1] = 9.0
    object_ids = np.zeros((3, 3), dtype=np.int32)

    filtered = ImageFilterService().apply(
        image,
        FilterSettings(name="median", radius=1, strength=1.0, preserve_object_flux=True),
        FilterGuide(object_ids=object_ids),
    )

    assert np.allclose(filtered.sum(axis=(0, 1)), image.sum(axis=(0, 1)), atol=1e-5)


def test_median_selects_existing_rgb_sample_by_intensity_rank() -> None:
    image = np.zeros((3, 3, 3), dtype=np.float32)
    image[:, :] = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    image[1, 1] = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    image[0, 0] = np.array([0.0, 3.0, 0.0], dtype=np.float32)

    filtered = ImageFilterService().apply(
        image,
        FilterSettings(name="median", radius=1, strength=1.0, preserve_object_flux=False),
    )

    unique_samples = {tuple(pixel) for row in image for pixel in row}
    assert tuple(filtered[1, 1]) in unique_samples


def test_guided_median_does_not_mix_object_windows() -> None:
    image = np.zeros((3, 5, 3), dtype=np.float32)
    image[:, :2] = 1.0
    image[:, 2:] = 10.0
    object_ids = np.zeros((3, 5), dtype=np.int32)
    object_ids[:, 2:] = 1

    filtered = ImageFilterService().apply(
        image,
        FilterSettings(name="median", radius=2, strength=1.0, preserve_object_flux=False),
        FilterGuide(object_ids=object_ids),
    )

    assert np.allclose(filtered[:, :2], 1.0)
    assert np.allclose(filtered[:, 2:], 10.0)
