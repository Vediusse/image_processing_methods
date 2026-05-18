from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from rich.console import Console

from image_lab4.io.config_loader import load_config
from image_lab4.models.scene import DenoiseSettings
from image_lab4.report.exporters import save_hdr, save_png, save_ppm
from image_lab4.services.image_filters import ImageFilterService
from image_lab4.services.path_tracer import PathTracer


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 4 path tracer CLI")
    parser.add_argument("--config", required=True, help="Path to JSON scene config")
    parser.add_argument("--output", required=True, help="Output PPM path")
    parser.add_argument("--png", help="Optional PNG path")
    parser.add_argument("--hdr", help="Optional HDR path for absolute radiance")
    parser.add_argument("--preview", action="store_true", help="Use fast preview settings")
    parser.add_argument("--realtime", action="store_true", help="Use realtime raster preview backend")
    parser.add_argument("--gpu-pathtrace", action="store_true", help="Use Taichi Metal progressive Monte-Carlo path tracer")
    parser.add_argument("--frames", type=int, default=0, help="GPU frames/SPP override; 0 means use render.samples_per_pixel")
    parser.add_argument("--filter", choices=sorted(ImageFilterService().names), help="Linear-light post filter: none, box, gaussian, median, bilateral")
    parser.add_argument("--filter-radius", type=int, help="Filter window radius in pixels")
    parser.add_argument("--filter-strength", type=float, help="Blend factor from 0 to 1")
    parser.add_argument("--no-denoise", action="store_true", help="Disable radiance denoising/filtering")
    args = parser.parse_args()

    console = Console()
    config_path = Path(args.config)
    output_path = Path(args.output)
    config = load_config(config_path)
    if args.no_denoise or args.filter or args.filter_radius is not None or args.filter_strength is not None:
        config = replace(
            config,
            denoise=DenoiseSettings(
                enabled=not args.no_denoise,
                filter_name=args.filter or config.denoise.filter_name,
                radius=args.filter_radius if args.filter_radius is not None else config.denoise.radius,
                sigma_spatial=config.denoise.sigma_spatial,
                sigma_color=config.denoise.sigma_color,
                sigma_depth=config.denoise.sigma_depth,
                sigma_normal=config.denoise.sigma_normal,
                strength=args.filter_strength if args.filter_strength is not None else config.denoise.strength,
                preserve_object_flux=config.denoise.preserve_object_flux,
            ),
        )
    strict_resolution = True
    if args.realtime:
        from image_lab4.services.taichi_realtime import TaichiRealtimeRenderer

        renderer = TaichiRealtimeRenderer()
        frame_count = max(1, int(args.frames))
        times = []
        artifact = None
        for _ in range(frame_count):
            started = perf_counter()
            artifact = renderer.render(config, config_path=config_path)
            times.append(perf_counter() - started)
        assert artifact is not None
        save_ppm(output_path, artifact.display)
        if args.png:
            save_png(Path(args.png), artifact.display)
        if args.hdr:
            save_hdr(Path(args.hdr), artifact.radiance)
        console.print(artifact.summary)
        if frame_count > 1:
            warm_times = times[1:] or times
            average = sum(warm_times) / len(warm_times)
            console.print("Realtime frames: {0}; warm average: {1:.2f} ms; warm FPS: {2:.1f}".format(
                frame_count,
                average * 1000.0,
                1.0 / max(average, 1e-9),
            ))
        console.print(f"PPM saved to {output_path}")
        return

    if args.gpu_pathtrace:
        from image_lab4.services.taichi_progressive_path_tracer import TaichiProgressivePathTracer

        frame_count = int(args.frames) if int(args.frames) > 0 else int(config.render.samples_per_pixel)
        artifact = TaichiProgressivePathTracer().render_frames(
            config,
            frames=max(1, frame_count),
            config_path=config_path,
        )
        save_ppm(output_path, artifact.display)
        if args.png:
            save_png(Path(args.png), artifact.display)
        if args.hdr:
            save_hdr(Path(args.hdr), artifact.radiance)
        console.print(artifact.summary)
        console.print(f"PPM saved to {output_path}")
        return

    if args.preview:
        render = config.render.__class__(
            width=min(256, config.render.width),
            height=min(256, config.render.height),
            samples_per_pixel=1,
            max_depth=min(2, config.render.max_depth),
            min_depth=min(1, config.render.min_depth),
            gamma=config.render.gamma,
            normalization=config.render.normalization,
            normalization_value=config.render.normalization_value,
            seed=config.render.seed,
            background=config.render.background,
        )
        config = config.__class__(
            camera=config.camera,
            render=render,
            materials=config.materials,
            triangles=config.triangles,
            obj_meshes=config.obj_meshes,
            point_lights=config.point_lights,
            denoise=config.denoise,
        )
        strict_resolution = False
    artifact = PathTracer().render(config, config_path=config_path, strict_resolution=strict_resolution)
    save_ppm(output_path, artifact.display)
    if args.png:
        save_png(Path(args.png), artifact.display)
    if args.hdr:
        save_hdr(Path(args.hdr), artifact.radiance)
    console.print(artifact.summary)
    console.print(f"PPM saved to {output_path}")


if __name__ == "__main__":
    main()
