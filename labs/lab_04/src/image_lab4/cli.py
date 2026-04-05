from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from image_lab4.io.config_loader import load_config
from image_lab4.report.exporters import save_hdr, save_png, save_ppm
from image_lab4.services.path_tracer import PathTracer


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 4 path tracer CLI")
    parser.add_argument("--config", required=True, help="Path to JSON scene config")
    parser.add_argument("--output", required=True, help="Output PPM path")
    parser.add_argument("--png", help="Optional PNG path")
    parser.add_argument("--hdr", help="Optional HDR path for absolute radiance")
    parser.add_argument("--preview", action="store_true", help="Use fast preview settings")
    args = parser.parse_args()

    console = Console()
    config_path = Path(args.config)
    output_path = Path(args.output)
    config = load_config(config_path)
    strict_resolution = True
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
