from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from modern_sc2 import Sc2Matrix, parse_modern_sc


IDENTITY = np.eye(3, dtype=np.float32)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def inline_texture_png_name(sc_stem: str, index: int) -> str:
    return f"{sc_stem}_{index:03d}.png"


def matrix_from_sc2(matrix: dict[str, Any] | Sc2Matrix | None) -> np.ndarray:
    if matrix is None:
        return IDENTITY.copy()
    if isinstance(matrix, Sc2Matrix):
        a = matrix.a
        b = matrix.b
        c = matrix.c
        d = matrix.d
        tx = matrix.tx
        ty = matrix.ty
    else:
        a = matrix["a"]
        b = matrix["b"]
        c = matrix["c"]
        d = matrix["d"]
        tx = matrix["tx"]
        ty = matrix["ty"]
    return np.array([[a, c, tx], [b, d, ty], [0.0, 0.0, 1.0]], dtype=np.float32)


def apply_matrix(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    out = matrix @ np.array([x, y, 1.0], dtype=np.float32)
    return float(out[0]), float(out[1])


def iter_strip_triangles(vertices: list[Any]) -> list[tuple[Any, Any, Any]]:
    triangles = []
    for index in range(len(vertices) - 2):
        if index % 2 == 0:
            triangle = (vertices[index], vertices[index + 1], vertices[index + 2])
        else:
            triangle = (vertices[index + 1], vertices[index], vertices[index + 2])
        triangles.append(triangle)
    return triangles


def load_texture_map(
    parsed: Any,
    raw_sc_path: Path,
    raw_sc_root: Path,
    sc_workspace: Path,
    sctx_decoded_root: Path,
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    workspace_data = sc_workspace / "data.json"
    if workspace_data.is_file():
        payload = json.loads(workspace_data.read_text(encoding="utf-8"))
        for texture in payload.get("textures", []):
            png_name = texture.get("png_name")
            index = texture.get("index")
            if png_name is None or index is None:
                continue
            png_path = sc_workspace / png_name
            if png_path.is_file():
                result[int(index)] = png_path

    for texture_set in parsed.textures:
        if texture_set.highres is None:
            continue
        index = texture_set.index
        if index in result:
            continue
        external = texture_set.highres.external_texture
        if not external:
            continue
        external_raw = (raw_sc_path.parent / external).resolve()
        try:
            relative = external_raw.relative_to(raw_sc_root)
        except ValueError:
            continue
        workspace = sctx_decoded_root / relative.with_suffix("")
        png_path = workspace / f"{workspace.name}.png"
        if png_path.is_file():
            result[index] = png_path
    return result


def decode_inline_raw_texture(texture: Any) -> Image.Image | None:
    if not texture.inline_data or texture.texture_format != 0:
        return None
    width = int(texture.width)
    height = int(texture.height)
    data = texture.inline_data
    pixel_type = int(texture.pixel_type)
    if pixel_type == 0:
        expected = width * height * 4
        if len(data) != expected:
            return None
        return Image.frombytes("RGBA", (width, height), data)
    if pixel_type == 2:
        expected = width * height * 2
        if len(data) != expected:
            return None
        packed = np.frombuffer(data, dtype=np.uint16)
        r = ((packed >> 12) & 0xF).astype(np.uint8) * 17
        g = ((packed >> 8) & 0xF).astype(np.uint8) * 17
        b = ((packed >> 4) & 0xF).astype(np.uint8) * 17
        a = (packed & 0xF).astype(np.uint8) * 17
        rgba = np.stack((r, g, b, a), axis=1).reshape(height, width, 4)
        return Image.fromarray(rgba, mode="RGBA")
    if pixel_type == 6:
        expected = width * height * 2
        if len(data) != expected:
            return None
        la = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 2)
        l = la[..., 0]
        a = la[..., 1]
        rgba = np.stack((l, l, l, a), axis=2)
        return Image.fromarray(rgba, mode="RGBA")
    return None


def materialize_inline_texture_pngs(parsed: Any, sc_workspace: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for texture_set in parsed.textures:
        texture = texture_set.highres
        if texture is None or texture.external_texture:
            continue
        png_path = sc_workspace / inline_texture_png_name(
            sc_workspace.name, texture_set.index
        )
        if png_path.is_file():
            result[texture_set.index] = png_path
            continue
        image = decode_inline_raw_texture(texture)
        if image is None:
            continue
        ensure_parent(png_path)
        image.save(png_path)
        result[texture_set.index] = png_path
    return result


def build_object_maps(parsed: Any) -> tuple[dict[int, Any], dict[int, Any]]:
    return (
        {shape.id: shape for shape in parsed.shapes},
        {movieclip.id: movieclip for movieclip in parsed.movieclips},
    )


def get_bank_matrix(parsed: Any, bank_index: int, matrix_index: int) -> np.ndarray:
    if matrix_index == 0xFFFF:
        return IDENTITY.copy()
    if bank_index < 0 or bank_index >= len(parsed.matrix_banks):
        return IDENTITY.copy()
    matrices = parsed.matrix_banks[bank_index]["matrices"]
    if matrix_index < 0 or matrix_index >= len(matrices):
        return IDENTITY.copy()
    return matrix_from_sc2(matrices[matrix_index])


def collect_bounds_for_object(
    object_id: int,
    frame_index: int,
    transform: np.ndarray,
    parsed: Any,
    shape_map: dict[int, Any],
    movieclip_map: dict[int, Any],
    depth: int = 0,
) -> tuple[float, float, float, float] | None:
    if depth > 32:
        return None
    if object_id in shape_map:
        xs: list[float] = []
        ys: list[float] = []
        for command in shape_map[object_id].commands:
            for vertex in command.vertices:
                x, y = apply_matrix(transform, vertex.x, vertex.y)
                xs.append(x)
                ys.append(y)
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    movieclip = movieclip_map.get(object_id)
    if movieclip is None or not movieclip.frames:
        return None
    frame = movieclip.frames[frame_index % len(movieclip.frames)]
    bounds: tuple[float, float, float, float] | None = None
    for element in frame.elements:
        if element.instance_index >= len(movieclip.children):
            continue
        child = movieclip.children[element.instance_index]
        local = get_bank_matrix(
            parsed, movieclip.matrix_bank_index, element.matrix_index
        )
        child_bounds = collect_bounds_for_object(
            child.object_id,
            frame_index,
            transform @ local,
            parsed,
            shape_map,
            movieclip_map,
            depth + 1,
        )
        if child_bounds is None:
            continue
        if bounds is None:
            bounds = child_bounds
        else:
            bounds = (
                min(bounds[0], child_bounds[0]),
                min(bounds[1], child_bounds[1]),
                max(bounds[2], child_bounds[2]),
                max(bounds[3], child_bounds[3]),
            )
    return bounds


def rasterize_triangle(
    canvas: np.ndarray,
    texture: np.ndarray,
    points: np.ndarray,
    uvs: np.ndarray,
) -> None:
    min_x = max(int(math.floor(float(points[:, 0].min()))), 0)
    min_y = max(int(math.floor(float(points[:, 1].min()))), 0)
    max_x = min(int(math.ceil(float(points[:, 0].max()))), canvas.shape[1] - 1)
    max_y = min(int(math.ceil(float(points[:, 1].max()))), canvas.shape[0] - 1)
    if min_x > max_x or min_y > max_y:
        return

    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denom)) < 1e-6:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denom
    w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denom
    w2 = 1.0 - w0 - w1
    mask = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
    if not np.any(mask):
        return

    tex_h, tex_w = texture.shape[:2]
    us = w0 * uvs[0, 0] + w1 * uvs[1, 0] + w2 * uvs[2, 0]
    vs = w0 * uvs[0, 1] + w1 * uvs[1, 1] + w2 * uvs[2, 1]
    sample_x = np.clip(np.rint(us * (tex_w - 1)).astype(np.int32), 0, tex_w - 1)
    sample_y = np.clip(np.rint(vs * (tex_h - 1)).astype(np.int32), 0, tex_h - 1)

    region = canvas[min_y : max_y + 1, min_x : max_x + 1]
    sampled = texture[sample_y, sample_x].astype(np.float32) / 255.0
    alpha = sampled[..., 3:4]
    region[:] = np.where(
        mask[..., None],
        np.concatenate(
            [
                sampled[..., :3] * alpha + region[..., :3] * (1.0 - alpha),
                alpha + region[..., 3:4] * (1.0 - alpha),
            ],
            axis=2,
        ),
        region,
    )


def draw_shape(
    canvas: np.ndarray,
    shape: Any,
    transform: np.ndarray,
    textures: dict[int, np.ndarray],
    bounds: tuple[float, float, float, float],
    padding: int,
) -> None:
    min_x, min_y, _, _ = bounds
    for command in shape.commands:
        texture = textures.get(command.texture_index)
        if texture is None:
            continue
        for triangle in iter_strip_triangles(command.vertices):
            points = []
            uvs = []
            for vertex in triangle:
                x, y = apply_matrix(transform, vertex.x, vertex.y)
                points.append((x - min_x + padding, y - min_y + padding))
                uvs.append((vertex.u, vertex.v))
            rasterize_triangle(
                canvas,
                texture,
                np.asarray(points, dtype=np.float32),
                np.asarray(uvs, dtype=np.float32),
            )


def draw_object(
    canvas: np.ndarray,
    object_id: int,
    frame_index: int,
    transform: np.ndarray,
    parsed: Any,
    shape_map: dict[int, Any],
    movieclip_map: dict[int, Any],
    textures: dict[int, np.ndarray],
    bounds: tuple[float, float, float, float],
    padding: int,
    depth: int = 0,
) -> None:
    if depth > 32:
        return
    shape = shape_map.get(object_id)
    if shape is not None:
        draw_shape(canvas, shape, transform, textures, bounds, padding)
        return

    movieclip = movieclip_map.get(object_id)
    if movieclip is None or not movieclip.frames:
        return
    frame = movieclip.frames[frame_index % len(movieclip.frames)]
    for element in frame.elements:
        if element.instance_index >= len(movieclip.children):
            continue
        child = movieclip.children[element.instance_index]
        local = get_bank_matrix(
            parsed, movieclip.matrix_bank_index, element.matrix_index
        )
        draw_object(
            canvas,
            child.object_id,
            frame_index,
            transform @ local,
            parsed,
            shape_map,
            movieclip_map,
            textures,
            bounds,
            padding,
            depth + 1,
        )


def load_texture_arrays(texture_map: dict[int, Path]) -> dict[int, np.ndarray]:
    result = {}
    for index, path in texture_map.items():
        result[index] = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    return result


def render_exports(
    raw_sc_path: Path,
    raw_sc_root: Path,
    sc_workspace: Path,
    sctx_decoded_root: Path,
    output_root: Path,
    only_exports: set[str] | None = None,
    max_frames_per_export: int | None = None,
) -> dict[str, Any]:
    parsed = parse_modern_sc(raw_sc_path)
    texture_map = load_texture_map(
        parsed,
        raw_sc_path,
        raw_sc_root,
        sc_workspace,
        sctx_decoded_root,
    )
    for index, png_path in materialize_inline_texture_pngs(
        parsed, sc_workspace
    ).items():
        texture_map.setdefault(index, png_path)
    textures = load_texture_arrays(texture_map)
    shape_map, movieclip_map = build_object_maps(parsed)
    export_names = [
        item["name"]
        for item in parsed.exports
        if item["kind"] == "movieclip" and item.get("name")
    ]
    if only_exports is not None:
        export_names = [name for name in export_names if name in only_exports]

    stats = {
        "sc": raw_sc_path.name,
        "workspace": str(sc_workspace),
        "textures_resolved": len(texture_map),
        "exports_found": len(export_names),
        "exports_rendered": 0,
        "frames_rendered": 0,
        "skipped_empty": [],
    }

    for export_name in export_names:
        movieclip = next(
            (item for item in parsed.movieclips if item.export_name == export_name),
            None,
        )
        if movieclip is None or not movieclip.frames:
            continue
        export_dir = output_root / export_name
        rendered_any = False
        frames = movieclip.frames
        if max_frames_per_export is not None and max_frames_per_export >= 0:
            frames = frames[:max_frames_per_export]
        for frame in frames:
            bounds = collect_bounds_for_object(
                movieclip.id,
                frame.index,
                IDENTITY,
                parsed,
                shape_map,
                movieclip_map,
            )
            if bounds is None:
                continue
            min_x, min_y, max_x, max_y = bounds
            width = max(1, int(math.ceil(max_x - min_x)) + 8)
            height = max(1, int(math.ceil(max_y - min_y)) + 8)
            if width <= 8 or height <= 8:
                continue
            canvas = np.zeros((height, width, 4), dtype=np.float32)
            draw_object(
                canvas,
                movieclip.id,
                frame.index,
                IDENTITY,
                parsed,
                shape_map,
                movieclip_map,
                textures,
                bounds,
                4,
            )
            image = Image.fromarray(
                np.clip(np.rint(canvas * 255.0), 0, 255).astype(np.uint8),
                mode="RGBA",
            )
            if image.getbbox() is None:
                continue
            frame_path = export_dir / f"frame_{frame.index:04d}.png"
            ensure_parent(frame_path)
            image.save(frame_path)
            stats["frames_rendered"] += 1
            rendered_any = True
        if rendered_any:
            stats["exports_rendered"] += 1
        else:
            stats["skipped_empty"].append(export_name)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Render modern SC2 exports to PNG")
    parser.add_argument("raw_sc", type=Path, help="Path to raw .sc file")
    parser.add_argument(
        "sc_workspace", type=Path, help="Decoded .sc workspace directory"
    )
    parser.add_argument(
        "sctx_decoded_root",
        type=Path,
        help="Root of decoded .sctx workspaces matching raw assets/sc",
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for rendered exports"
    )
    parser.add_argument(
        "--raw-sc-root",
        type=Path,
        required=True,
        help="Root raw assets/sc directory for external texture resolution",
    )
    parser.add_argument(
        "--export",
        action="append",
        dest="exports",
        help="Optional export name filter; can be passed multiple times",
    )
    parser.add_argument(
        "--max-frames-per-export",
        type=int,
        help="Optional frame cap per export for faster batch extraction",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    stats = render_exports(
        raw_sc_path=args.raw_sc,
        raw_sc_root=args.raw_sc_root,
        sc_workspace=args.sc_workspace,
        sctx_decoded_root=args.sctx_decoded_root,
        output_root=args.output_dir,
        only_exports=set(args.exports) if args.exports else None,
        max_frames_per_export=args.max_frames_per_export,
    )
    text = json.dumps(stats, indent=2, sort_keys=False)
    if args.report:
        ensure_parent(args.report)
        args.report.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
