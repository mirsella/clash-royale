#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/home/mirsella/dev/apks")
DEFAULT_SOURCE_ROOT = (
    ROOT / "extracted" / "com.supercell.clashroyale_phone_arm64" / "organized"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "extracted" / "com.supercell.clashroyale_phone_arm64" / "public_flat"
)
ENCODED_CONTAINER_SUFFIXES = {".sc", ".sctx", ".ktx", ".zktx"}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_json(path: Path, data: object) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def flatten_rel_path(rel: Path) -> str:
    parts = [part for part in rel.parts if part not in ("", ".")]
    if not parts:
        return "root"
    return "__".join(parts)


def add_index(
    index: dict[str, str], dst: Path, output_root: Path, src: Path, source_root: Path
) -> None:
    index[str(dst.relative_to(output_root))] = str(src.relative_to(source_root))


def load_glb_tools() -> tuple[object, object]:
    sys.path.insert(0, str(ROOT))
    from merge_supercell_animation_glbs import read_glb, write_glb  # type: ignore

    return read_glb, write_glb


def copy_flat_files(
    src_root: Path,
    dst_root: Path,
    output_root: Path,
    source_root: Path,
    index: dict[str, str],
    counts: Counter,
    suffixes: set[str] | None = None,
) -> None:
    if not src_root.is_dir():
        return
    for src in sorted(path for path in src_root.rglob("*") if path.is_file()):
        if suffixes is not None and src.suffix.lower() not in suffixes:
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / flatten_rel_path(rel)
        hardlink_or_copy(src, dst)
        add_index(index, dst, output_root, src, source_root)
        counts[str(dst_root.relative_to(output_root))] += 1


def copy_flat_workspaces(
    src_root: Path,
    dst_root: Path,
    output_root: Path,
    source_root: Path,
    index: dict[str, str],
    counts: Counter,
) -> None:
    if not src_root.is_dir():
        return
    workspace_dirs = sorted({path.parent for path in src_root.rglob("data.json")})
    for workspace in workspace_dirs:
        rel_dir = workspace.relative_to(src_root)
        flat_dir = dst_root / flatten_rel_path(rel_dir)
        for src in sorted(path for path in workspace.rglob("*") if path.is_file()):
            if src.suffix.lower() in ENCODED_CONTAINER_SUFFIXES:
                continue
            rel_file = src.relative_to(workspace)
            dst = flat_dir / rel_file
            hardlink_or_copy(src, dst)
            add_index(index, dst, output_root, src, source_root)
            counts[str(dst_root.relative_to(output_root))] += 1


def build_flat_texture_map(
    src_root: Path,
    dst_root: Path,
    output_root: Path,
    source_root: Path,
    index: dict[str, str],
    counts: Counter,
) -> dict[Path, Path]:
    mapping: dict[Path, Path] = {}
    if not src_root.is_dir():
        return mapping
    for src in sorted(src_root.rglob("*.png")):
        rel = src.relative_to(src_root)
        dst = dst_root / flatten_rel_path(rel)
        hardlink_or_copy(src, dst)
        mapping[src.resolve()] = dst.resolve()
        add_index(index, dst, output_root, src, source_root)
        counts[str(dst_root.relative_to(output_root))] += 1
    return mapping


def rewrite_glb_image_uris(
    json_obj: dict,
    src_path: Path,
    dst_path: Path,
    texture_map: dict[Path, Path],
) -> tuple[bool, int]:
    changed = False
    missing = 0
    for image in json_obj.get("images", []):
        uri = image.get("uri")
        if not isinstance(uri, str) or not uri or uri.startswith("data:"):
            continue
        resolved = (src_path.parent / uri).resolve()
        mapped = texture_map.get(resolved)
        if mapped is None:
            missing += 1
            continue
        new_uri = os.path.relpath(mapped, start=dst_path.parent)
        if new_uri != uri:
            image["uri"] = new_uri
            changed = True
    return changed, missing


def copy_flat_glbs(
    src_root: Path,
    dst_root: Path,
    output_root: Path,
    source_root: Path,
    index: dict[str, str],
    counts: Counter,
    texture_map: dict[Path, Path],
    read_glb,
    write_glb,
) -> int:
    missing_refs = 0
    if not src_root.is_dir():
        return missing_refs
    for src in sorted(src_root.rglob("*.glb")):
        rel = src.relative_to(src_root)
        dst = dst_root / flatten_rel_path(rel)
        glb = read_glb(src)
        changed, missing = rewrite_glb_image_uris(glb.json_obj, src, dst, texture_map)
        missing_refs += missing
        if changed:
            write_glb(dst, glb.json_obj, glb.bin_chunk)
        else:
            hardlink_or_copy(src, dst)
        add_index(index, dst, output_root, src, source_root)
        counts[str(dst_root.relative_to(output_root))] += 1
    return missing_refs


def copy_report_file(
    src: Path,
    dst: Path,
    output_root: Path,
    source_root: Path,
    index: dict[str, str],
    counts: Counter,
) -> None:
    if not src.is_file():
        return
    hardlink_or_copy(src, dst)
    add_index(index, dst, output_root, src, source_root)
    counts[str(dst.parent.relative_to(output_root))] += 1


def write_text_report(path: Path, report: dict) -> None:
    lines = [
        "Clash Royale flatter public dump",
        "===============================",
        "",
        f"Source dump: {report['source_dump']}",
        f"Flat dump: {report['flat_dump']}",
        "",
        "Layout",
        "------",
        "Files are flattened by replacing original relative path separators with '__'.",
        "Decoded `.sc` and `.sctx` workspaces stay as single directories because they contain multiple related files.",
        "Decoded and merged GLBs had image URIs rewritten to point at the flat `textures/png` bucket.",
        "",
        "Counts",
        "------",
    ]
    for bucket, count in sorted(report["counts"].items()):
        lines.append(f"{bucket}: {count}")
    lines.extend(
        [
            "",
            "Paths",
            "-----",
        ]
    )
    for label, rel in sorted(report["paths"].items()):
        lines.append(f"{label}: {rel}")
    lines.extend(
        [
            "",
            f"GLB image URI rewrites missing flat texture target: {report['glb_missing_texture_uris']}",
            f"Indexed output paths: {report['indexed_paths']}",
        ]
    )
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a much flatter public Clash Royale asset dump"
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source_root.is_dir():
        raise SystemExit(f"missing source root: {args.source_root}")

    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    read_glb, write_glb = load_glb_tools()

    index: dict[str, str] = {}
    counts: Counter = Counter()

    texture_map = build_flat_texture_map(
        args.source_root / "sc3d" / "textures" / "png",
        args.output_root / "textures" / "png",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    glb_missing_texture_uris = 0
    glb_missing_texture_uris += copy_flat_glbs(
        args.source_root / "sc3d" / "models" / "decoded",
        args.output_root / "models" / "decoded",
        args.output_root,
        args.source_root,
        index,
        counts,
        texture_map,
        read_glb,
        write_glb,
    )
    glb_missing_texture_uris += copy_flat_glbs(
        args.source_root / "sc3d" / "models" / "merged",
        args.output_root / "models" / "merged",
        args.output_root,
        args.source_root,
        index,
        counts,
        texture_map,
        read_glb,
        write_glb,
    )
    copy_flat_files(
        args.source_root / "previews" / "decoded",
        args.output_root / "previews" / "decoded",
        args.output_root,
        args.source_root,
        index,
        counts,
        {".mp4"},
    )
    copy_flat_files(
        args.source_root / "previews" / "merged",
        args.output_root / "previews" / "merged",
        args.output_root,
        args.source_root,
        index,
        counts,
        {".mp4"},
    )

    copy_flat_files(
        args.source_root / "audio" / "ogg",
        args.output_root / "audio" / "ogg",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_flat_files(
        args.source_root / "audio" / "bank",
        args.output_root / "audio" / "bank",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_flat_files(
        args.source_root / "fonts" / "ttf",
        args.output_root / "fonts" / "ttf",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_flat_files(
        args.source_root / "fonts" / "fnt",
        args.output_root / "fonts" / "fnt",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_flat_files(
        args.source_root / "images" / "png",
        args.output_root / "images" / "png",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_flat_files(
        args.source_root / "images" / "pvr",
        args.output_root / "images" / "pvr",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_flat_files(
        args.source_root / "materials" / "rmat",
        args.output_root / "materials",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_flat_files(
        args.source_root / "shaders" / "shader",
        args.output_root / "shaders",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_flat_files(
        args.source_root / "shaders" / "fsh",
        args.output_root / "shaders",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_flat_files(
        args.source_root / "shaders" / "vsh",
        args.output_root / "shaders",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_flat_files(
        args.source_root / "special" / "scw",
        args.output_root / "special",
        args.output_root,
        args.source_root,
        index,
        counts,
    )

    copy_flat_workspaces(
        args.source_root / "supercell_sc" / "decoded",
        args.output_root / "sprites" / "sc" / "decoded",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_flat_workspaces(
        args.source_root / "supercell_sctx" / "decoded",
        args.output_root / "sprites" / "sctx" / "decoded",
        args.output_root,
        args.source_root,
        index,
        counts,
    )

    copy_report_file(
        args.source_root / "reports" / "report.json",
        args.output_root / "reports" / "source_report.json",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_report_file(
        args.source_root / "reports" / "report.txt",
        args.output_root / "reports" / "source_report.txt",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_report_file(
        args.source_root / "previews" / "decoded" / "render_report.txt",
        args.output_root / "reports" / "decoded_render_report.txt",
        args.output_root,
        args.source_root,
        index,
        counts,
    )
    copy_report_file(
        args.source_root / "previews" / "merged" / "render_report.txt",
        args.output_root / "reports" / "merged_render_report.txt",
        args.output_root,
        args.source_root,
        index,
        counts,
    )

    report = {
        "source_dump": str(args.source_root.relative_to(args.source_root.parent)),
        "flat_dump": ".",
        "counts": dict(sorted(counts.items())),
        "glb_missing_texture_uris": glb_missing_texture_uris,
        "indexed_paths": len(index),
        "paths": {
            "audio_bank": "audio/bank",
            "audio_ogg": "audio/ogg",
            "fonts_fnt": "fonts/fnt",
            "fonts_ttf": "fonts/ttf",
            "images_png": "images/png",
            "images_pvr": "images/pvr",
            "materials": "materials",
            "models_decoded": "models/decoded",
            "models_merged": "models/merged",
            "previews_decoded": "previews/decoded",
            "previews_merged": "previews/merged",
            "reports": "reports",
            "shaders": "shaders",
            "special": "special",
            "sprites_sc_decoded": "sprites/sc/decoded",
            "sprites_sctx_decoded": "sprites/sctx/decoded",
            "textures_png": "textures/png",
        },
    }

    write_json(args.output_root / "reports" / "path_index.json", index)
    write_json(args.output_root / "reports" / "report.json", report)
    write_text_report(args.output_root / "reports" / "report.txt", report)

    print(f"flat dump root: {args.output_root}")
    print(f"indexed paths: {len(index)}")
    print(f"glb uri misses: {glb_missing_texture_uris}")
    for bucket, count in sorted(counts.items()):
        print(f"{bucket}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
