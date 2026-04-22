#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch render GLB previews with Blender"
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--blender", default="blender")
    parser.add_argument(
        "--renderer-script",
        type=Path,
        default=Path("/home/mirsella/dev/apks/render_glb_preview_blender.py"),
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution-x", type=int, default=720)
    parser.add_argument("--resolution-y", type=int, default=720)
    parser.add_argument("--orbit-turns", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=96)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--mesh-only", action="store_true")
    return parser.parse_args()


def iter_glbs(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(root.rglob("*.glb"))
    return sorted(root.glob("*.glb"))


def glb_has_meshes(path: Path) -> bool:
    raw = path.read_bytes()
    if len(raw) < 20:
        return False
    json_len, json_type = struct.unpack_from("<II", raw, 12)
    if json_type != 0x4E4F534A:
        return False
    doc = json.loads(raw[20 : 20 + json_len])
    return bool(doc.get("meshes"))


def main() -> int:
    args = parse_args()
    files = iter_glbs(args.input_dir, args.recursive)
    if args.mesh_only:
        files = [path for path in files if glb_has_meshes(path)]
    if not files:
        print(f"No GLB files found in {args.input_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "render_report.txt"
    failures: list[tuple[str, int, str]] = []
    rendered = 0
    skipped = 0

    for index, src in enumerate(files, start=1):
        rel = src.relative_to(args.input_dir)
        out = args.output_dir / rel.with_suffix(".mp4")
        if args.skip_existing and out.exists():
            skipped += 1
            print(f"[{index}/{len(files)}] skip {rel}", flush=True)
            continue

        print(f"[{index}/{len(files)}] render {rel}", flush=True)
        cmd = [
            args.blender,
            "-b",
            "--python",
            str(args.renderer_script),
            "--",
            str(src),
            str(out),
            "--fps",
            str(args.fps),
            "--resolution-x",
            str(args.resolution_x),
            "--resolution-y",
            str(args.resolution_y),
            "--orbit-turns",
            str(args.orbit_turns),
            "--min-frames",
            str(args.min_frames),
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if result.returncode == 0 and out.exists():
            rendered += 1
            continue

        failures.append((str(rel), result.returncode, result.stdout[-4000:]))
        print(f"[{index}/{len(files)}] failed {rel}", flush=True)

    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Rendered: {rendered}\n")
        fh.write(f"Skipped existing: {skipped}\n")
        fh.write(f"Failures: {len(failures)}\n\n")
        for name, code, output in failures:
            fh.write(f"## {name}\n")
            fh.write(f"Return code: {code}\n")
            fh.write(output.rstrip())
            fh.write("\n\n")

    print(f"Rendered: {rendered}")
    print(f"Skipped existing: {skipped}")
    print(f"Failures: {len(failures)}")
    print(f"Report: {report_path}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
