#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="Render a quick MP4 preview for a GLB")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution-x", type=int, default=960)
    parser.add_argument("--resolution-y", type=int, default=960)
    parser.add_argument("--orbit-turns", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=96)
    return parser.parse_args(argv)


def world_bounds() -> tuple[Vector, Vector]:
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects found after import")

    min_v = Vector((1e9, 1e9, 1e9))
    max_v = Vector((-1e9, -1e9, -1e9))
    for obj in meshes:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            min_v.x = min(min_v.x, point.x)
            min_v.y = min(min_v.y, point.y)
            min_v.z = min(min_v.z, point.z)
            max_v.x = max(max_v.x, point.x)
            max_v.y = max(max_v.y, point.y)
            max_v.z = max(max_v.z, point.z)
    return min_v, max_v


def normalize_import_orientation() -> None:
    root_objects = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    if not root_objects:
        return

    rotation = Matrix.Rotation(math.radians(90.0), 4, "X")
    for obj in root_objects:
        obj.matrix_world = rotation @ obj.matrix_world

    bpy.context.view_layer.update()


def setup_camera(
    min_v: Vector, max_v: Vector, frame_start: int, frame_end: int, orbit_turns: float
) -> None:
    center = (min_v + max_v) * 0.5
    extent = max_v - min_v
    size = max(extent.x, extent.y, extent.z)
    xy_radius = max(extent.x, extent.y) * 0.5
    sphere_radius = max((max_v - center).length, 1.0)

    camera_data = bpy.data.cameras.new("PreviewCamera")
    camera_data.lens_unit = "FOV"
    camera_data.angle = math.radians(40)
    camera = bpy.data.objects.new("PreviewCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)

    target = center + Vector((0.0, 0.0, extent.z * 0.08))

    distance = max(
        sphere_radius / math.sin(camera_data.angle / 2),
        (xy_radius * 1.8) + sphere_radius,
    )
    height = max(size * 0.22, extent.z * 0.14, 0.35)

    base_x = math.radians(8)
    base_z = math.radians(90)
    span = max(frame_end - frame_start, 1)
    for frame in range(frame_start, frame_end + 1):
        t = (frame - frame_start) / span
        angle = base_z + (math.tau * orbit_turns * t)
        horizontal = Vector(
            (math.cos(angle) * distance, math.sin(angle) * distance, 0.0)
        )
        vertical = Vector((0.0, 0.0, height))
        offset = Matrix.Rotation(base_x, 4, "X") @ (horizontal + vertical)
        camera.location = target + offset
        direction = target - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        camera.rotation_euler.rotate_axis("Z", math.pi)
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)

    bpy.context.scene.camera = camera


def setup_lights(min_v: Vector, max_v: Vector) -> None:
    center = (min_v + max_v) * 0.5
    size = max(max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z)

    sun_data = bpy.data.lights.new(name="PreviewSun", type="SUN")
    sun_data.energy = 2.0
    sun = bpy.data.objects.new(name="PreviewSun", object_data=sun_data)
    sun.location = center + Vector((size * 3, -size * 2, size * 4))
    sun.rotation_euler = (math.radians(35), 0.0, math.radians(35))
    bpy.context.scene.collection.objects.link(sun)


def setup_scene(args: argparse.Namespace) -> tuple[int, int]:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.fps = args.fps
    scene.render.image_settings.file_format = "PNG"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("PreviewWorld")
    scene.world.color = (0.05, 0.05, 0.06)

    if bpy.data.actions:
        start = min(action.frame_range[0] for action in bpy.data.actions)
        end = max(action.frame_range[1] for action in bpy.data.actions)
        scene.frame_start = int(math.floor(start))
        scene.frame_end = max(
            int(math.ceil(end)), scene.frame_start + args.min_frames - 1
        )
    else:
        scene.frame_start = 1
        scene.frame_end = args.min_frames

    return scene.frame_start, scene.frame_end


def encode_video(frames_dir: Path, output: Path, fps: int) -> None:
    frame_pattern = frames_dir / "frame_%04d.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_pattern),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.input))
    normalize_import_orientation()

    min_v, max_v = world_bounds()
    frame_start, frame_end = setup_scene(args)
    setup_camera(min_v, max_v, frame_start, frame_end, args.orbit_turns)
    setup_lights(min_v, max_v)
    frames_dir = Path(tempfile.mkdtemp(prefix="glb_preview_"))
    try:
        bpy.context.scene.render.filepath = str(frames_dir / "frame_")
        bpy.ops.render.render(animation=True)
        encode_video(frames_dir, args.output, args.fps)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
