#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
GLB_MAGIC = 0x46546C67


@dataclass
class GlbFile:
    path: Path
    json_obj: dict
    bin_chunk: bytes


def align4(value: int) -> int:
    return (value + 3) & ~3


def padded_json_bytes(obj: dict) -> bytes:
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return data + (b" " * (align4(len(data)) - len(data)))


def padded_bin_bytes(data: bytes) -> bytes:
    return data + (b"\x00" * (align4(len(data)) - len(data)))


def read_glb(path: Path) -> GlbFile:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError(f"GLB too small: {path}")

    magic, version, _length = struct.unpack_from("<III", raw, 0)
    if magic != GLB_MAGIC or version != 2:
        raise ValueError(f"Not a GLB v2 file: {path}")

    json_len, json_type = struct.unpack_from("<II", raw, 12)
    if json_type != JSON_CHUNK:
        raise ValueError(f"Missing JSON chunk: {path}")

    json_start = 20
    json_end = json_start + json_len
    json_obj = json.loads(raw[json_start:json_end])

    if json_end + 8 > len(raw):
        raise ValueError(f"Missing BIN chunk header: {path}")

    bin_len, bin_type = struct.unpack_from("<II", raw, json_end)
    if bin_type != BIN_CHUNK:
        raise ValueError(f"Missing BIN chunk: {path}")

    bin_start = json_end + 8
    bin_end = bin_start + bin_len
    return GlbFile(path=path, json_obj=json_obj, bin_chunk=raw[bin_start:bin_end])


def write_glb(path: Path, json_obj: dict, bin_chunk: bytes) -> None:
    json_bytes = padded_json_bytes(json_obj)
    bin_bytes = padded_bin_bytes(bin_chunk)
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<III", GLB_MAGIC, 2, total_len))
        fh.write(struct.pack("<II", len(json_bytes), JSON_CHUNK))
        fh.write(json_bytes)
        fh.write(struct.pack("<II", len(bin_bytes), BIN_CHUNK))
        fh.write(bin_bytes)


def get_node_names(glb: GlbFile) -> list[str | None]:
    return [node.get("name") for node in glb.json_obj.get("nodes", [])]


def normalize_node_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = name.lower()
    normalized = re.sub(r"\.\d+$", "", normalized)
    normalized = re.sub(r"_+lod\d+\b", "", normalized)
    normalized = re.sub(r"\blod\d+\b", "", normalized)
    normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def node_name_aliases(name: str | None) -> list[str]:
    normalized = normalize_node_name(name)
    if normalized is None:
        return []

    aliases: list[str] = []
    for alias in (normalized, normalized.rsplit("|", 1)[-1]):
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def collect_target_node_indices(glb_json: dict) -> list[int]:
    indices = {
        target["node"]
        for animation in glb_json.get("animations", [])
        for channel in animation.get("channels", [])
        for target in [channel.get("target") or {}]
        if isinstance(target.get("node"), int)
    }
    return sorted(indices)


def build_parent_map(nodes: list[dict]) -> dict[int, int]:
    parents: dict[int, int] = {}
    for parent_index, node in enumerate(nodes):
        for child_index in node.get("children", []):
            parents[int(child_index)] = parent_index
    return parents


def build_rig_alias_index(rig_names: list[str | None]) -> dict[str, list[int]]:
    alias_index: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(rig_names):
        for alias in node_name_aliases(name):
            alias_index[alias].append(index)
    return alias_index


def choose_rig_index(
    candidates: list[int], used: set[int], preferred_after: int | None
) -> int | None:
    remaining = [index for index in candidates if index not in used]
    if not remaining:
        return None
    if preferred_after is not None:
        for index in remaining:
            if index >= preferred_after:
                return index
    return remaining[0]


def map_target_nodes_to_rig(
    rig_names: list[str | None],
    anim_names: list[str | None],
    target_indices: list[int],
) -> dict[int, int]:
    if rig_names == anim_names:
        return {index: index for index in target_indices}

    alias_index = build_rig_alias_index(rig_names)
    mapping: dict[int, int] = {}
    used: set[int] = set()
    preferred_after = 0
    for target_index in target_indices:
        matched = None
        for alias in node_name_aliases(anim_names[target_index]):
            matched = choose_rig_index(
                alias_index.get(alias, []), used, preferred_after
            )
            if matched is not None:
                break
        if matched is None:
            continue
        mapping[target_index] = matched
        used.add(matched)
        preferred_after = matched
    return mapping


def clone_animation_node(node: dict) -> dict:
    cloned = copy.deepcopy(node)
    cloned.pop("mesh", None)
    cloned.pop("skin", None)
    cloned.pop("camera", None)
    cloned["children"] = []
    return cloned


def append_missing_target_nodes(
    merged_json: dict,
    animation_json: dict,
    node_mapping: dict[int, int],
    target_indices: list[int],
) -> None:
    anim_nodes = animation_json.get("nodes", [])
    rig_nodes = merged_json.setdefault("nodes", [])
    parent_map = build_parent_map(anim_nodes)
    required: set[int] = set()

    for target_index in target_indices:
        current = target_index
        while current not in node_mapping and current not in required:
            required.add(current)
            parent = parent_map.get(current)
            if parent is None or parent in node_mapping:
                break
            current = parent

    appended: dict[int, int] = {}
    pending = set(required)
    while pending:
        progressed = False
        for node_index in sorted(list(pending)):
            parent = parent_map.get(node_index)
            if parent is not None and parent not in node_mapping:
                continue
            appended_index = len(rig_nodes)
            rig_nodes.append(clone_animation_node(anim_nodes[node_index]))
            node_mapping[node_index] = appended_index
            appended[node_index] = appended_index
            pending.remove(node_index)
            progressed = True
        if not progressed:
            unresolved = [anim_nodes[index].get("name") for index in sorted(pending)]
            raise ValueError(f"unable to append missing nodes: {unresolved}")

    scenes = merged_json.get("scenes") or []
    scene_index = merged_json.get("scene", 0)
    root_nodes = None
    if scenes and 0 <= scene_index < len(scenes):
        root_nodes = scenes[scene_index].setdefault("nodes", [])

    for node_index, appended_index in appended.items():
        parent = parent_map.get(node_index)
        if parent is None:
            if root_nodes is not None and appended_index not in root_nodes:
                root_nodes.append(appended_index)
            continue
        parent_index = node_mapping[parent]
        children = rig_nodes[parent_index].setdefault("children", [])
        if appended_index not in children:
            children.append(appended_index)


def merge_animation_with_rig(rig: GlbFile, animation: GlbFile) -> dict:
    rig_names = get_node_names(rig)
    anim_names = get_node_names(animation)

    anims = animation.json_obj.get("animations") or []
    if not anims:
        raise ValueError("Animation file has no animations")

    merged_json = copy.deepcopy(rig.json_obj)
    target_indices = collect_target_node_indices(animation.json_obj)
    node_mapping = map_target_nodes_to_rig(rig_names, anim_names, target_indices)
    missing_targets = [index for index in target_indices if index not in node_mapping]
    if missing_targets:
        append_missing_target_nodes(
            merged_json,
            animation.json_obj,
            node_mapping,
            missing_targets,
        )
    unresolved = [index for index in target_indices if index not in node_mapping]
    if unresolved:
        names = [anim_names[index] for index in unresolved[:20]]
        raise ValueError(f"Node layout differs from rig: {names}")

    merged_bin_offset = align4(len(rig.bin_chunk))
    merged_bin = (
        rig.bin_chunk
        + (b"\x00" * (merged_bin_offset - len(rig.bin_chunk)))
        + animation.bin_chunk
    )

    buffer_views = merged_json.setdefault("bufferViews", [])
    accessors = merged_json.setdefault("accessors", [])
    merged_animations = merged_json.setdefault("animations", [])

    buffer_view_offset = len(buffer_views)
    accessor_offset = len(accessors)

    for buffer_view in animation.json_obj.get("bufferViews", []):
        new_view = copy.deepcopy(buffer_view)
        new_view["buffer"] = 0
        new_view["byteOffset"] = merged_bin_offset + buffer_view.get("byteOffset", 0)
        buffer_views.append(new_view)

    for accessor in animation.json_obj.get("accessors", []):
        new_accessor = copy.deepcopy(accessor)
        if "bufferView" in new_accessor:
            new_accessor["bufferView"] += buffer_view_offset
        accessors.append(new_accessor)

    for animation_obj in anims:
        new_animation = copy.deepcopy(animation_obj)
        for sampler in new_animation.get("samplers", []):
            sampler["input"] += accessor_offset
            sampler["output"] += accessor_offset
        for channel in new_animation.get("channels", []):
            target = channel.get("target") or {}
            if "node" in target:
                target["node"] = node_mapping[target["node"]]
        merged_animations.append(new_animation)

    merged_json["buffers"][0]["byteLength"] = len(merged_bin)
    return {"json": merged_json, "bin": merged_bin}


def find_matching_rig(animation_name: str, available_files: set[str]) -> str | None:
    if "_anim_" not in animation_name:
        return None

    direct = animation_name.split("_anim_", 1)[0] + "_rig.glb"
    if direct in available_files:
        return direct

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Supercell animation-only GLBs with matching rig GLBs"
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    files = {path.name: path for path in args.input_dir.glob("*.glb")}
    merged = 0
    skipped_no_rig: list[str] = []
    skipped_no_animation: list[str] = []
    skipped_invalid: list[str] = []

    for name, path in sorted(files.items()):
        if "_anim_" not in name:
            continue

        rig_name = find_matching_rig(name, set(files))
        if rig_name is None:
            skipped_no_rig.append(name)
            continue

        rig = read_glb(files[rig_name])
        animation = read_glb(path)
        if not (animation.json_obj.get("animations") or []):
            skipped_no_animation.append(name)
            continue

        try:
            merged_glb = merge_animation_with_rig(rig, animation)
        except Exception as exc:
            skipped_invalid.append(f"{name}: {exc}")
            continue

        output_name = name.replace("_anim_", "_merged_anim_", 1)
        write_glb(args.output_dir / output_name, merged_glb["json"], merged_glb["bin"])
        merged += 1

    report_lines = [
        f"Merged animation files: {merged}",
        f"Skipped without matching rig: {len(skipped_no_rig)}",
        f"Skipped without animation tracks: {len(skipped_no_animation)}",
        f"Skipped due to merge validation errors: {len(skipped_invalid)}",
        "",
        "No matching rig samples:",
        *skipped_no_rig[:50],
        "",
        "No animation track samples:",
        *skipped_no_animation[:50],
        "",
        "Validation error samples:",
        *skipped_invalid[:50],
    ]

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Merged {merged} animation file(s)")
    print(f"Skipped without matching rig: {len(skipped_no_rig)}")
    print(f"Skipped without animation tracks: {len(skipped_no_animation)}")
    print(f"Skipped due to merge validation errors: {len(skipped_invalid)}")


if __name__ == "__main__":
    main()
