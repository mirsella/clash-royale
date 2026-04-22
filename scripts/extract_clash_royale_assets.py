#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/mirsella/dev/apks")
DEFAULT_PACKAGE_ROOT = ROOT / "extracted" / "com.supercell.clashroyale_phone_arm64"

KTX1_MAGIC = b"\xabKTX 11\xbb\r\n\x1a\n"
KTX2_MAGIC = b"\xabKTX 20\xbb\r\n\x1a\n"
ASTC_MAGIC = b"\x13\xab\xa1\x5c"

KTX1_EXT_TO_MODE = {
    36196: ("decode_etc1", "RGB"),
    37492: ("decode_etc2", "RGB"),
    37496: ("decode_etc2a8", "RGBA"),
}

VK_ASTC_FORMATS = {
    157: ("4x4", False),
    158: ("4x4", True),
    159: ("5x4", False),
    160: ("5x4", True),
    161: ("5x5", False),
    162: ("5x5", True),
    163: ("6x5", False),
    164: ("6x5", True),
    165: ("6x6", False),
    166: ("6x6", True),
    167: ("8x5", False),
    168: ("8x5", True),
    169: ("8x6", False),
    170: ("8x6", True),
    171: ("8x8", False),
    172: ("8x8", True),
    173: ("10x5", False),
    174: ("10x5", True),
    175: ("10x6", False),
    176: ("10x6", True),
    177: ("10x8", False),
    178: ("10x8", True),
    179: ("10x10", False),
    180: ("10x10", True),
    181: ("12x10", False),
    182: ("12x10", True),
    183: ("12x12", False),
    184: ("12x12", True),
}

ASSET_BUCKETS = {
    ".ogg": "audio/ogg",
    ".bank": "audio/bank",
    ".png": "images/png",
    ".pvr": "images/pvr",
    ".ttf": "fonts/ttf",
    ".fnt": "fonts/fnt",
    ".shader": "shaders/shader",
    ".fsh": "shaders/fsh",
    ".vsh": "shaders/vsh",
    ".rmat": "materials/rmat",
    ".scw": "special/scw",
}

BASE_TEXTURE_KEYS = ("diffuseTex", "mainTex", "baseTex", "albedoTex")
NORMAL_TEXTURE_KEYS = ("normalTex",)
MRA_TEXTURE_KEYS = ("mraTex", "maskTex")
EMISSIVE_TEXTURE_KEYS = ("emissionTex",)
MODEL_DIR_NAMES = {"model", "models"}
ANIMATION_DIR_NAMES = {"animation", "animations"}
PUBLIC_EXCLUDED_CONTENT = [
    ".assetdbhash",
    "*.csv",
    "*.ini",
    "*.json",
    "*.number",
    "*.scdb",
    "*.toml",
    "data/",
    "misc/",
]


def add_site_packages(venv_root: Path) -> None:
    for path in sorted(venv_root.glob("lib/python*/site-packages")):
        sys.path.insert(0, str(path))


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


def write_json(path: Path, data: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def limit_sources(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None or limit < 0:
        return paths
    return paths[:limit]


def strip_sc3d_prefix(path_str: str) -> Path:
    path = Path(path_str)
    if path.parts and path.parts[0] == "sc3d":
        return Path(*path.parts[1:])
    return path


def texture_png_name(texture_rel: Path) -> Path:
    return texture_rel.with_name(texture_rel.name + "2.png")


def first_part_index(parts: tuple[str, ...], names: set[str]) -> int | None:
    for index, part in enumerate(parts):
        if part in names:
            return index
    return None


def build_astc_header(width: int, height: int, block: str) -> bytes:
    block_w, block_h = (int(part) for part in block.split("x", 1))
    header = bytearray(ASTC_MAGIC)
    header.extend(bytes([block_w, block_h, 1]))
    for value in (width, height, 1):
        header.extend(bytes((value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF)))
    return bytes(header)


def parse_ktx2_level0(data: bytes) -> dict:
    if len(data) < 80 or data[:12] != KTX2_MAGIC:
        raise ValueError("missing KTX2 header")

    (
        _magic,
        vk_format,
        type_size,
        pixel_width,
        pixel_height,
        pixel_depth,
        layer_count,
        face_count,
        level_count,
        supercompression_scheme,
        _dfd_offset,
        _dfd_length,
        _kvd_offset,
        _kvd_length,
        _sgd_offset,
        _sgd_length,
    ) = struct.unpack_from("<12s9I4I2Q", data, 0)

    if pixel_depth not in (0, 1):
        raise ValueError(f"unsupported pixelDepth={pixel_depth}")
    if layer_count not in (0, 1):
        raise ValueError(f"unsupported layerCount={layer_count}")
    if face_count != 1:
        raise ValueError(f"unsupported faceCount={face_count}")
    if level_count < 1:
        raise ValueError("ktx2 has no mip levels")
    if type_size != 1:
        raise ValueError(f"unexpected typeSize={type_size}")
    if supercompression_scheme != 0:
        raise ValueError(
            f"unsupported supercompressionScheme={supercompression_scheme}"
        )
    if vk_format not in VK_ASTC_FORMATS:
        raise ValueError(f"unsupported vkFormat={vk_format}")

    level_index_off = 80
    if len(data) < level_index_off + (level_count * 24):
        raise ValueError("truncated KTX2 level index")

    byte_offset, byte_length, uncompressed_length = struct.unpack_from(
        "<3Q", data, level_index_off
    )
    byte_end = byte_offset + byte_length
    if byte_end > len(data):
        raise ValueError("truncated KTX2 level payload")

    block, is_srgb = VK_ASTC_FORMATS[vk_format]
    payload = data[byte_offset:byte_end]
    if uncompressed_length != byte_length:
        raise ValueError(
            f"unexpected compressed level0 payload {byte_length} != {uncompressed_length}"
        )

    return {
        "kind": "ktx2",
        "vk_format": vk_format,
        "block": block,
        "is_srgb": is_srgb,
        "width": pixel_width,
        "height": pixel_height,
        "level_count": level_count,
        "payload": payload,
        "payload_length": byte_length,
    }


def parse_ktx1_level0(data: bytes) -> dict:
    if len(data) < 64 or data[:12] != KTX1_MAGIC:
        raise ValueError("missing KTX1 header")

    (
        endianness,
        gl_type,
        _gl_type_size,
        gl_format,
        gl_internal,
        _gl_base,
        width,
        height,
        _depth,
        _array_elements,
        _faces,
        _mip_levels,
        kv_size,
    ) = struct.unpack_from("<13I", data, 12)

    if endianness != 0x04030201:
        raise ValueError("unsupported KTX1 endianness")
    if gl_type != 0 or gl_format != 0:
        raise ValueError("expected compressed KTX1 texture")

    decoder_info = KTX1_EXT_TO_MODE.get(gl_internal)
    if decoder_info is None:
        raise ValueError(f"unsupported KTX1 glInternalformat={gl_internal}")

    offset = 64 + kv_size
    if len(data) < offset + 4:
        raise ValueError("truncated KTX1 image size")
    image_size = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    payload_end = offset + image_size
    if payload_end > len(data):
        raise ValueError("truncated KTX1 image payload")

    decoder_name, mode = decoder_info
    return {
        "kind": "ktx1",
        "gl_internal": gl_internal,
        "decoder_name": decoder_name,
        "mode": mode,
        "width": width,
        "height": height,
        "payload": data[offset:payload_end],
    }


def parse_ktx_level0(data: bytes) -> dict:
    if data[:12] == KTX2_MAGIC:
        return parse_ktx2_level0(data)
    if data[:12] == KTX1_MAGIC:
        return parse_ktx1_level0(data)
    raise ValueError("missing KTX header")


def run_astc_decode(astcenc_path: Path, astc_bytes: bytes, out_png: Path) -> None:
    ensure_parent(out_png)
    with tempfile.TemporaryDirectory(prefix="cr_astc_") as tmpdir:
        astc_path = Path(tmpdir) / "input.astc"
        astc_path.write_bytes(astc_bytes)
        subprocess.run(
            [str(astcenc_path), "-dl", str(astc_path), str(out_png)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


def texture_ref_to_png_path(
    texture_ref: str, texture_png_root: Path
) -> tuple[Path, str]:
    clean = texture_ref.split("#", 1)[0]
    rel = strip_sc3d_prefix(clean)
    return texture_png_root / texture_png_name(rel), clean


def first_texture_value(textures: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = textures.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def ensure_image_texture(doc: dict, image_uri: str) -> int:
    images = doc.setdefault("images", [])
    textures = doc.setdefault("textures", [])
    cache = doc.setdefault("_oc_image_cache", {})
    existing = cache.get(image_uri)
    if existing is not None:
        return int(existing)

    image_index = len(images)
    texture_index = len(textures)
    images.append({"uri": image_uri, "name": Path(image_uri).name})
    textures.append({"source": image_index, "name": Path(image_uri).stem})
    cache[image_uri] = texture_index
    return texture_index


def sanitize_glb_doc(doc: dict) -> None:
    for key in ("extensionsUsed", "extensionsRequired"):
        values = [value for value in doc.get(key, []) if value != "SC_shader"]
        if values:
            doc[key] = values
        elif key in doc:
            doc.pop(key)
    doc.pop("_oc_image_cache", None)


def inspect_material_bindings(doc: dict, dst: Path, texture_png_root: Path) -> dict:
    stats = {
        "materials": 0,
        "materials_with_base_color": 0,
        "materials_with_normal": 0,
        "materials_with_mra": 0,
        "missing_texture_refs": [],
    }

    for material in doc.get("materials", []):
        stats["materials"] += 1
        pbr = material.get("pbrMetallicRoughness") or {}
        if "baseColorTexture" in pbr:
            stats["materials_with_base_color"] += 1
        if "metallicRoughnessTexture" in pbr:
            stats["materials_with_mra"] += 1
        if "normalTexture" in material:
            stats["materials_with_normal"] += 1

        textures = (
            material.get("extras", {})
            .get("supercell", {})
            .get("variables", {})
            .get("textures", {})
        ) or {}
        for ref in (
            first_texture_value(textures, BASE_TEXTURE_KEYS),
            first_texture_value(textures, NORMAL_TEXTURE_KEYS),
            first_texture_value(textures, MRA_TEXTURE_KEYS),
            first_texture_value(textures, EMISSIVE_TEXTURE_KEYS),
        ):
            if ref is None:
                continue
            png_path, clean_ref = texture_ref_to_png_path(ref, texture_png_root)
            if not png_path.is_file():
                stats["missing_texture_refs"].append(clean_ref)

    return stats


def rewrite_materials(doc: dict, dst: Path, texture_png_root: Path) -> dict:
    stats = {
        "materials": 0,
        "materials_with_base_color": 0,
        "materials_with_normal": 0,
        "materials_with_mra": 0,
        "missing_texture_refs": [],
    }

    for material in doc.get("materials", []):
        stats["materials"] += 1
        original = dict(material)
        variables = original.get("variables") or {}
        textures = variables.get("textures") or {}
        floats = variables.get("floats") or {}

        new_material = {
            "name": original.get("name", "material"),
            "extras": {"supercell": original},
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": float(floats.get("metalness", 0.0)),
                "roughnessFactor": float(floats.get("roughness", 1.0)),
            },
        }

        for ref, bucket, stat_key in (
            (
                first_texture_value(textures, BASE_TEXTURE_KEYS),
                "baseColorTexture",
                "materials_with_base_color",
            ),
            (
                first_texture_value(textures, NORMAL_TEXTURE_KEYS),
                "normalTexture",
                "materials_with_normal",
            ),
            (
                first_texture_value(textures, MRA_TEXTURE_KEYS),
                "metallicRoughnessTexture",
                "materials_with_mra",
            ),
            (
                first_texture_value(textures, EMISSIVE_TEXTURE_KEYS),
                "emissiveTexture",
                None,
            ),
        ):
            if ref is None:
                continue
            png_path, clean_ref = texture_ref_to_png_path(ref, texture_png_root)
            if not png_path.is_file():
                stats["missing_texture_refs"].append(clean_ref)
                continue

            image_uri = os.path.relpath(png_path, start=dst.parent)
            texture_index = ensure_image_texture(doc, image_uri)
            if bucket == "baseColorTexture":
                new_material["pbrMetallicRoughness"][bucket] = {"index": texture_index}
            elif bucket == "metallicRoughnessTexture":
                new_material["pbrMetallicRoughness"][bucket] = {"index": texture_index}
            else:
                new_material[bucket] = {"index": texture_index}
                if bucket == "emissiveTexture":
                    new_material["emissiveFactor"] = [1.0, 1.0, 1.0]
            if stat_key is not None:
                stats[stat_key] += 1

        material.clear()
        material.update(new_material)

    sanitize_glb_doc(doc)
    return stats


def read_glb_json(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError(f"GLB too small: {path}")
    json_length, json_type = struct.unpack_from("<II", raw, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"Missing JSON chunk: {path}")
    data = json.loads(raw[20 : 20 + json_length])
    if not isinstance(data, dict):
        raise ValueError(f"GLB JSON is not an object: {path}")
    return data


def load_runtime(converter_dir: Path):
    add_site_packages(Path.home() / "dev" / "tools" / "venvs" / "sc-flat-converter")
    add_site_packages(Path.home() / "dev" / "tools" / "venvs" / "apk-assets")
    sys.path.insert(0, str(converter_dir))
    sys.path.insert(0, str(ROOT))

    from lib.glTF import glTF  # type: ignore
    from lib.odin import SupercellOdinGLTF  # type: ignore
    from PIL import Image  # type: ignore
    from mb_sc_tools import decode_file as decode_supercell_file  # type: ignore
    from mb_sc_tools.astc import ensure_astcenc  # type: ignore
    import texture2ddecoder  # type: ignore

    from merge_supercell_animation_glbs import (  # type: ignore
        merge_animation_with_rig,
        read_glb,
        write_glb,
    )
    from render_modern_sc2_exports import render_exports as render_modern_sc_exports

    return {
        "glTF": glTF,
        "SupercellOdinGLTF": SupercellOdinGLTF,
        "Image": Image,
        "decode_supercell_file": decode_supercell_file,
        "ensure_astcenc": ensure_astcenc,
        "merge_animation_with_rig": merge_animation_with_rig,
        "read_glb": read_glb,
        "render_modern_sc_exports": render_modern_sc_exports,
        "texture2ddecoder": texture2ddecoder,
        "write_glb": write_glb,
    }


def decode_sc_batch(
    assets_root: Path,
    organized_root: Path,
    decode_supercell_file,
    render_modern_sc_exports,
    failures: list[dict],
    limit: int | None,
) -> dict:
    sources = limit_sources(sorted((assets_root / "sc").rglob("*.sc")), limit)
    result = {
        "found": len(sources),
        "decoded": 0,
        "failed": 0,
        "modern_parsed": 0,
        "modern_exports_rendered": 0,
        "modern_frames_rendered": 0,
    }
    originals_root = organized_root / "supercell_sc" / "original"
    decoded_root = organized_root / "supercell_sc" / "decoded"
    sctx_decoded_root = organized_root / "supercell_sctx" / "decoded"

    def accumulate_render_stats(stats: dict) -> None:
        result["modern_parsed"] += 1
        result["modern_exports_rendered"] += int(stats.get("exports_rendered", 0))
        result["modern_frames_rendered"] += int(stats.get("frames_rendered", 0))

    for src in sources:
        rel = src.relative_to(assets_root / "sc")
        hardlink_or_copy(src, originals_root / rel)
        out_dir = decoded_root / rel.with_suffix("")
        data_json_path = out_dir / "data.json"
        if not data_json_path.is_file():
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                decode_supercell_file(src, out_dir)
            except Exception as exc:  # noqa: BLE001
                result["failed"] += 1
                failures.append({"type": "sc", "path": str(rel), "error": str(exc)})
                continue

        result["decoded"] += 1

        try:
            decoded_info = read_json(data_json_path)
            if not decoded_info.get("container", {}).get("modern"):
                continue
            report_path = out_dir / "sc2_exports_report.json"
            if report_path.is_file():
                existing_stats = read_json(report_path)
                if int(existing_stats.get("exports_rendered", 0)) > 0:
                    accumulate_render_stats(existing_stats)
                    continue
            stats = render_modern_sc_exports(
                raw_sc_path=src,
                raw_sc_root=assets_root / "sc",
                sc_workspace=out_dir,
                sctx_decoded_root=sctx_decoded_root,
                output_root=out_dir / "exports",
                max_frames_per_export=1,
            )
            write_json(report_path, stats)
            accumulate_render_stats(stats)
        except Exception as exc:  # noqa: BLE001
            failures.append({"type": "sc_render", "path": str(rel), "error": str(exc)})
    return result


def decode_sctx_batch(
    assets_root: Path,
    organized_root: Path,
    decode_supercell_file,
    failures: list[dict],
    limit: int | None,
) -> dict:
    sources = limit_sources(sorted((assets_root / "sc").rglob("*.sctx")), limit)
    result = {"found": len(sources), "decoded": 0, "failed": 0}
    originals_root = organized_root / "supercell_sctx" / "original"
    decoded_root = organized_root / "supercell_sctx" / "decoded"
    for src in sources:
        rel = src.relative_to(assets_root / "sc")
        hardlink_or_copy(src, originals_root / rel)
        out_dir = decoded_root / rel.with_suffix("")
        if (out_dir / "data.json").is_file():
            result["decoded"] += 1
            continue
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            decode_supercell_file(src, out_dir)
            result["decoded"] += 1
        except Exception as exc:  # noqa: BLE001
            result["failed"] += 1
            failures.append({"type": "sctx", "path": str(rel), "error": str(exc)})
    return result


def decode_ktx_batch(
    assets_root: Path,
    organized_root: Path,
    astcenc_path: Path,
    pil_image,
    texture2ddecoder,
    failures: list[dict],
    limit: int | None,
) -> dict:
    sc3d_root = assets_root / "sc3d"
    sources = limit_sources(sorted(sc3d_root.rglob("*.ktx")), limit)
    originals_root = organized_root / "sc3d" / "textures" / "original"
    png_root = organized_root / "sc3d" / "textures" / "png"
    result = {
        "found": len(sources),
        "decoded": 0,
        "failed": 0,
        "kinds": Counter(),
        "ktx1_internal_formats": Counter(),
        "vk_formats": Counter(),
        "blocks": Counter(),
    }
    for src in sources:
        rel = src.relative_to(sc3d_root)
        hardlink_or_copy(src, originals_root / rel)
        dst = png_root / texture_png_name(rel)
        data = src.read_bytes()
        if dst.is_file():
            try:
                parsed = parse_ktx_level0(data)
                result["kinds"][parsed["kind"]] += 1
                if parsed["kind"] == "ktx2":
                    result["vk_formats"][str(parsed["vk_format"])] += 1
                    result["blocks"][parsed["block"]] += 1
                else:
                    result["ktx1_internal_formats"][str(parsed["gl_internal"])] += 1
            except Exception:
                pass
            result["decoded"] += 1
            continue
        try:
            parsed = parse_ktx_level0(data)
            result["kinds"][parsed["kind"]] += 1
            if parsed["kind"] == "ktx2":
                result["vk_formats"][str(parsed["vk_format"])] += 1
                result["blocks"][parsed["block"]] += 1
                astc_bytes = (
                    build_astc_header(
                        parsed["width"], parsed["height"], parsed["block"]
                    )
                    + parsed["payload"]
                )
                run_astc_decode(astcenc_path, astc_bytes, dst)
            else:
                result["ktx1_internal_formats"][str(parsed["gl_internal"])] += 1
                decoder = getattr(texture2ddecoder, parsed["decoder_name"])
                pixels = decoder(parsed["payload"], parsed["width"], parsed["height"])
                ensure_parent(dst)
                pil_image.frombytes(
                    parsed["mode"], (parsed["width"], parsed["height"]), pixels
                ).save(dst)
            result["decoded"] += 1
        except Exception as exc:  # noqa: BLE001
            result["failed"] += 1
            failures.append({"type": "ktx", "path": str(rel), "error": str(exc)})
    result["kinds"] = dict(result["kinds"])
    result["ktx1_internal_formats"] = dict(result["ktx1_internal_formats"])
    result["vk_formats"] = dict(result["vk_formats"])
    result["blocks"] = dict(result["blocks"])
    return result


def decode_glb_batch(
    assets_root: Path,
    organized_root: Path,
    gltf_cls,
    odin_cls,
    failures: list[dict],
    limit: int | None,
) -> dict:
    sc3d_root = assets_root / "sc3d"
    sources = limit_sources(sorted(sc3d_root.rglob("*.glb")), limit)
    originals_root = organized_root / "sc3d" / "models" / "original"
    decoded_root = organized_root / "sc3d" / "models" / "decoded"
    texture_png_root = organized_root / "sc3d" / "textures" / "png"
    result = {
        "found": len(sources),
        "decoded": 0,
        "failed": 0,
        "with_animations": 0,
        "with_materials": 0,
        "with_base_color": 0,
        "missing_texture_refs": Counter(),
    }
    for src in sources:
        rel = src.relative_to(sc3d_root)
        hardlink_or_copy(src, originals_root / rel)
        dst = decoded_root / rel
        if dst.is_file():
            doc = read_glb_json(dst)
            material_stats = inspect_material_bindings(doc, dst, texture_png_root)
            if doc.get("animations"):
                result["with_animations"] += 1
            if doc.get("materials"):
                result["with_materials"] += 1
            if material_stats["materials_with_base_color"]:
                result["with_base_color"] += 1
            for missing in material_stats["missing_texture_refs"]:
                result["missing_texture_refs"][missing] += 1
            result["decoded"] += 1
            continue
        try:
            gltf = gltf_cls()
            gltf.read(src.read_bytes())
            for chunk in gltf.chunks:
                chunk.deserialize_json()
            gltf = odin_cls(gltf).process()
            json_chunk = gltf.get_chunk("JSON")
            if not isinstance(json_chunk.data, dict):
                raise ValueError("decoded JSON chunk is not a dict")
            material_stats = rewrite_materials(json_chunk.data, dst, texture_png_root)
            if json_chunk.data.get("animations"):
                result["with_animations"] += 1
            if json_chunk.data.get("materials"):
                result["with_materials"] += 1
            if material_stats["materials_with_base_color"]:
                result["with_base_color"] += 1
            for missing in material_stats["missing_texture_refs"]:
                result["missing_texture_refs"][missing] += 1
            ensure_parent(dst)
            dst.write_bytes(gltf.write())
            result["decoded"] += 1
        except Exception as exc:  # noqa: BLE001
            result["failed"] += 1
            failures.append({"type": "glb", "path": str(rel), "error": str(exc)})
    result["missing_texture_refs"] = dict(
        result["missing_texture_refs"].most_common(50)
    )
    return result


def organize_misc_assets(assets_root: Path, organized_root: Path) -> dict:
    counts = Counter()
    skipped_suffixes = {".glb", ".ktx", ".sc", ".sctx"}
    for src in assets_root.rglob("*"):
        if not src.is_file():
            continue
        suffix = src.suffix.lower()
        if suffix in skipped_suffixes:
            continue
        bucket = ASSET_BUCKETS.get(suffix)
        if bucket is None:
            continue
        rel = src.relative_to(assets_root)
        hardlink_or_copy(src, organized_root / bucket / rel)
        counts[bucket] += 1
    return dict(counts)


def summarize_previews(organized_root: Path) -> dict:
    previews_root = organized_root / "previews"
    decoded_root = previews_root / "decoded"
    merged_root = previews_root / "merged"
    decoded = (
        sum(1 for _ in decoded_root.rglob("*.mp4")) if decoded_root.is_dir() else 0
    )
    merged = sum(1 for _ in merged_root.rglob("*.mp4")) if merged_root.is_dir() else 0
    return {
        "decoded": decoded,
        "merged": merged,
        "total": decoded + merged,
    }


def group_candidate_rigs(decoded_root: Path) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = defaultdict(list)
    for rig in decoded_root.rglob("*.glb"):
        rel = rig.relative_to(decoded_root)
        if first_part_index(rel.parts, ANIMATION_DIR_NAMES) is not None:
            continue
        parts = rel.parts
        idx = first_part_index(parts, MODEL_DIR_NAMES)
        if idx is not None:
            groups[Path(*parts[:idx])].append(rig)
    for value in groups.values():
        value.sort(
            key=lambda path: (
                0 if path.name.endswith("_rig.glb") else 1,
                0 if "lod1" not in path.name else 1,
                path.name,
            )
        )
    return groups


def merge_animation_batch(
    organized_root: Path,
    read_glb,
    write_glb,
    merge_animation_with_rig,
    failures: list[dict],
) -> dict:
    decoded_root = organized_root / "sc3d" / "models" / "decoded"
    merged_root = organized_root / "sc3d" / "models" / "merged"
    candidate_files = sorted(
        path
        for path in decoded_root.rglob("*.glb")
        if first_part_index(path.parts, ANIMATION_DIR_NAMES) is not None
    )
    groups = group_candidate_rigs(decoded_root)
    result = {
        "animations": 0,
        "merged": 0,
        "no_rig": 0,
        "invalid": 0,
        "skipped_no_animation": 0,
    }
    for animation_path in candidate_files:
        rel = animation_path.relative_to(decoded_root)
        animation = read_glb(animation_path)
        if not (animation.json_obj.get("animations") or []):
            result["skipped_no_animation"] += 1
            continue

        result["animations"] += 1
        parts = rel.parts
        idx = first_part_index(parts, ANIMATION_DIR_NAMES)
        if idx is None:
            result["invalid"] += 1
            failures.append(
                {
                    "type": "merge",
                    "path": str(rel),
                    "error": "missing animation directory marker",
                }
            )
            continue
        family = Path(*parts[:idx])
        candidates = groups.get(family, [])
        if not candidates:
            result["no_rig"] += 1
            continue
        matched = False
        for rig_path in candidates:
            try:
                rig = read_glb(rig_path)
                merged = merge_animation_with_rig(rig, animation)
            except Exception:
                continue
            output_name = animation_path.stem + ".merged.glb"
            out_path = merged_root / rel.parent / output_name
            write_glb(out_path, merged["json"], merged["bin"])
            result["merged"] += 1
            matched = True
            break
        if not matched:
            result["invalid"] += 1
            failures.append(
                {
                    "type": "merge",
                    "path": str(rel),
                    "error": "no compatible rig in sibling model directory",
                }
            )
    return result


def write_text_report(path: Path, report: dict) -> None:
    ensure_parent(path)
    lines = [
        "Clash Royale public asset dump report",
        "====================================",
        "",
        f"Package: {report['package']}",
        f"Dump root: {report['organized_root']}",
        "",
        "Excluded non-asset/config content",
        "-------------------------------",
        *report["excluded_non_asset_content"],
        "",
        "Counts",
        "------",
    ]
    for key in ("sc", "sctx", "ktx", "glb", "merge", "previews"):
        section = report["sections"].get(key, {})
        lines.append(f"{key}: {json.dumps(section, sort_keys=True)}")
    lines.extend(["", "Misc buckets", "------------"])
    for bucket, count in sorted(report["misc"].items()):
        lines.append(f"{bucket}: {count}")
    lines.extend(["", "Paths", "-----"])
    for label, rel in sorted(report["paths"].items()):
        lines.append(f"{label}: {rel}")
    lines.extend(["", "Failures", "--------"])
    for item in report["failures"][:200]:
        lines.append(f"{item['type']}: {item['path']} :: {item['error']}")
    if len(report["failures"]) > 200:
        lines.append(
            f"... truncated {len(report['failures']) - 200} additional failures"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch decode and organize Clash Royale assets"
    )
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument(
        "--converter-dir",
        type=Path,
        default=Path.home() / "dev" / "Supercell-Flat-Converter",
    )
    parser.add_argument("--limit-sc", type=int)
    parser.add_argument("--limit-sctx", type=int)
    parser.add_argument("--limit-ktx", type=int)
    parser.add_argument("--limit-glb", type=int)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets_root = (
        args.package_root / "raw" / "splits" / "install_time_asset_pack" / "assets"
    )
    if not assets_root.is_dir():
        raise SystemExit(f"missing assets root: {assets_root}")

    organized_root = args.package_root / "organized"
    if args.clean and organized_root.exists():
        shutil.rmtree(organized_root)
    organized_root.mkdir(parents=True, exist_ok=True)

    runtime = load_runtime(args.converter_dir)
    astcenc_path = runtime["ensure_astcenc"]()

    failures: list[dict] = []
    misc_counts = organize_misc_assets(assets_root, organized_root)
    sc_result = decode_sc_batch(
        assets_root,
        organized_root,
        runtime["decode_supercell_file"],
        runtime["render_modern_sc_exports"],
        failures,
        args.limit_sc,
    )
    sctx_result = decode_sctx_batch(
        assets_root,
        organized_root,
        runtime["decode_supercell_file"],
        failures,
        args.limit_sctx,
    )
    ktx_result = decode_ktx_batch(
        assets_root,
        organized_root,
        astcenc_path,
        runtime["Image"],
        runtime["texture2ddecoder"],
        failures,
        args.limit_ktx,
    )
    glb_result = decode_glb_batch(
        assets_root,
        organized_root,
        runtime["glTF"],
        runtime["SupercellOdinGLTF"],
        failures,
        args.limit_glb,
    )
    merge_result = merge_animation_batch(
        organized_root,
        runtime["read_glb"],
        runtime["write_glb"],
        runtime["merge_animation_with_rig"],
        failures,
    )
    preview_result = summarize_previews(organized_root)

    report = {
        "package": args.package_root.name,
        "organized_root": ".",
        "excluded_non_asset_content": PUBLIC_EXCLUDED_CONTENT,
        "sections": {
            "sc": sc_result,
            "sctx": sctx_result,
            "ktx": ktx_result,
            "glb": glb_result,
            "merge": merge_result,
            "previews": preview_result,
        },
        "misc": misc_counts,
        "failures": failures,
        "paths": {
            "audio": "audio",
            "fonts": "fonts",
            "images": "images",
            "materials": "materials",
            "previews": "previews",
            "sc_decoded": "supercell_sc/decoded",
            "sctx_decoded": "supercell_sctx/decoded",
            "sc3d_models_decoded": "sc3d/models/decoded",
            "sc3d_models_merged": "sc3d/models/merged",
            "sc3d_textures_png": "sc3d/textures/png",
            "shaders": "shaders",
            "special": "special",
        },
    }

    write_json(organized_root / "reports" / "report.json", report)
    write_text_report(organized_root / "reports" / "report.txt", report)

    print(f"organized root: {organized_root}")
    print(f"sc decoded: {sc_result['decoded']} / {sc_result['found']}")
    print(f"sctx decoded: {sctx_result['decoded']} / {sctx_result['found']}")
    print(f"ktx decoded: {ktx_result['decoded']} / {ktx_result['found']}")
    print(f"glb decoded: {glb_result['decoded']} / {glb_result['found']}")
    print(f"merged animations: {merge_result['merged']} / {merge_result['animations']}")
    print(f"preview mp4s: {preview_result['total']}")
    print(f"failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
