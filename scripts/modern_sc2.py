from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
APK_ASSETS_SITE = (
    ROOT.parent
    / "tools"
    / "venvs"
    / "apk-assets"
    / "lib"
    / "python3.14"
    / "site-packages"
)
GENERATED_SC2_ROOT = ROOT / "generated_sc2"


class Sc2ParseError(RuntimeError):
    pass


@dataclass
class Sc2Matrix:
    a: float
    b: float
    c: float
    d: float
    tx: float
    ty: float


@dataclass
class Sc2ColorTransform:
    r_mul: int
    g_mul: int
    b_mul: int
    alpha: int
    r_add: int
    g_add: int
    b_add: int


@dataclass
class Sc2TextureData:
    texture_format: int
    pixel_type: int
    width: int
    height: int
    inline_length: int
    inline_data: bytes | None
    external_texture: str | None


@dataclass
class Sc2TextureSet:
    index: int
    lowres: Sc2TextureData | None
    highres: Sc2TextureData | None


@dataclass
class Sc2ShapeVertex:
    x: float
    y: float
    u: float
    v: float


@dataclass
class Sc2ShapeCommand:
    texture_index: int
    points_count: int
    points_offset: int
    vertices: list[Sc2ShapeVertex]


@dataclass
class Sc2Shape:
    id: int
    export_name: str | None
    commands: list[Sc2ShapeCommand]


@dataclass
class Sc2MovieChild:
    object_id: int
    name: str | None
    blend_mode: int | None


@dataclass
class Sc2FrameElement:
    instance_index: int
    matrix_index: int
    color_transform_index: int


@dataclass
class Sc2Frame:
    index: int
    elements_count: int
    label: str | None
    elements: list[Sc2FrameElement]


@dataclass
class Sc2MovieClip:
    id: int
    export_name: str | None
    framerate: int
    matrix_bank_index: int
    frames_count: int
    children: list[Sc2MovieChild]
    frames: list[Sc2Frame]


@dataclass
class Sc2ParsedFile:
    source: str
    container: dict[str, Any]
    descriptor: dict[str, Any]
    chunks: dict[str, Any]
    strings: list[str]
    matrix_banks: list[dict[str, Any]]
    exports: list[dict[str, Any]]
    shapes: list[Sc2Shape]
    movieclips: list[Sc2MovieClip]
    textures: list[Sc2TextureSet]

    def summary(self) -> dict[str, Any]:
        exported_shape_count = sum(1 for shape in self.shapes if shape.export_name)
        exported_movieclip_count = sum(
            1 for movieclip in self.movieclips if movieclip.export_name
        )
        external_textures = []
        inline_textures = 0
        for texture_set in self.textures:
            for label in ("lowres", "highres"):
                texture = getattr(texture_set, label)
                if texture is None:
                    continue
                if texture.external_texture:
                    external_textures.append(
                        {
                            "index": texture_set.index,
                            "slot": label,
                            "path": texture.external_texture,
                            "width": texture.width,
                            "height": texture.height,
                        }
                    )
                elif texture.inline_length:
                    inline_textures += 1

        shape_summaries = []
        for shape in self.shapes:
            texture_indices = sorted(
                {command.texture_index for command in shape.commands}
            )
            vertex_count = sum(len(command.vertices) for command in shape.commands)
            shape_summaries.append(
                {
                    "id": shape.id,
                    "export_name": shape.export_name,
                    "commands": len(shape.commands),
                    "vertices": vertex_count,
                    "textures": texture_indices,
                }
            )

        movieclip_summaries = []
        for movieclip in self.movieclips:
            movieclip_summaries.append(
                {
                    "id": movieclip.id,
                    "export_name": movieclip.export_name,
                    "framerate": movieclip.framerate,
                    "matrix_bank_index": movieclip.matrix_bank_index,
                    "children": len(movieclip.children),
                    "frames": len(movieclip.frames),
                    "frame_elements_total": sum(
                        frame.elements_count for frame in movieclip.frames
                    ),
                    "labels": [
                        frame.label for frame in movieclip.frames if frame.label
                    ],
                }
            )

        return {
            "source": self.source,
            "container": self.container,
            "descriptor": self.descriptor,
            "chunks": self.chunks,
            "counts": {
                "strings": len(self.strings),
                "matrix_banks": len(self.matrix_banks),
                "exports": len(self.exports),
                "shapes": len(self.shapes),
                "movieclips": len(self.movieclips),
                "textures": len(self.textures),
                "exported_shapes": exported_shape_count,
                "exported_movieclips": exported_movieclip_count,
            },
            "external_textures": external_textures,
            "inline_texture_slots": inline_textures,
            "exports": self.exports,
            "shapes": shape_summaries,
            "movieclips": movieclip_summaries,
        }


def add_runtime_paths() -> None:
    for path in (APK_ASSETS_SITE, GENERATED_SC2_ROOT):
        if not path.is_dir():
            raise Sc2ParseError(f"Missing runtime path: {path}")
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_runtime() -> dict[str, Any]:
    add_runtime_paths()

    from mb_sc_tools.codec import inspect_container
    from sc.flash.SC2.DataStorage import DataStorage
    from sc.flash.SC2.ExportNames import ExportNames
    from sc.flash.SC2.FileDescriptor import FileDescriptor
    from sc.flash.SC2.MovieClipModifiers import MovieClipModifiers
    from sc.flash.SC2.MovieClips import MovieClips
    from sc.flash.SC2.Precision import Precision
    from sc.flash.SC2.Shapes import Shapes
    from sc.flash.SC2.TextFields import TextFields
    from sc.flash.SC2.Textures import Textures

    return {
        "inspect_container": inspect_container,
        "DataStorage": DataStorage,
        "ExportNames": ExportNames,
        "FileDescriptor": FileDescriptor,
        "MovieClipModifiers": MovieClipModifiers,
        "MovieClips": MovieClips,
        "Precision": Precision,
        "Shapes": Shapes,
        "TextFields": TextFields,
        "Textures": Textures,
    }


def fb_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


def read_root(blob: bytes, root_cls: Any, label: str) -> tuple[Any, int]:
    if len(blob) < 4:
        raise Sc2ParseError(f"{label} is truncated before root offset")
    return root_cls.GetRootAs(blob, 0), len(blob)


def read_size_prefixed_root(
    blob: bytes, root_cls: Any, label: str
) -> tuple[Any, int, int]:
    if len(blob) < 4:
        raise Sc2ParseError(f"{label} is truncated before size prefix")
    size = struct.unpack_from("<I", blob, 0)[0]
    end = 4 + size
    if end > len(blob):
        raise Sc2ParseError(
            f"{label} exceeds available bytes: need {end}, have {len(blob)}"
        )
    return root_cls.GetRootAs(blob[4:end], 0), size, end


def read_chunk(
    blob: bytes, offset: int, root_cls: Any, label: str
) -> tuple[Any, int, dict[str, int]]:
    if offset + 4 > len(blob):
        raise Sc2ParseError(f"{label} size prefix out of bounds at offset {offset}")
    size = struct.unpack_from("<I", blob, offset)[0]
    start = offset + 4
    end = start + size
    if end > len(blob):
        raise Sc2ParseError(
            f"{label} chunk exceeds available bytes: need {end}, have {len(blob)}"
        )
    return root_cls.GetRootAs(blob[start:end], 0), end, {"offset": offset, "size": size}


def precision_multiplier(precision: int) -> float:
    if precision == 2:
        return 20.0
    if precision == 3:
        return 1024.0
    return 1.0


def raw_vector_bytes(root: Any, slot_offset: int, element_size: int) -> bytes:
    tab = root._tab
    vector_offset = tab.Offset(slot_offset)
    if vector_offset == 0:
        return b""
    start = tab.Vector(vector_offset)
    length = tab.VectorLen(vector_offset) * element_size
    return bytes(tab.Bytes[start : start + length])


def parse_matrix_banks(storage: Any, descriptor: Any) -> list[dict[str, Any]]:
    scale = precision_multiplier(descriptor.ScalePrecision())
    translation = precision_multiplier(descriptor.TranslationPrecision())
    result = []
    for index in range(storage.MatrixBanksLength()):
        bank = storage.MatrixBanks(index)
        matrices = []
        for matrix_index in range(bank.MatricesLength()):
            matrix = bank.Matrices(matrix_index)
            matrices.append(
                Sc2Matrix(
                    a=matrix.A(),
                    b=matrix.B(),
                    c=matrix.C(),
                    d=matrix.D(),
                    tx=matrix.Tx(),
                    ty=matrix.Ty(),
                )
            )
        if not matrices and bank.HalfMatricesLength():
            for matrix_index in range(bank.HalfMatricesLength()):
                matrix = bank.HalfMatrices(matrix_index)
                matrices.append(
                    Sc2Matrix(
                        a=matrix.A() / scale,
                        b=matrix.B() / scale,
                        c=matrix.C() / scale,
                        d=matrix.D() / scale,
                        tx=matrix.Tx() / translation,
                        ty=matrix.Ty() / translation,
                    )
                )
        colors = []
        for color_index in range(bank.ColorsLength()):
            color = bank.Colors(color_index)
            colors.append(
                Sc2ColorTransform(
                    r_mul=color.RMul(),
                    g_mul=color.GMul(),
                    b_mul=color.BMul(),
                    alpha=color.Alpha(),
                    r_add=color.RAdd(),
                    g_add=color.GAdd(),
                    b_add=color.BAdd(),
                )
            )
        result.append(
            {
                "index": index,
                "matrices": [asdict(matrix) for matrix in matrices],
                "colors": [asdict(color) for color in colors],
            }
        )
    return result


def parse_exports(
    exports_root: Any, strings: list[str], shape_ids: set[int], movieclip_ids: set[int]
) -> list[dict[str, Any]]:
    exports = []
    for index in range(exports_root.ObjectIdsLength()):
        object_id = exports_root.ObjectIds(index)
        name_ref = exports_root.NameRefIds(index)
        name = strings[name_ref] if 0 <= name_ref < len(strings) else None
        kind = "unknown"
        if object_id in shape_ids:
            kind = "shape"
        elif object_id in movieclip_ids:
            kind = "movieclip"
        exports.append({"object_id": object_id, "name": name, "kind": kind})
    return exports


def parse_textures(textures_root: Any) -> list[Sc2TextureSet]:
    result = []
    for index in range(textures_root.TexturesLength()):
        texture_set = textures_root.Textures(index)

        def parse_texture_data(texture: Any) -> Sc2TextureData | None:
            if texture is None:
                return None
            return Sc2TextureData(
                texture_format=texture.TextureFormat(),
                pixel_type=texture.PixelType(),
                width=texture.Width(),
                height=texture.Height(),
                inline_length=texture.DataLength(),
                inline_data=(
                    bytes(texture.DataAsNumpy()) if texture.DataLength() else None
                ),
                external_texture=fb_string(texture.ExternalTexture()),
            )

        result.append(
            Sc2TextureSet(
                index=index,
                lowres=parse_texture_data(texture_set.Lowres()),
                highres=parse_texture_data(texture_set.Highres()),
            )
        )
    return result


def parse_shapes(
    shapes_root: Any, shape_points: bytes, export_map: dict[int, str]
) -> list[Sc2Shape]:
    result = []
    for index in range(shapes_root.ShapesLength()):
        shape = shapes_root.Shapes(index)
        commands = []
        for command_index in range(shape.CommandsLength()):
            command = shape.Commands(command_index)
            vertices = []
            for vertex_index in range(command.PointsCount()):
                point_index = command.PointsOffset() + vertex_index
                point_offset = point_index * 12
                if point_offset + 12 > len(shape_points):
                    raise Sc2ParseError(
                        f"shape {shape.Id()} command {command_index} vertex {vertex_index} exceeds bitmap point buffer"
                    )
                x, y, u_raw, v_raw = struct.unpack_from(
                    "<ffHH", shape_points, point_offset
                )
                vertices.append(
                    Sc2ShapeVertex(x=x, y=y, u=u_raw / 65535.0, v=v_raw / 65535.0)
                )
            commands.append(
                Sc2ShapeCommand(
                    texture_index=command.TextureIndex(),
                    points_count=command.PointsCount(),
                    points_offset=command.PointsOffset(),
                    vertices=vertices,
                )
            )
        result.append(
            Sc2Shape(
                id=shape.Id(), export_name=export_map.get(shape.Id()), commands=commands
            )
        )
    return result


def parse_movieclips(
    movieclips_root: Any, storage: Any, strings: list[str], export_map: dict[int, str]
) -> list[Sc2MovieClip]:
    frame_elements_raw = raw_vector_bytes(storage, 12, 2)
    frame_elements = (
        struct.unpack(f"<{len(frame_elements_raw) // 2}H", frame_elements_raw)
        if frame_elements_raw
        else ()
    )

    result = []
    for index in range(movieclips_root.MovieclipsLength()):
        movieclip = movieclips_root.Movieclips(index)
        children = []
        child_count = movieclip.ChildrenIdsLength()
        for child_index in range(child_count):
            name = None
            if child_index < movieclip.ChildrenNameRefIdsLength():
                name_ref = movieclip.ChildrenNameRefIds(child_index)
                if 0 <= name_ref < len(strings):
                    name = strings[name_ref]
            blend_mode = (
                movieclip.ChildrenBlending(child_index)
                if child_index < movieclip.ChildrenBlendingLength()
                else None
            )
            children.append(
                Sc2MovieChild(
                    object_id=movieclip.ChildrenIds(child_index),
                    name=name,
                    blend_mode=blend_mode,
                )
            )

        frames = []
        frame_defs = []
        if movieclip.FramesLength():
            for frame_index in range(movieclip.FramesLength()):
                frame = movieclip.Frames(frame_index)
                label = None
                label_ref = frame.LabelRefId()
                if 0 <= label_ref < len(strings):
                    label = strings[label_ref]
                frame_defs.append((frame.UsedTransform(), label))
        elif movieclip.ShortFramesLength():
            for frame_index in range(movieclip.ShortFramesLength()):
                frame = movieclip.ShortFrames(frame_index)
                frame_defs.append((frame.UsedTransform(), None))

        element_offset = movieclip.FrameElementsOffset()
        if element_offset == 0xFFFFFFFF:
            element_offset = 0
        for frame_index, (elements_count, label) in enumerate(frame_defs):
            elements = []
            for _ in range(elements_count):
                base = element_offset
                if base + 2 >= len(frame_elements):
                    raise Sc2ParseError(
                        f"movieclip {movieclip.Id()} frame {frame_index} exceeds frame element buffer"
                    )
                elements.append(
                    Sc2FrameElement(
                        instance_index=frame_elements[base],
                        matrix_index=frame_elements[base + 1],
                        color_transform_index=frame_elements[base + 2],
                    )
                )
                element_offset += 3
            frames.append(
                Sc2Frame(
                    index=frame_index,
                    elements_count=elements_count,
                    label=label,
                    elements=elements,
                )
            )

        result.append(
            Sc2MovieClip(
                id=movieclip.Id(),
                export_name=export_map.get(movieclip.Id()),
                framerate=movieclip.Framerate(),
                matrix_bank_index=movieclip.MatrixBankIndex(),
                frames_count=movieclip.FramesCount(),
                children=children,
                frames=frames,
            )
        )
    return result


def descriptor_dict(descriptor: Any) -> dict[str, Any]:
    return {
        "translation_precision": descriptor.TranslationPrecision(),
        "scale_precision": descriptor.ScalePrecision(),
        "shape_count": descriptor.ShapeCount(),
        "movie_clips_count": descriptor.MovieClipsCount(),
        "texture_count": descriptor.TextureCount(),
        "text_fields_count": descriptor.TextFieldsCount(),
        "resources_offset": descriptor.ResourcesOffset(),
        "textures_length": descriptor.TexturesLength(),
        "compressed_size": descriptor.CompressedSize(),
        "external_matrix_bank_size": descriptor.ExternalMatrixBankSize(),
    }


def parse_modern_sc(path: Path) -> Sc2ParsedFile:
    runtime = load_runtime()
    inspect_container = runtime["inspect_container"]

    info, payload = inspect_container(path.read_bytes())
    if not info.get("modern"):
        raise Sc2ParseError(f"{path} is not a modern SC2 container")

    metadata = bytes.fromhex(info.get("metadata_hex", ""))
    descriptor, descriptor_size = read_root(
        metadata, runtime["FileDescriptor"], "descriptor"
    )
    storage, data_storage_size, data_storage_end = read_size_prefixed_root(
        payload, runtime["DataStorage"], "data_storage"
    )

    strings = [
        fb_string(storage.Strings(index)) or ""
        for index in range(storage.StringsLength())
    ]
    matrix_banks = parse_matrix_banks(storage, descriptor)

    chunk_offset = descriptor.ResourcesOffset()
    export_names, chunk_offset, export_info = read_chunk(
        payload, chunk_offset, runtime["ExportNames"], "export_names"
    )
    text_fields, chunk_offset, text_info = read_chunk(
        payload, chunk_offset, runtime["TextFields"], "text_fields"
    )
    shapes_root, chunk_offset, shapes_info = read_chunk(
        payload, chunk_offset, runtime["Shapes"], "shapes"
    )
    movieclips_root, chunk_offset, movieclips_info = read_chunk(
        payload, chunk_offset, runtime["MovieClips"], "movieclips"
    )
    modifiers, chunk_offset, modifiers_info = read_chunk(
        payload, chunk_offset, runtime["MovieClipModifiers"], "movieclip_modifiers"
    )
    textures_root, chunk_offset, textures_info = read_chunk(
        payload, chunk_offset, runtime["Textures"], "textures"
    )

    shape_ids = {
        shapes_root.Shapes(index).Id() for index in range(shapes_root.ShapesLength())
    }
    movieclip_ids = {
        movieclips_root.Movieclips(index).Id()
        for index in range(movieclips_root.MovieclipsLength())
    }
    exports = parse_exports(export_names, strings, shape_ids, movieclip_ids)
    export_map = {item["object_id"]: item["name"] for item in exports if item["name"]}

    shape_points = raw_vector_bytes(storage, 14, 1)
    shapes = parse_shapes(shapes_root, shape_points, export_map)
    movieclips = parse_movieclips(movieclips_root, storage, strings, export_map)
    textures = parse_textures(textures_root)

    return Sc2ParsedFile(
        source=str(path),
        container=info,
        descriptor=descriptor_dict(descriptor),
        chunks={
            "descriptor_size": descriptor_size,
            "descriptor_total_bytes": descriptor_size,
            "data_storage_size": data_storage_size,
            "data_storage_total_bytes": data_storage_end,
            "resources_end": chunk_offset,
            "payload_bytes": len(payload),
            "metadata_bytes": len(metadata),
            "export_names": export_info,
            "text_fields": text_info,
            "shapes": shapes_info,
            "movieclips": movieclips_info,
            "movieclip_modifiers": modifiers_info,
            "textures": textures_info,
            "text_field_count": text_fields.TextfieldsLength(),
            "movieclip_modifier_count": modifiers.ModifiersLength(),
        },
        strings=strings,
        matrix_banks=matrix_banks,
        exports=exports,
        shapes=shapes,
        movieclips=movieclips,
        textures=textures,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect modern Supercell SC2 files")
    parser.add_argument("input", type=Path, help="Path to a modern .sc file")
    parser.add_argument("-o", "--output", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Write full parsed structure instead of summary",
    )
    args = parser.parse_args()

    parsed = parse_modern_sc(args.input)
    data = asdict(parsed) if args.full else parsed.summary()
    text = json.dumps(data, indent=2, sort_keys=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
