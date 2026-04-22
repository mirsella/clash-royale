# Clash Royale

Extracted assets from the installed Android arm64 split set for `com.supercell.clashroyale`.

## Version

- Source dump: installed phone extract + install-time asset pack
- Package name: `com.supercell.clashroyale`
- Public repo layout: flattened asset-only export

## Contents

- `audio/`: OGG music/sfx and FMOD bank files
- `fonts/`: TTF and BMFont assets
- `images/`: directly extracted PNG and PVR images
- `materials/`: `.rmat` material files
- `models/`: decoded GLBs and merged rig+animation GLBs
- `previews/`: prerendered MP4 previews for decoded and merged models
- `reports/`: extraction reports, render reports, and flat path index
- `scripts/`: helper scripts used to extract, flatten, parse SC2, merge GLBs, and render previews
- `shaders/`: shader assets
- `special/`: `.scw` special Supercell assets
- `sprites/`: decoded Supercell `.sc` / `.sctx` workspaces and rendered SC2 export frames
- `textures/`: decoded PNG exports for extracted textures

## Extraction Notes

1. Extracted the installed split APK set and install-time asset pack from an Android phone into a raw working tree.
2. Identified the main payload areas under `assets/sc`, `assets/sc3d`, `assets/sfx`, `assets/soundbanks`, `assets/font`, and related asset directories.
3. Decoded all `.sctx` textures and all `.ktx` textures to PNG, including both KTX2 ASTC and legacy KTX1 ETC1 variants.
4. Patched `Supercell-Flat-Converter` locally to decode Clash Royale SC3D `FLA2` / `SC_odin_format` GLBs into standard-structure GLBs.
5. Rewrote GLB material/image references so decoded and merged models resolve the exported texture PNGs.
6. Merged compatible animation-only GLBs with sibling rig GLBs and prerendered preview MP4s for all mesh-bearing decoded and merged models.
7. Parsed and rendered modern SC2-era Supercell `.sc` files, including inline texture payloads and external `.sctx` references, to produce export-frame PNGs for every modern `.sc` workspace.
8. Flattened the final public dump into browse-friendly top-level buckets while preserving reports and a path index.

## Notes

- This repo intentionally keeps decoded extraction outputs suitable for public release, not the original encoded package asset files.
- Non-asset/config/account-related content was excluded from the published layout.
- `reports/report.json` is the authoritative summary for counts and paths.
- `reports/path_index.json` maps flattened public paths back to their original organized dump paths.
- `reports/report.json` still records `6` unresolved GLB texture URI misses from the source dump.

## Included Scripts

- `scripts/extract_clash_royale_assets.py`: main package-specific extractor and organizer
- `scripts/flatten_clash_royale_public_dump.py`: builds the flatter public export layout
- `scripts/modern_sc2.py`: parser for modern SC2/Titan-era `.sc` files using generated FlatBuffers bindings
- `scripts/render_modern_sc2_exports.py`: renders SC2 movieclip exports into PNG frames
- `scripts/merge_supercell_animation_glbs.py`: merges animation GLBs into compatible rig GLBs
- `scripts/render_glb_preview_blender.py`: renders a single GLB preview via Blender
- `scripts/render_glb_preview_batch.py`: batch-renders preview MP4s for directories of GLBs
- `scripts/generated_sc2/`: generated FlatBuffers Python bindings for the SC2 schemas
