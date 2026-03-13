# Building Shortening

Collapse the vertical height of building sprites while preserving their base and top details. This enables "short walls" style mods and compact building visuals.

## Concept

Every isometric building sprite has three vertical zones:

```
  ╱╲          ← top details (crenellations, roof peaks)
 ╱  ╲
╱ mid ╲       ← middle body (repetitive stone/brick)
╲ zone ╱
 ╲  ╱
  ╲╱          ← foundation base
```

Shortening removes a horizontal band from the middle and shifts the top down onto the base:

```
Before:          After:
  ╱╲              ╱╲
 ╱  ╲            ╱  ╲
╱    ╲           ╲  ╱   ← top details now sit closer to base
╲    ╱            ╲╱
 ╲  ╱
  ╲╱
```

The cut follows a **V-shaped line** matching the isometric diamond angle, so the collapse looks natural from the isometric perspective.

## V-Line Geometry

The cut line uses the same isometric math as the existing foundation diamond. Two V-lines define the removed band:

```
cut_bottom_y(px) = foundation_top_y(px) + keep_bottom
cut_top_y(px)    = foundation_top_y(px) + keep_bottom + remove_height
```

Where `foundation_top_y(px)` is the per-pixel-column top edge of the isometric diamond (already computed by `compute_dither_masks`):

```python
top_y = max(-margin_u - dx/2, -margin_v + dx/2)
```

- `keep_bottom`: pixels above foundation to preserve (the base zone)
- `remove_height`: pixels to delete (the middle zone)
- Everything above `cut_top_y` shifts down by `remove_height`

For walls (1×1 tile), typical values might be `keep_bottom=8, remove_height=24` to cut half the wall height.

## Processing Pipeline

### Approach: Full Decode → Collapse → Re-encode

Since shortening changes the sprite's vertical geometry (pixel rows are removed, canvas height shrinks), we cannot stay purely in the compressed block domain. The pipeline:

1. **Decode** all layers (main, shadow, playercolor, damage) to RGBA/grayscale pixel arrays
2. **Compute the V-line** per pixel column using existing isometric diamond math
3. **Collapse**: for each pixel column, remove rows between `cut_bottom_y` and `cut_top_y`, shift everything above downward
4. **Re-encode** to DXT1/BC4 blocks
5. **Rebuild** SLD layer commands (skip/draw) for the new, shorter canvas

### Per-Column Collapse

For each pixel column `px`:

```
foundation_top = max(-margin_u - dx/2, -margin_v + dx/2)  # existing math
cut_bottom = cy + foundation_top + keep_bottom
cut_top    = cut_bottom + remove_height

output_column = concat(
    pixels[0 : cut_bottom],           # base zone (unchanged)
    pixels[cut_top : canvas_height]    # top zone (shifted down)
)
```

Canvas height shrinks by `remove_height`. The hotspot y-coordinate stays the same (foundation position unchanged).

### Block Re-encoding

After pixel-level collapse:

1. Pad canvas dimensions to 4×4 block alignment
2. Encode each 4×4 tile to DXT1 (main/damage) or BC4 (shadow/playercolor)
3. Generate skip/draw commands: skip fully-transparent blocks, draw the rest
4. Update frame header with new canvas dimensions

We can use `dxt.py`'s existing `encode_dxt1_block` (to be added) or a simple encoder since quality loss is acceptable for a mod tool.

### DXT1 Encoding (New)

Need a basic DXT1 encoder in `dxt.py`. For mod quality, a simple approach suffices:

```python
def encode_dxt1_block(pixels_4x4_rgba):
    """Encode 4x4 RGBA pixels to 8-byte DXT1 block.

    Uses endpoint selection from min/max luminance pixels.
    Supports transparent pixels (index 3 in transparent mode).
    """
```

Similarly for BC4:

```python
def encode_bc4_block(values_4x4):
    """Encode 4x4 single-channel values to 8-byte BC4 block."""
```

## CLI Interface

```
--shorten HEIGHT     Remove HEIGHT pixels from building middles (default: 0, off)
--keep-bottom N      Pixels above foundation to preserve before cut (default: 8)
```

Examples:

```bash
# Short walls: remove 24px from middle
python build_mod.py --file wall.sld --shorten 24

# Short walls with transparency
python build_mod.py --file wall.sld --shorten 24 --dither-intensity 8

# Compact castles: remove 32px, keep more base
python build_mod.py --file castle.sld --shorten 32 --keep-bottom 16
```

## Interaction with Existing Features

| Feature | Interaction |
|---------|-------------|
| **Dithering** | Applied after shortening on the collapsed sprite |
| **Foundation outline** | Drawn at original foundation position (unchanged by shortening) |
| **Edge protection** | Computed on collapsed sprite silhouette |
| **Animation protection** | Animated blocks shift with their pixels; delta frames need re-encoding |
| **Contour** | Applied after shortening |

Processing order in `process_frame`:

```
1. Shorten (collapse pixels)     ← NEW
2. Re-encode to blocks           ← NEW
3. Compute dither masks           (existing)
4. Compute edge protection        (existing)
5. Compute outline masks          (existing)
6. Apply masks to blocks          (existing)
```

## Animated Buildings

Delta-encoded frames (mills, folwarks) need special handling:

- **Option A**: Resolve deltas to full frames, shorten each, re-encode as full frames (loses delta compression, increases file size)
- **Option B**: Only shorten frame 0, adjust delta frame block positions by the vertical shift (complex, may break if delta blocks span the cut zone)

Recommendation: Start with **Option A** for correctness. Animated buildings that would benefit from shortening (e.g., towers) are rare. File size increase is acceptable for a mod.

## Implementation Steps

1. Add `decode_dxt1_block` and `encode_dxt1_block` to `dxt.py`
2. Add `decode_bc4_block` and `encode_bc4_block` to `dxt.py`
3. Add `decode_frame_to_pixels(frame) → dict[layer_type, ndarray]` to `sld.py` or a new module
4. Add `encode_pixels_to_frame(pixels, ...) → SLDFrame`
5. Add `shorten_pixels(pixels, cx, cy, margins, keep_bottom, remove_height) → pixels` to `build_mod.py`
6. Integrate into `process_frame`: if shortening enabled, decode → shorten → re-encode before mask computation
7. Add CLI arguments `--shorten` and `--keep-bottom`
8. Add tests for the collapse geometry and round-trip encode/decode

## Edge Cases

- **Columns outside the diamond**: no foundation top edge exists → no shortening applied, those columns pass through unchanged
- **remove_height > available pixels**: clamp to available height above `cut_bottom`
- **Block alignment**: after collapse, canvas may not be 4-aligned → pad with transparent pixels
- **Empty blocks**: after shortening, some previously-drawn blocks may become fully transparent → emit as skip commands
- **Gates**: directional gates (1×2, 2×1) use asymmetric diamonds → V-line follows the same asymmetry
