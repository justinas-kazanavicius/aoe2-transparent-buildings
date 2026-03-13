"""Building shortening: collapse vertical height of building sprites.

Removes one or more V-shaped bands from building sprites, following the
isometric diamond angle so the collapse looks natural. Each cut is defined
by an offset (pixels above foundation top) and a height (pixels to remove).
Multiple cuts can stack to chop different sections of a building.
"""

import numpy as np

from dxt import decode_dxt1_block, encode_dxt1_block, decode_bc4_block, encode_bc4_block
from sld import (
    get_layer, get_block_positions,
    LAYER_MAIN, LAYER_SHADOW, LAYER_DAMAGE, LAYER_PLAYERCOLOR,
    DXT1_LAYERS,
)


def resolve_cuts(cuts, cy, margin_u, margin_v):
    """Convert cuts with direction to foundation-relative (offset, height) tuples.

    Each cut can be (offset, height) or (offset, height, direction).
    direction='bottom' (default): offset is pixels above foundation top.
    direction='top': offset is pixels below sprite top; converted using
    the center column's foundation top position.
    """
    ft_center = cy - max(margin_u, margin_v)
    result = []
    for item in cuts:
        if len(item) == 3:
            offset, height, direction = item
        else:
            offset, height = item
            direction = 'bottom'
        if direction == 'top':
            foundation_offset = max(0, ft_center - (offset + height))
            result.append((foundation_offset, height))
        else:
            result.append((offset, height))
    return result


def shorten_sld(sld, tile_hh, tiles_u, tiles_v, cuts, preview_only=False):
    """Shorten all frames in an SLD by removing V-shaped bands.

    Modifies the SLD in-place. Delta frames are resolved to full frames.

    Args:
        sld: Parsed SLDFile object
        tile_hh: Tile half-height in pixels
        tiles_u: Footprint tiles in U direction
        tiles_v: Footprint tiles in V direction
        cuts: list of (offset, height) tuples where:
            offset = pixels above foundation top edge where cut zone ends
            height = pixels to remove
        preview_only: if True, only process frame 0 (faster for preview)
    """
    # Filter to active cuts
    active = [(o, h) for o, h in cuts if h > 0]
    if not sld.frames or not active:
        return

    total_remove = sum(h for _, h in active)
    frame0 = sld.frames[0]
    cx = frame0.center_x
    cy = frame0.center_y
    margin_u = tiles_u * tile_hh
    margin_v = tiles_v * tile_hh
    new_h = frame0.canvas_height - total_remove

    if new_h <= 0:
        return

    if preview_only:
        # Only decode and shorten frame 0
        frame_px = _resolve_single_frame(sld.frames[0])
        _shorten_frame(sld.frames[0], frame_px, cx, cy,
                       margin_u, margin_v, active, total_remove, new_h)
        return

    has_delta = any(
        _layer_is_delta(get_layer(f, LAYER_MAIN))
        for f in sld.frames[1:]
    )

    all_pixels = _resolve_frames(sld, has_delta)

    for i, frame in enumerate(sld.frames):
        _shorten_frame(frame, all_pixels[i], cx, cy,
                       margin_u, margin_v, active, total_remove, new_h)


def _layer_is_delta(layer):
    return layer is not None and bool(layer.flag1 & 0x80)


def _resolve_single_frame(frame):
    """Decode a single frame to pixel dict (no delta compositing)."""
    frame_px = {}
    for layer in frame.layers:
        lt = layer.layer_type
        if lt not in (LAYER_MAIN, LAYER_SHADOW, LAYER_DAMAGE, LAYER_PLAYERCOLOR):
            continue
        is_dxt1 = lt in DXT1_LAYERS
        h, w = frame.canvas_height, frame.canvas_width
        if is_dxt1:
            canvas = np.zeros((h, w, 4), dtype=np.uint8)
        else:
            canvas = np.zeros((h, w), dtype=np.uint8)
        _overlay_blocks(canvas, layer, frame, is_dxt1)
        frame_px[lt] = canvas
    return frame_px


def _resolve_frames(sld, has_delta):
    """Decode all frames to pixel dicts. Composites delta frames on previous."""
    result = []
    prev = {}

    for frame in sld.frames:
        frame_px = {}
        for layer in frame.layers:
            lt = layer.layer_type
            if lt not in (LAYER_MAIN, LAYER_SHADOW, LAYER_DAMAGE, LAYER_PLAYERCOLOR):
                continue

            is_dxt1 = lt in DXT1_LAYERS
            is_delta = bool(layer.flag1 & 0x80)

            if is_delta and has_delta and lt in prev:
                canvas = prev[lt].copy()
            else:
                h, w = frame.canvas_height, frame.canvas_width
                if is_dxt1:
                    canvas = np.zeros((h, w, 4), dtype=np.uint8)
                else:
                    canvas = np.zeros((h, w), dtype=np.uint8)

            _overlay_blocks(canvas, layer, frame, is_dxt1)
            frame_px[lt] = canvas

        if has_delta:
            prev = {lt: px.copy() for lt, px in frame_px.items()}
        result.append(frame_px)

    return result


def _overlay_blocks(canvas, layer, frame, is_dxt1):
    """Decode and paint a layer's draw blocks onto a pixel canvas."""
    ch, cw = canvas.shape[0], canvas.shape[1]
    for bi, bx, by in get_block_positions(layer, frame):
        if bi >= len(layer.blocks):
            continue
        y1, y2 = max(by, 0), min(by + 4, ch)
        x1, x2 = max(bx, 0), min(bx + 4, cw)
        if y2 <= y1 or x2 <= x1:
            continue
        if is_dxt1:
            decoded = decode_dxt1_block(layer.blocks[bi])
            canvas[y1:y2, x1:x2] = decoded[y1 - by:y2 - by, x1 - bx:x2 - bx]
        else:
            decoded = decode_bc4_block(layer.blocks[bi])
            canvas[y1:y2, x1:x2] = decoded[y1 - by:y2 - by, x1 - bx:x2 - bx]


def _shorten_pixels(pixels, cx, cy, margin_u, margin_v, cuts, total_remove):
    """Remove V-shaped bands from a pixel array.

    Each cut is (offset, height):
        offset = pixels above foundation top where the cut zone's bottom edge is
        height = pixels to remove from this zone

    All cuts follow the same V-line (isometric diamond angle). The offset
    controls how far above the foundation each cut sits.

    Returns array with height reduced by total_remove.
    """
    old_h, old_w = pixels.shape[0], pixels.shape[1]
    new_h = old_h - total_remove
    is_rgba = pixels.ndim == 3

    if new_h <= 0:
        return pixels

    if is_rgba:
        result = np.zeros((new_h, old_w, 4), dtype=np.uint8)
    else:
        result = np.zeros((new_h, old_w), dtype=np.uint8)

    # Precompute foundation top for all columns
    cols_arr = np.arange(old_w, dtype=np.float64)
    dx = cols_arr - cx
    top_a = -margin_u - dx * 0.5
    top_b = -margin_v + dx * 0.5
    foundation_top_f = cy + np.maximum(top_a, top_b)
    foundation_top_int = np.floor(foundation_top_f).astype(np.int32)

    if not cuts:
        # No cuts - return copy trimmed/padded to new_h
        take = min(new_h, old_h)
        if is_rgba:
            result[:take, :, :] = pixels[:take, :, :]
        else:
            result[:take, :] = pixels[:take, :]
        return result

    # Precompute cut zones for all columns at once using vectorized ops
    ft_arr = foundation_top_int.astype(np.int64)  # (old_w,)
    offsets = np.array([o for o, h in cuts], dtype=np.int64)
    heights = np.array([h for o, h in cuts], dtype=np.int64)

    # ce[c, k] = ft[c] - offsets[k], cs[c, k] = ce[c, k] - heights[k]
    ce_all = ft_arr[:, None] - offsets[None, :]  # (old_w, n_cuts)
    cs_all = ce_all - heights[None, :]

    # Clamp: if cs < 0, shift both
    shift = np.maximum(-cs_all, 0)
    cs_all = cs_all + shift
    ce_all = ce_all + shift
    # Clamp: if ce > old_h, shift both
    shift2 = np.maximum(ce_all - old_h, 0)
    ce_all = ce_all - shift2
    cs_all = cs_all - shift2
    cs_all = np.maximum(cs_all, 0)  # edge case: height > old_h

    # Per-column collapse (merge zones and copy segments)
    n_cuts = len(cuts)
    for col in range(old_w):
        # Get sorted, merged zones for this column
        zones = sorted(zip(cs_all[col], ce_all[col]))
        ms, me = zones[0]
        merged = [(int(ms), int(me))]
        for cs, ce in zones[1:]:
            cs, ce = int(cs), int(ce)
            if cs <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], ce))
            else:
                merged.append((cs, ce))

        # Copy non-cut segments into result
        pos = 0
        prev = 0
        for cs, ce in merged:
            if cs > prev:
                length = cs - prev
                if is_rgba:
                    result[pos:pos + length, col, :] = pixels[prev:cs, col, :]
                else:
                    result[pos:pos + length, col] = pixels[prev:cs, col]
                pos += length
            prev = max(prev, ce)
        # Copy remainder after last cut
        remaining = new_h - pos
        if remaining > 0 and prev < old_h:
            take = min(remaining, old_h - prev)
            if is_rgba:
                result[pos:pos + take, col, :] = pixels[prev:prev + take, col, :]
            else:
                result[pos:pos + take, col] = pixels[prev:prev + take, col]

    return result


def _find_bounds(pixels, is_dxt1):
    """Find 4-pixel-aligned bounding box of non-empty content."""
    if is_dxt1:
        has_content = pixels[:, :, 3] > 0
    else:
        has_content = pixels > 0

    rows = np.any(has_content, axis=1)
    cols = np.any(has_content, axis=0)

    if not np.any(rows):
        return 0, 0, 4, 4

    y1 = (int(np.argmax(rows)) // 4) * 4
    y2 = ((int(len(rows) - np.argmax(rows[::-1])) + 3) // 4) * 4
    x1 = (int(np.argmax(cols)) // 4) * 4
    x2 = ((int(len(cols) - np.argmax(cols[::-1])) + 3) // 4) * 4

    return x1, y1, x2, y2


def _pad_to_bounds(pixels, x2, y2, is_dxt1):
    """Pad pixel array if needed to cover the required bounds."""
    h, w = pixels.shape[0], pixels.shape[1]
    pad_h = max(0, y2 - h)
    pad_w = max(0, x2 - w)
    if pad_h or pad_w:
        if is_dxt1:
            return np.pad(pixels, ((0, pad_h), (0, pad_w), (0, 0)))
        else:
            return np.pad(pixels, ((0, pad_h), (0, pad_w)))
    return pixels


def _encode_region(pixels, x1, y1, x2, y2, is_dxt1):
    """Encode a pixel region to DXT/BC4 blocks and skip/draw commands."""
    bpr = (x2 - x1) // 4
    bpc = (y2 - y1) // 4

    drawn = []
    for row in range(bpc):
        for col in range(bpr):
            by = y1 + row * 4
            bx = x1 + col * 4
            tile = pixels[by:by + 4, bx:bx + 4]

            if is_dxt1:
                if not np.any(tile[:, :, 3] > 0):
                    continue
                block = encode_dxt1_block(tile)
            else:
                if not np.any(tile > 0):
                    continue
                block = encode_bc4_block(tile)

            drawn.append((row * bpr + col, block))

    if not drawn:
        return [(0, 0)], []

    commands = []
    blocks = []
    prev_end = 0
    i = 0

    while i < len(drawn):
        pos = drawn[i][0]
        skip = pos - prev_end

        while skip > 255:
            commands.append((255, 0))
            skip -= 255

        run_start = i
        while i < len(drawn) - 1 and drawn[i + 1][0] == drawn[i][0] + 1:
            i += 1
        run_len = i - run_start + 1

        emitted = 0
        while emitted < run_len:
            chunk = min(run_len - emitted, 255)
            commands.append((skip if emitted == 0 else 0, chunk))
            for j in range(run_start + emitted, run_start + emitted + chunk):
                blocks.append(drawn[j][1])
            emitted += chunk

        prev_end = drawn[i][0] + 1
        i += 1

    return commands, blocks


def _shorten_frame(frame, frame_pixels, cx, cy, margin_u, margin_v,
                   cuts, total_remove, new_h):
    """Shorten a single frame given its resolved pixel arrays."""
    if LAYER_MAIN not in frame_pixels:
        return

    shortened = {}
    for lt, pixels in frame_pixels.items():
        shortened[lt] = _shorten_pixels(pixels, cx, cy, margin_u, margin_v,
                                        cuts, total_remove)

    frame.canvas_height = new_h
    frame.center_y = cy - total_remove

    # Re-encode MAIN layer
    main_px = shortened[LAYER_MAIN]
    x1, y1, x2, y2 = _find_bounds(main_px, True)
    main_px = _pad_to_bounds(main_px, x2, y2, True)

    main_layer = get_layer(frame, LAYER_MAIN)
    main_layer.offset_x1 = x1
    main_layer.offset_y1 = y1
    main_layer.offset_x2 = x2
    main_layer.offset_y2 = y2
    main_layer.flag1 &= ~0x80
    cmds, blks = _encode_region(main_px, x1, y1, x2, y2, True)
    main_layer.commands = cmds
    main_layer.command_count = len(cmds)
    main_layer.blocks = blks

    # Re-encode SHADOW (own bounds)
    if LAYER_SHADOW in shortened:
        shadow_px = shortened[LAYER_SHADOW]
        sx1, sy1, sx2, sy2 = _find_bounds(shadow_px, False)
        shadow_px = _pad_to_bounds(shadow_px, sx2, sy2, False)

        shadow_layer = get_layer(frame, LAYER_SHADOW)
        if shadow_layer is not None:
            shadow_layer.offset_x1 = sx1
            shadow_layer.offset_y1 = sy1
            shadow_layer.offset_x2 = sx2
            shadow_layer.offset_y2 = sy2
            shadow_layer.flag1 &= ~0x80
            cmds, blks = _encode_region(shadow_px, sx1, sy1, sx2, sy2, False)
            shadow_layer.commands = cmds
            shadow_layer.command_count = len(cmds)
            shadow_layer.blocks = blks

    # Re-encode DAMAGE and PLAYERCOLOR (share MAIN bounds)
    for lt in (LAYER_DAMAGE, LAYER_PLAYERCOLOR):
        if lt not in shortened:
            continue
        layer = get_layer(frame, lt)
        if layer is None:
            continue
        is_dxt1 = lt in DXT1_LAYERS
        px = _pad_to_bounds(shortened[lt], x2, y2, is_dxt1)
        layer.flag1 &= ~0x80
        cmds, blks = _encode_region(px, x1, y1, x2, y2, is_dxt1)
        layer.commands = cmds
        layer.command_count = len(cmds)
        layer.blocks = blks
