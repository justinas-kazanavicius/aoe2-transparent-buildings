"""Verify dithering consistency across delta-encoded frames of animated buildings.

Only checks files with delta-encoded frames (flag1 & 0x80), where the static
body should have identical dithering across all frames. Construction-stage
sprites (non-delta multi-frame) are skipped since each frame is independent.

Handles multi-direction buildings correctly: each full frame (flag1 & 0x80 == 0)
starts a new "group", and delta frames are compared against their group's full
frame, not always frame 0.

Usage:
    uv run verify-dithering b_west_mill_age3_x1.sld
    uv run verify-dithering --all              # Check all delta-animated buildings
    uv run verify-dithering --type mill         # Check all mills
"""
import os
import sys
import glob
import struct
import argparse

import numpy as np

from paths import get_graphics_dir, get_mod_graphics_dir
from sld import parse_sld, get_layer, get_block_positions, LAYER_MAIN
from dxt import decode_dxt1_block
from build_mod import get_building_tiles, TILE_WIDTH


def has_delta_frames(sld):
    """Check if any frame uses delta encoding."""
    for frame in sld.frames[1:]:
        layer = get_layer(frame, LAYER_MAIN)
        if layer and (layer.flag1 & 0x80):
            return True
    return False


def render_frame_onto(frame, prev_canvas=None):
    """Render a single main-layer frame with optional delta accumulation."""
    layer = get_layer(frame, LAYER_MAIN)
    if layer is None:
        return prev_canvas.copy() if prev_canvas is not None else None

    w = frame.canvas_width
    h = frame.canvas_height
    is_delta = prev_canvas is not None and (layer.flag1 & 0x80)

    if is_delta:
        # Delta: prev must match size
        if prev_canvas.shape[0] == h and prev_canvas.shape[1] == w:
            canvas = prev_canvas.copy()
        else:
            canvas = np.zeros((h, w, 4), dtype=np.uint8)
    else:
        canvas = np.zeros((h, w, 4), dtype=np.uint8)

    positions = get_block_positions(layer, frame)
    for block_idx, bx, by in positions:
        if block_idx >= len(layer.blocks):
            break
        pixels = decode_dxt1_block(layer.blocks[block_idx])
        for r in range(4):
            for c in range(4):
                py, px = by + r, bx + c
                if 0 <= py < h and 0 <= px < w:
                    canvas[py, px] = pixels[r, c]

    return canvas


def compute_foundation_y(cx, cy, margin_u, margin_v, xs):
    """Compute the foundation top-edge Y for each x coordinate."""
    dx = (xs - cx).astype(np.float64)
    top_a = -margin_u - dx * 0.5
    top_b = -margin_v + dx * 0.5
    return cy + np.maximum(top_a, top_b)


def group_frames(sld):
    """Group frames into (full_frame_idx, [delta_frame_indices]) sequences.

    Each full frame (flag1 bit 7 = 0) starts a new group.
    Delta frames (flag1 bit 7 = 1) belong to the preceding full frame's group.
    """
    groups = []
    current_full = None
    current_deltas = []

    for i in range(sld.num_frames):
        layer = get_layer(sld.frames[i], LAYER_MAIN)
        is_delta = layer is not None and (layer.flag1 & 0x80) and i > 0

        if is_delta:
            current_deltas.append(i)
        else:
            if current_full is not None and current_deltas:
                groups.append((current_full, current_deltas))
            current_full = i
            current_deltas = []

    if current_full is not None and current_deltas:
        groups.append((current_full, current_deltas))

    return groups


def verify_file(filename, orig_dir, mod_dir, tiles=None, verbose=False):
    """Verify dithering consistency for one file.

    Returns (filename, num_frames, num_inconsistent_frames, details).
    """
    orig_path = os.path.join(orig_dir, filename)
    mod_path = os.path.join(mod_dir, filename)

    if not os.path.exists(orig_path):
        return (filename, 0, -1, "original not found")
    if not os.path.exists(mod_path):
        return (filename, 0, -1, "mod not found")

    with open(orig_path, 'rb') as f:
        orig_sld = parse_sld(f.read())
    with open(mod_path, 'rb') as f:
        mod_sld = parse_sld(f.read())

    if orig_sld.num_frames <= 1:
        return (filename, orig_sld.num_frames, 0, "single frame")

    if not has_delta_frames(orig_sld):
        return (filename, orig_sld.num_frames, -1, "no delta frames (construction stages)")

    groups = group_frames(orig_sld)
    if not groups:
        return (filename, orig_sld.num_frames, -1, "no delta groups found")

    # Foundation geometry
    frame0 = orig_sld.frames[0]
    main0 = get_layer(frame0, LAYER_MAIN)
    if main0 is None:
        return (filename, orig_sld.num_frames, -1, "no main layer")

    if tiles is None:
        layer_w = main0.offset_x2 - main0.offset_x1
        tile_w = TILE_WIDTH['x1']
        tiles = get_building_tiles(filename, layer_w, tile_w)
    tile_hh = 24
    margin_u = tiles[0] * tile_hh
    margin_v = tiles[1] * tile_hh

    inconsistent_frames = []

    for full_idx, delta_indices in groups:
        # Render the full frame for orig and mod (with accumulation up to full_idx)
        orig_prev = None
        mod_prev = None
        for i in range(full_idx + 1):
            orig_prev = render_frame_onto(orig_sld.frames[i], orig_prev)
            mod_prev = render_frame_onto(mod_sld.frames[i], mod_prev)

        if orig_prev is None or mod_prev is None:
            continue

        # Use this group's full frame as reference
        full_frame = orig_sld.frames[full_idx]
        cx = full_frame.center_x
        cy = full_frame.center_y
        h, w = orig_prev.shape[:2]
        mh, mw = mod_prev.shape[:2]
        ch, cw = min(h, mh), min(w, mw)

        # Foundation mask for this group's canvas
        xs = np.arange(cw)
        ys = np.arange(ch)
        xx, yy = np.meshgrid(xs, ys)
        foundation_y = compute_foundation_y(cx, cy, margin_u, margin_v, xx)
        above = yy < foundation_y

        ref_orig_alpha = orig_prev[:ch, :cw, 3]
        ref_mod_alpha = mod_prev[:ch, :cw, 3]

        ref_dithered = (ref_orig_alpha > 0) & (ref_mod_alpha == 0) & above
        ref_kept = (ref_orig_alpha > 0) & (ref_mod_alpha > 0) & above

        # Now check each delta frame
        orig_canvas = orig_prev
        mod_canvas = mod_prev
        for di in delta_indices:
            orig_canvas = render_frame_onto(orig_sld.frames[di], orig_canvas)
            mod_canvas = render_frame_onto(mod_sld.frames[di], mod_canvas)

            if orig_canvas is None or mod_canvas is None:
                continue

            ih = min(orig_canvas.shape[0], mod_canvas.shape[0], ch)
            iw = min(orig_canvas.shape[1], mod_canvas.shape[1], cw)

            orig_alpha = orig_canvas[:ih, :iw, 3]
            mod_alpha = mod_canvas[:ih, :iw, 3]
            ab = above[:ih, :iw]

            frame_dithered = (orig_alpha > 0) & (mod_alpha == 0) & ab
            frame_kept = (orig_alpha > 0) & (mod_alpha > 0) & ab

            both_exist = (ref_orig_alpha[:ih, :iw] > 0) & (orig_alpha > 0) & ab

            lost_dither = ref_dithered[:ih, :iw] & both_exist & frame_kept
            gained_dither = ref_kept[:ih, :iw] & both_exist & frame_dithered

            n_lost = int(lost_dither.sum())
            n_gained = int(gained_dither.sum())
            n_total = int(both_exist.sum())

            # Edge protection legitimately varies for animated sprites (edge
            # follows the moving silhouette). Tolerate small differences
            # (< 1% of above-foundation pixels).
            if n_total > 0 and (n_lost + n_gained) / n_total > 0.01:
                inconsistent_frames.append((di, n_lost, n_gained))

    details = []
    if inconsistent_frames:
        for fi, lost, gained in inconsistent_frames[:5]:
            parts = []
            if lost:
                parts.append(f"{lost} undithered")
            if gained:
                parts.append(f"{gained} newly dithered")
            details.append(f"    frame {fi}: {', '.join(parts)}")
        if len(inconsistent_frames) > 5:
            details.append(f"    ... and {len(inconsistent_frames) - 5} more frames")

    n_frames = orig_sld.num_frames
    return (filename, n_frames, len(inconsistent_frames), '\n'.join(details))


def main():
    parser = argparse.ArgumentParser(description="Verify dithering consistency across delta frames")
    parser.add_argument('file', nargs='?', help='Specific SLD filename')
    parser.add_argument('--all', action='store_true', help='Check all delta-animated buildings')
    parser.add_argument('--type', help='Building type (e.g. mill, folwark)')
    parser.add_argument('--tiles', type=int, default=None, help='Override footprint (NxN)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    orig_dir = get_graphics_dir()
    mod_dir = get_mod_graphics_dir()
    tiles = (args.tiles, args.tiles) if args.tiles else None

    if args.file:
        files = [args.file]
    elif args.type:
        pattern = os.path.join(orig_dir, f'b_*_{args.type}_*_x1.sld')
        files = [os.path.basename(f) for f in sorted(glob.glob(pattern))
                 if '_destruction_' not in f and '_rubble_' not in f]
    elif args.all:
        pattern = os.path.join(orig_dir, 'b_*_x1.sld')
        files = []
        for path in sorted(glob.glob(pattern)):
            if '_destruction_' in path or '_rubble_' in path:
                continue
            with open(path, 'rb') as f:
                data = f.read()
            if len(data) < 16:
                continue
            _, _, num_frames, _, _, _ = struct.unpack_from('<4s4HI', data)
            if num_frames > 1:
                sld = parse_sld(data)
                if has_delta_frames(sld):
                    files.append(os.path.basename(path))
    else:
        parser.print_help()
        return

    if not files:
        print("No delta-animated files found.")
        return

    print(f"Verifying dithering consistency for {len(files)} delta-animated file(s)...")
    print(f"  Original: {orig_dir}")
    print(f"  Mod:      {mod_dir}")
    print()

    total_ok = 0
    total_bad = 0
    total_skip = 0

    for filename in files:
        name, n_frames, n_bad, detail = verify_file(
            filename, orig_dir, mod_dir, tiles=tiles, verbose=args.verbose)

        if n_bad < 0:
            print(f"  SKIP  {name}: {detail}")
            total_skip += 1
        elif n_bad == 0:
            print(f"  OK    {name} ({n_frames} frames)")
            total_ok += 1
        else:
            print(f"  FAIL  {name} ({n_frames} frames, {n_bad} inconsistent)")
            if detail:
                print(detail)
            total_bad += 1

    print(f"\nSummary: {total_ok} OK, {total_bad} FAIL, {total_skip} skipped")


if __name__ == '__main__':
    main()
