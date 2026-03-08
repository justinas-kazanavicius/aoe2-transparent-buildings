"""Analyze animated building sprites: frame counts, changing pixels, foundation overlap.

Usage:
    uv run check-animations                     # List all animated buildings
    uv run check-animations <filename>           # Analyze specific file's pixel changes
    uv run check-animations --type mill          # Analyze all mills
    uv run check-animations --min-frames 10      # Only show buildings with 10+ frames
"""
import os, sys, glob, struct, argparse
from collections import defaultdict

import numpy as np

from paths import get_graphics_dir
from sld import parse_sld, get_layer, get_block_positions, LAYER_MAIN
from dxt import decode_dxt1_block


def list_animated(gfx, min_frames=2):
    """List all buildings with multiple frames."""
    files = sorted(glob.glob(os.path.join(gfx, 'b_*_x1.sld')))
    files = [f for f in files if '_destruction_' not in f and '_rubble_' not in f]

    by_frames = defaultdict(list)
    for path in files:
        name = os.path.basename(path)
        with open(path, 'rb') as f:
            header = f.read(16)
        if len(header) < 16:
            continue
        _magic, _ver, num_frames, _u1, _u2, _u3 = struct.unpack_from('<4s4HI', header)
        if num_frames >= min_frames:
            by_frames[num_frames].append(name)

    if not by_frames:
        print("No animated buildings found.")
        return

    total = 0
    for count in sorted(by_frames.keys(), reverse=True):
        names = by_frames[count]
        total += len(names)
        print(f"\n=== {count} frames ({len(names)} files) ===")
        for name in names:
            print(f"  {name}")

    print(f"\nTotal: {total} animated building files (x1 only)")


def decode_frame_pixels(frame):
    """Decode all main layer DXT1 blocks into a pixel canvas.

    Returns (canvas, mask) where canvas is (H, W, 4) RGBA uint8 and
    mask is (H, W) bool indicating which pixels have data.
    """
    main = get_layer(frame, LAYER_MAIN)
    if not main:
        return None, None

    w = frame.canvas_width
    h = frame.canvas_height
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)

    positions = get_block_positions(main, frame)
    for block_idx, bx, by in positions:
        if block_idx >= len(main.blocks):
            break
        pixels = decode_dxt1_block(main.blocks[block_idx])
        # Clip to canvas bounds
        for r in range(4):
            for c in range(4):
                py, px = by + r, bx + c
                if 0 <= py < h and 0 <= px < w:
                    canvas[py, px] = pixels[r, c]
                    mask[py, px] = True

    return canvas, mask


def compute_foundation_mask(frame, tiles):
    """Compute a boolean mask of pixels inside the isometric foundation diamond.

    Uses the same geometry as build_mod: the diamond defined by the building's
    tile footprint centered at (center_x, center_y).
    """
    main = get_layer(frame, LAYER_MAIN)
    if not main:
        return None

    w = frame.canvas_width
    h = frame.canvas_height
    cx = frame.center_x
    cy = frame.center_y

    tiles_u, tiles_v = tiles
    # Half-height per tile: compute from layer width
    layer_w = main.offset_x2 - main.offset_x1
    tile_w_guess = layer_w / max(tiles_u, tiles_v) if max(tiles_u, tiles_v) > 0 else 96
    tile_hh = tile_w_guess / 4  # tile half-height = tile_width / 4

    margin_u = tiles_u * tile_hh
    margin_v = tiles_v * tile_hh

    ys, xs = np.mgrid[0:h, 0:w]
    dx = (xs - cx).astype(np.float64)
    dy = (ys - cy).astype(np.float64)

    # Isometric diamond: top edge and bottom edge
    top_y = np.maximum(-margin_u - dx * 0.5, -margin_v + dx * 0.5)
    bottom_y = np.minimum(margin_u - dx * 0.5, margin_v + dx * 0.5)

    inside = (dy >= top_y) & (dy < bottom_y)
    return inside


def analyze_file(filepath, tiles=None, max_frames=None):
    """Analyze pixel changes across frames for one file."""
    name = os.path.basename(filepath)
    with open(filepath, 'rb') as f:
        data = f.read()

    sld = parse_sld(data)
    num_frames = sld.num_frames

    if num_frames <= 1:
        print(f"{name}: only {num_frames} frame, nothing to analyze.")
        return

    limit = min(num_frames, max_frames) if max_frames else num_frames
    print(f"\n{name}: {num_frames} frames (analyzing {limit})")

    # Get canvas size from first frame
    first = sld.frames[0]
    cw, ch = first.canvas_width, first.canvas_height
    print(f"  Canvas: {cw}x{ch}, Center: ({first.center_x},{first.center_y})")

    # Decode all frames
    print(f"  Decoding {limit} frames...", end=" ", flush=True)
    canvases = []
    masks = []
    for i in range(limit):
        canvas, mask = decode_frame_pixels(sld.frames[i])
        if canvas is None:
            print(f"\n  Frame {i} has no main layer, skipping.")
            continue
        # Resize if canvas dimensions differ across frames
        if canvas.shape[0] != ch or canvas.shape[1] != cw:
            new_canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
            new_mask = np.zeros((ch, cw), dtype=bool)
            mh, mw = min(canvas.shape[0], ch), min(canvas.shape[1], cw)
            new_canvas[:mh, :mw] = canvas[:mh, :mw]
            new_mask[:mh, :mw] = mask[:mh, :mw]
            canvas, mask = new_canvas, new_mask
        canvases.append(canvas)
        masks.append(mask)
    print("done.")

    if len(canvases) < 2:
        print("  Not enough frames to compare.")
        return

    # Stack for analysis
    all_canvases = np.stack(canvases)  # (F, H, W, 4)
    all_masks = np.stack(masks)  # (F, H, W)

    # A pixel "exists" if it has alpha > 0 in any frame
    ever_drawn = all_masks.any(axis=0)  # (H, W)
    total_drawn = int(ever_drawn.sum())

    # A pixel "changes" if its RGBA value differs across frames
    # Compare each frame's RGB to the first frame where it exists
    ref = all_canvases[0]  # reference frame
    changes = np.zeros((ch, cw), dtype=bool)
    for i in range(1, len(canvases)):
        # Pixel changed if it was drawn in both frames but has different color,
        # or drawn in one but not the other
        both_drawn = all_masks[0] & all_masks[i]
        diff_color = (all_canvases[i] != ref).any(axis=-1) & both_drawn
        drawn_mismatch = all_masks[0] != all_masks[i]
        changes |= diff_color | drawn_mismatch

    total_changing = int(changes.sum())

    # Count how many unique RGBA values each pixel takes across frames
    # For changing pixels, count frame distribution
    changing_ys, changing_xs = np.where(changes)
    value_counts = np.zeros((ch, cw), dtype=np.int32)
    if total_changing > 0:
        for i in range(len(canvases)):
            # For each frame, hash the RGBA into a single uint32
            pass
        # Simpler: count how many distinct (R,G,B,A) tuples each changing pixel has
        for y, x in zip(changing_ys[:min(len(changing_ys), 10000)], changing_xs[:min(len(changing_xs), 10000)]):
            seen = set()
            for i in range(len(canvases)):
                val = tuple(all_canvases[i, y, x].tolist())
                seen.add(val)
            value_counts[y, x] = len(seen)

    # Stability: for each pixel, in how many frames does it match its most common value?
    # This tells us: "pixel X is stable 89/90 frames, changes only 1 frame"
    # We'll compute this for changing pixels only
    stability = np.full((ch, cw), len(canvases), dtype=np.int32)
    if total_changing > 0 and total_changing <= 50000:
        for y, x in zip(changing_ys, changing_xs):
            counts = defaultdict(int)
            for i in range(len(canvases)):
                val = tuple(all_canvases[i, y, x].tolist())
                counts[val] += 1
            stability[y, x] = max(counts.values())

    # Foundation analysis
    if tiles is None:
        tiles = (2, 2)  # default guess
    foundation = compute_foundation_mask(first, tiles)
    if foundation is not None:
        in_foundation = changes & foundation
        above_foundation = changes & ~foundation
        total_in_foundation = int(in_foundation.sum())
        total_above = int(above_foundation.sum())
    else:
        total_in_foundation = 0
        total_above = total_changing

    # Print summary
    print(f"\n  === Pixel Change Summary ===")
    print(f"  Total drawn pixels (any frame): {total_drawn:,}")
    print(f"  Static pixels (same all frames): {total_drawn - total_changing:,} ({100*(total_drawn-total_changing)/max(total_drawn,1):.1f}%)")
    print(f"  Changing pixels: {total_changing:,} ({100*total_changing/max(total_drawn,1):.1f}%)")
    print(f"    In foundation:    {total_in_foundation:,}")
    print(f"    Above foundation: {total_above:,}")

    if total_changing > 0:
        # Stability breakdown
        stable_90 = int(((stability[changes] >= len(canvases) * 0.9) & changes[changes]).sum())
        stable_50 = int(((stability[changes] >= len(canvases) * 0.5) & changes[changes]).sum())
        print(f"\n  === Stability (most common value frequency) ===")
        print(f"  Stable >=90% of frames: {stable_90:,} pixels")
        print(f"  Stable >=50% of frames: {stable_50:,} pixels")
        print(f"  Highly variable (<50%): {total_changing - stable_50:,} pixels")

    if total_changing > 0 and total_changing <= 10000:
        # Bounding box of changing region
        min_y, max_y = int(changing_ys.min()), int(changing_ys.max())
        min_x, max_x = int(changing_xs.min()), int(changing_xs.max())
        print(f"\n  === Changing Region Bounds ===")
        print(f"  X: {min_x} to {max_x} ({max_x - min_x + 1}px wide)")
        print(f"  Y: {min_y} to {max_y} ({max_y - min_y + 1}px tall)")
        print(f"  Center: ({first.center_x}, {first.center_y})")
        if foundation is not None:
            print(f"  Foundation top Y ~= {first.center_y - int(tiles[0] * 24 / 2)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze animated building sprites")
    parser.add_argument('file', nargs='?', help='Specific SLD filename to analyze')
    parser.add_argument('--type', help='Building type to analyze (e.g. mill, stable)')
    parser.add_argument('--min-frames', type=int, default=2, help='Minimum frames to list (default: 2)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Max frames to decode per file (default: all)')
    parser.add_argument('--tiles', type=int, default=None,
                        help='Footprint size in tiles (e.g. 2 for 2x2)')
    args = parser.parse_args()

    gfx = get_graphics_dir()

    if args.file:
        filepath = os.path.join(gfx, args.file)
        tiles = (args.tiles, args.tiles) if args.tiles else None
        analyze_file(filepath, tiles=tiles, max_frames=args.max_frames)
    elif args.type:
        pattern = os.path.join(gfx, f'b_*_{args.type}_*_x1.sld')
        files = sorted(glob.glob(pattern))
        files = [f for f in files if '_destruction_' not in f and '_rubble_' not in f]
        if not files:
            print(f"No files matching type '{args.type}'")
            return
        tiles = (args.tiles, args.tiles) if args.tiles else None
        for path in files:
            analyze_file(path, tiles=tiles, max_frames=args.max_frames)
    else:
        list_animated(gfx, min_frames=args.min_frames)


if __name__ == '__main__':
    main()
