"""
Transparent Buildings Mod for AoE2 DE

Applies checkerboard dithering to the upper portions of building sprites,
keeping the building base/foundation opaque. Uses DXT1's transparent mode
(color0 <= color1, index 3 = transparent) to achieve the dithering effect.

Usage:
    uv run build-mod                # Process single prototype building
    uv run build-mod --all          # Process all buildings
    uv run build-mod --file X.sld   # Process specific file
"""

import os
import sys
import re
import glob
import argparse
import time
from multiprocessing import Pool, cpu_count
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

import numpy as np

from sld import (
    parse_sld, write_sld, get_layer, get_block_positions,
    LAYER_MAIN, LAYER_SHADOW, LAYER_DAMAGE, LAYER_PLAYERCOLOR,
    DXT1_LAYERS
)
from dxt import zero_bc4_pixels, decode_bc4_block, encode_bc4_block_from_flat, _ZERO_BLOCK
import shutil

from paths import get_graphics_dir, get_mod_dir, get_mod_graphics_dir

# Prototype test file

# Tile geometry (isometric tile half-height in pixels per scale)
TILE_HALF_HEIGHT = {'x1': 24, 'x2': 48}
TILE_WIDTH = {'x1': 96, 'x2': 192}

# Known building footprint sizes (NxN tiles) keyed by building type substring.
# Used to override the width-based heuristic which can be off for buildings
# whose sprites are wider or narrower than their actual footprint.
BUILDING_FOOTPRINT = {
    'house': 2,
    'outpost': 1,
    'tower': 1,
    'mill': 2,
    'lumber_camp': 2,
    'mining_camp': 2,
    'mule_cart': 1,
    'donjon': 2,
    'folwark': 3,
    'dock': 3,
    'shipyard': 3,
    'barracks': 3,
    'archery_range': 3,
    'stable': 3,
    'blacksmith': 3,
    'monastery': 3,
    'krepost': 3,
    'settlement': 3,
    'siege_workshop': 4,
    'market': 4,
    'university': 4,
    'town_center': 4,
    'castle': 4,
    'caravanserai': 4,
    'trade_workshop': 4,
    'wonder': 5,
    'wall': 1,
}

# Sorted longest-first so 'siege_workshop' matches before 'workshop', etc.
_FOOTPRINT_KEYS = sorted(BUILDING_FOOTPRINT.keys(), key=len, reverse=True)

# Gate direction regex: matches _ne_, _se_, _n_, _e_ (order matters: ne before n, se before e)
_GATE_DIR_RE = re.compile(r'_(ne|se|n|e)_')


def get_building_tiles(filename, layer_w, tile_w):
    """Get the footprint size (in tiles) for a building file.

    Looks up the building type from the filename in BUILDING_FOOTPRINT.
    Gates are handled specially based on direction suffix:
      - NE/N gates → (1, 2), SE/E gates → (2, 1)
      - Corners, flags, sides → (1, 1)
    Falls back to round(layer_w / tile_w) for unknown building types.

    Returns (tiles_u, tiles_v) tuple for the isometric footprint.
    For square NxN buildings, returns (N, N).
    """
    name = filename.lower()

    # Gate-specific handling: direction determines footprint orientation
    if 'gate' in name:
        m = _GATE_DIR_RE.search(name)
        if m:
            direction = m.group(1)
            if direction == 'n':
                return (1, 1)   # compound vertical |
            elif direction == 'ne':
                return (1, 2)   # / diagonal
            elif direction == 'se':
                return (2, 1)   # \ diagonal
            else:  # 'e'
                return (1, 1)   # compound horizontal <><>
        # Corner, flag, sides, etc. — treat as 1x1 wall piece
        return (1, 1)

    for key in _FOOTPRINT_KEYS:
        if key in name:
            val = BUILDING_FOOTPRINT[key]
            if isinstance(val, tuple):
                return val
            return (val, val)
    n = round(layer_w / tile_w)
    return (n, n)


# Pre-computed pixel offsets within a 4x4 DXT block (row-major order)
_PIX_ROWS = np.array([r for r in range(4) for _ in range(4)], dtype=np.int32)
_PIX_COLS = np.array([c for _ in range(4) for c in range(4)], dtype=np.int32)
_BIT_VALUES = (1 << np.arange(16, dtype=np.uint32))

# 4x4 Bayer ordered dithering threshold matrix (values 0-15)
# For threshold T (0-16): pixel transparent if BAYER[r%4][c%4] < T
# T=0 -> 0%, T=8 -> 50% (≈ checkerboard), T=16 -> 100%
_BAYER_4x4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
], dtype=np.int32)

# Precomputed: 16-bit dither mask -> 32-bit DXT1 index transparency pattern
# Each set bit i in the 16-bit mask becomes 0b11 at bits 2i+1:2i (DXT1 index 3 = transparent)
_MASK_EXPAND = np.zeros(65536, dtype=np.uint32)
for _j in range(16):
    _MASK_EXPAND += ((np.arange(65536, dtype=np.uint32) >> _j) & 1) * np.uint32(3 << (2 * _j))


def get_gate_compound_offsets(filename, tile_hh):
    """Get compound diamond offsets for N/E gates.

    N gates use two 1x1 diamonds stacked vertically (|).
    E gates use two 1x1 diamonds side by side horizontally (<><>).
    Returns list of (dx, dy) offsets from center, or None for non-compound.
    """
    name = filename.lower()
    if 'gate' not in name:
        return None
    m = _GATE_DIR_RE.search(name)
    if not m:
        return None
    direction = m.group(1)
    if direction == 'n':
        # Vertical stack: two diamonds above/below hotspot
        return [(0, -tile_hh), (0, tile_hh)]
    elif direction == 'e':
        # Horizontal stack: two diamonds left/right of hotspot
        tile_hw = 2 * tile_hh
        return [(-tile_hw, 0), (tile_hw, 0)]
    return None


def compute_compound_dither_masks(block_xs, block_ys, center_x, center_y,
                                  tile_hh, offsets, gradient_height=0,
                                  dither_intensity=8, dither_bottom=False):
    """Compute dither masks for compound diamonds (union of multiple 1x1 diamonds)."""
    if dither_intensity <= 0:
        return np.zeros(len(block_xs), dtype=np.uint32)

    M = tile_hh
    pixel_ys = block_ys[:, None] + _PIX_ROWS[None, :]
    pixel_xs = block_xs[:, None] + _PIX_COLS[None, :]

    bayer_threshold = _BAYER_4x4[pixel_ys % 4, pixel_xs % 4]
    bayer_pattern = bayer_threshold < dither_intensity
    above_hotspot = pixel_ys < center_y

    # A pixel is inside the union if inside ANY component diamond
    inside_any = np.zeros_like(pixel_ys, dtype=bool)
    below_all = np.ones_like(pixel_ys, dtype=bool)
    for ox, oy in offsets:
        dx = (pixel_xs - (center_x + ox)).astype(np.float64)
        dy = (pixel_ys - (center_y + oy)).astype(np.float64)
        top_edge = -M + np.abs(dx) * 0.5
        bot_edge = M - np.abs(dx) * 0.5
        inside = (dy >= top_edge) & (dy < bot_edge)
        inside_any |= inside
        below_all &= (dy >= bot_edge)

    above_dither = above_hotspot & ~inside_any & bayer_pattern
    if dither_bottom:
        dither = above_dither | (below_all & bayer_pattern)
    else:
        dither = above_dither

    return (dither.astype(np.uint32) * _BIT_VALUES[None, :]).sum(axis=1)


def compute_compound_outline_masks(block_xs, block_ys, center_x, center_y,
                                   tile_hh, offsets, thickness=1):
    """Compute outline masks for compound diamonds (OR of each component's outline)."""
    M = tile_hh
    result = np.zeros(len(block_xs), dtype=np.uint32)
    for ox, oy in offsets:
        result |= compute_outline_masks(
            block_xs, block_ys, center_x + ox, center_y + oy, M, M, thickness)
    return result


def compute_dither_masks(block_xs, block_ys, center_x, center_y,
                         margin_u, margin_v, gradient_height=0,
                         dither_intensity=8, dither_bottom=False):
    """
    Vectorized computation of 16-bit dither masks for N blocks at once.

    Uses a 4x4 Bayer ordered dithering matrix for variable transparency.
    dither_intensity controls pixel density: 0=opaque, 8=50%, 16=fully transparent.

    The isometric diamond is defined by two constraints in rotated space:
      |dx/2 + dy| <= margin_u  AND  |-dx/2 + dy| <= margin_v
    Top edge = max of both lower bounds, bottom edge = min of both upper bounds.
    For square NxN, margin_u == margin_v and this reduces to the simple formula.
    """
    if dither_intensity <= 0:
        return np.zeros(len(block_xs), dtype=np.uint32)

    # Expand to per-pixel coordinates: (N, 16)
    pixel_ys = block_ys[:, None] + _PIX_ROWS[None, :]
    pixel_xs = block_xs[:, None] + _PIX_COLS[None, :]

    # Bayer threshold: pixel is transparent if BAYER[r%4][c%4] < intensity
    bayer_threshold = _BAYER_4x4[pixel_ys % 4, pixel_xs % 4]
    bayer_pattern = bayer_threshold < dither_intensity

    # Isometric diamond edges from both constraints
    above_hotspot = pixel_ys < center_y
    dx = (pixel_xs - center_x).astype(np.float64)
    # Top edge: max of the two lower bounds
    top_a = -margin_u - dx * 0.5  # from |dx/2 + dy| <= margin_u
    top_b = -margin_v + dx * 0.5  # from |-dx/2 + dy| <= margin_v
    foundation_y = center_y + np.maximum(top_a, top_b)

    # Bottom edge of the isometric diamond
    bot_a = margin_u - dx * 0.5   # from |dx/2 + dy| <= margin_u
    bot_b = margin_v + dx * 0.5   # from |-dx/2 + dy| <= margin_v
    bottom_y = center_y + np.minimum(bot_a, bot_b)
    below_foundation = pixel_ys >= bottom_y

    # Gradient zone: sparser dithering near the foundation line
    if gradient_height > 0:
        full_zone = pixel_ys < (foundation_y - gradient_height)
        in_gradient = ~full_zone & (pixel_ys < foundation_y)
        # Sparse pattern uses half the main intensity
        sparse_intensity = max(1, dither_intensity // 2)
        sparse_pattern = bayer_threshold < sparse_intensity
        above_dither = above_hotspot & (
            (full_zone & bayer_pattern) |
            (in_gradient & sparse_pattern)
        )
        if dither_bottom:
            dither = above_dither | (below_foundation & bayer_pattern)
        else:
            dither = above_dither
    else:
        above_foundation = pixel_ys < foundation_y
        above_dither = above_hotspot & above_foundation & bayer_pattern
        if dither_bottom:
            dither = above_dither | (below_foundation & bayer_pattern)
        else:
            dither = above_dither

    # Pack boolean array into 16-bit masks
    return (dither.astype(np.uint32) * _BIT_VALUES[None, :]).sum(axis=1)


def compute_outline_masks(block_xs, block_ys, center_x, center_y,
                          margin_u, margin_v, thickness=1):
    """
    Vectorized computation of 16-bit outline masks for N blocks at once.

    Uses the same isometric diamond as compute_dither_masks:
      top_y = center_y + max(-margin_u - dx/2, -margin_v + dx/2)
      bottom_y = center_y + min(margin_u - dx/2, margin_v + dx/2)
    """
    # Expand to per-pixel coordinates: (N, 16)
    pixel_ys = block_ys[:, None] + _PIX_ROWS[None, :]
    pixel_xs = block_xs[:, None] + _PIX_COLS[None, :]

    dx = (pixel_xs - center_x).astype(np.float64)

    # Top edge: max of both lower bounds
    top_y = center_y + np.maximum(-margin_u - dx * 0.5, -margin_v + dx * 0.5)
    on_top = (pixel_ys < center_y) & (pixel_ys >= (top_y - thickness)) & (pixel_ys < top_y)

    # Bottom edge: min of both upper bounds
    bottom_y = center_y + np.minimum(margin_u - dx * 0.5, margin_v + dx * 0.5)
    on_bottom = (pixel_ys >= center_y) & (pixel_ys >= (bottom_y - thickness)) & (pixel_ys < bottom_y)

    # Pack combined boolean array into 16-bit masks
    on_outline = on_top | on_bottom
    return (on_outline.astype(np.uint32) * _BIT_VALUES[None, :]).sum(axis=1)


def compute_animation_protection(sld, threshold=0.5):
    """Compute per-block protection masks for animated blocks using delta encoding.

    SLD animated sprites use delta encoding: frame 0 is a full frame, and
    subsequent frames have flag1 bit 7 set, meaning "skip" commands copy from
    the previous frame instead of being transparent. Only the CHANGED blocks
    have draw commands in delta frames.

    This function counts how often each block position appears as a draw block
    across delta frames. Positions drawn frequently are continuously animated
    (e.g. mill sails) and should be kept opaque.

    Args:
        sld: Parsed SLD object with multiple frames
        threshold: Fraction of delta frames a block must be drawn in to be protected

    Returns:
        dict: (block_x, block_y) -> 0xFFFF protection mask for animated blocks
    """
    if sld.num_frames <= 1:
        return {}

    # Count how often each position appears as a draw block in delta frames
    draw_counts = {}
    num_delta = 0
    for frame in sld.frames[1:]:
        layer = get_layer(frame, LAYER_MAIN)
        if layer is None or not (layer.flag1 & 0x80):
            continue
        num_delta += 1
        positions = get_block_positions(layer, frame)
        for _, bx, by in positions:
            key = (bx, by)
            draw_counts[key] = draw_counts.get(key, 0) + 1

    if num_delta == 0:
        return {}

    # Protect positions drawn in >= threshold fraction of delta frames
    min_draws = max(1, int(num_delta * threshold))
    return {key: 0xFFFF for key, count in draw_counts.items()
            if count >= min_draws}


def _dxt1_opaque_mask(block_data):
    """Get a 4x4 boolean array of opaque pixels from a DXT1 block.

    In DXT1, if color0 <= color1 (transparent mode), index 3 = transparent.
    Otherwise all pixels are opaque.
    """
    c0 = block_data[0] | (block_data[1] << 8)
    c1 = block_data[2] | (block_data[3] << 8)
    if c0 > c1:
        # Opaque mode: all pixels are opaque
        return np.ones((4, 4), dtype=bool)
    # Transparent mode: index 3 = transparent
    indices = int.from_bytes(block_data[4:8], 'little')
    opaque = np.ones(16, dtype=bool)
    for i in range(16):
        if ((indices >> (2 * i)) & 3) == 3:
            opaque[i] = False
    return opaque.reshape(4, 4)


def compute_edge_protection(positions, edge_inset, blocks=None):
    """
    Compute per-block 16-bit masks of pixels within edge_inset of the silhouette edge.

    Uses pixel-level transparency from DXT1 blocks (if provided) to accurately
    detect the building silhouette, then erodes to find edge pixels.

    Args:
        positions: list of (block_idx, block_x, block_y) from get_block_positions
        edge_inset: pixels from building edge to keep opaque
        blocks: optional list of DXT1 block data (bytes) for pixel-level accuracy

    Returns:
        dict: (block_x, block_y) -> uint16 protection mask (bit i = protect pixel i)
    """
    if edge_inset <= 0 or not positions:
        return {}

    # Gather block positions
    all_bx = [bx for _, bx, _ in positions]
    all_by = [by for _, _, by in positions]
    min_bx, max_bx = min(all_bx), max(all_bx)
    min_by, max_by = min(all_by), max(all_by)

    # Build pixel-level boolean grid with padding for erosion
    pad = edge_inset
    pixel_w = (max_bx - min_bx) + 4 + 2 * pad
    pixel_h = (max_by - min_by) + 4 + 2 * pad

    drawn = np.zeros((pixel_h, pixel_w), dtype=bool)
    for bi, bx, by in positions:
        px = (bx - min_bx) + pad
        py = (by - min_by) + pad
        if blocks and bi < len(blocks) and len(blocks[bi]) >= 8:
            # Use actual DXT1 pixel transparency
            drawn[py:py+4, px:px+4] = _dxt1_opaque_mask(blocks[bi])
        else:
            # Fallback: treat entire block as drawn
            drawn[py:py+4, px:px+4] = True

    # Find boundary pixels: drawn pixels within edge_inset of the silhouette edge.
    # We erode the drawn mask by edge_inset and take the difference.
    eroded = drawn.copy()
    for _ in range(edge_inset):
        inner = np.zeros_like(eroded)
        inner[1:-1, 1:-1] = (eroded[:-2, 1:-1] & eroded[2:, 1:-1] &
                              eroded[1:-1, :-2] & eroded[1:-1, 2:])
        eroded = inner
    boundary = drawn & ~eroded

    # Extract per-block 16-bit protection masks
    protection = {}
    for _, bx, by in positions:
        px = (bx - min_bx) + pad
        py = (by - min_by) + pad
        block = boundary[py:py+4, px:px+4].ravel()
        if block.any():
            mask = int((block.astype(np.uint32) * _BIT_VALUES).sum())
            protection[(bx, by)] = mask

    return protection


def compute_outer_contour(positions, contour_width, blocks=None):
    """
    Compute per-block 16-bit masks of pixels OUTSIDE the silhouette within
    contour_width pixels of the edge.  Returns masks for both existing and
    new (empty) block positions that need contour pixels.

    Args:
        positions: list of (block_idx, block_x, block_y) from get_block_positions
        contour_width: pixels outward from silhouette edge
        blocks: optional list of DXT1 block data for pixel-level accuracy

    Returns:
        dict: (block_x, block_y) -> uint16 contour mask
    """
    if contour_width <= 0 or not positions:
        return {}

    all_bx = [bx for _, bx, _ in positions]
    all_by = [by for _, _, by in positions]
    min_bx, max_bx = min(all_bx), max(all_bx)
    min_by, max_by = min(all_by), max(all_by)

    # Pad to accommodate outward expansion
    pad = contour_width + 4  # extra block width for neighbor blocks
    pixel_w = (max_bx - min_bx) + 4 + 2 * pad
    pixel_h = (max_by - min_by) + 4 + 2 * pad

    drawn = np.zeros((pixel_h, pixel_w), dtype=bool)
    for bi, bx, by in positions:
        px = (bx - min_bx) + pad
        py = (by - min_by) + pad
        if blocks and bi < len(blocks) and len(blocks[bi]) >= 8:
            drawn[py:py+4, px:px+4] = _dxt1_opaque_mask(blocks[bi])
        else:
            drawn[py:py+4, px:px+4] = True

    # Dilate outward by contour_width
    dilated = drawn.copy()
    for _ in range(contour_width):
        expanded = np.zeros_like(dilated)
        expanded[1:, :] |= dilated[:-1, :]
        expanded[:-1, :] |= dilated[1:, :]
        expanded[:, 1:] |= dilated[:, :-1]
        expanded[:, :-1] |= dilated[:, 1:]
        dilated = expanded | dilated

    # Contour = dilated ring outside the original silhouette
    contour = dilated & ~drawn

    # Extract per-block masks — scan all block positions that could have contour pixels
    # (existing blocks + neighbors within contour_width)
    block_min_bx = min_bx - ((contour_width + 3) // 4) * 4
    block_max_bx = max_bx + ((contour_width + 3) // 4) * 4
    block_min_by = min_by - ((contour_width + 3) // 4) * 4
    block_max_by = max_by + ((contour_width + 3) // 4) * 4

    result = {}
    bx = block_min_bx
    while bx <= block_max_bx + 4:
        by = block_min_by
        while by <= block_max_by + 4:
            px = (bx - min_bx) + pad
            py = (by - min_by) + pad
            if 0 <= px and px + 4 <= pixel_w and 0 <= py and py + 4 <= pixel_h:
                block = contour[py:py+4, px:px+4].ravel()
                if block.any():
                    mask = int((block.astype(np.uint32) * _BIT_VALUES).sum())
                    result[(bx, by)] = mask
            by += 4
        bx += 4

    return result


def inject_bc4_outline(block_data, mask_bits, value=200):
    """
    Set specific pixels in a BC4 block to a high value for team-color outline.

    Args:
        block_data: bytes, 8-byte BC4 block
        mask_bits: int, 16-bit bitmask where bit i = set this pixel to value
        value: int, brightness value (0-255) for outline pixels

    Returns:
        bytes: modified 8-byte BC4 block
    """
    if mask_bits == 0:
        return block_data

    values = decode_bc4_block(block_data)
    flat = values.flatten()

    # Set masked pixels to the outline value
    m = mask_bits
    while m:
        i = (m & -m).bit_length() - 1
        flat[i] = value
        m &= m - 1

    return encode_bc4_block_from_flat(flat)




def widen_layer_bounds(main_layer, frame, new_x1, new_x2):
    """Expand main layer X bounds, remapping all layer commands to the new grid.

    When offset_x1 or offset_x2 change, blocks_per_row changes, so existing
    grid cursor positions need remapping to preserve their pixel coordinates.

    This also remaps damage and playercolor layers which share main's grid.
    """
    old_x1 = main_layer.offset_x1
    old_x2 = main_layer.offset_x2
    old_w = old_x2 - old_x1
    old_bpr = (old_w + 3) // 4

    new_w = new_x2 - new_x1
    new_bpr = (new_w + 3) // 4

    if old_bpr == new_bpr and old_x1 == new_x1:
        # No grid change needed, just update bounds
        main_layer.offset_x1 = new_x1
        main_layer.offset_x2 = new_x2
        return

    # Remap all layers that use main's grid
    layers_to_remap = []
    for layer in frame.layers:
        if layer.layer_type in (LAYER_MAIN, LAYER_DAMAGE, LAYER_PLAYERCOLOR):
            layers_to_remap.append(layer)

    for layer in layers_to_remap:
        if not layer.commands:
            continue

        # Decode existing blocks to pixel positions
        old_positions = []  # (old_cursor, block_idx)
        block_idx = 0
        cursor = 0
        for skip, draw in layer.commands:
            cursor += skip
            for _ in range(draw):
                old_positions.append((cursor, block_idx))
                block_idx += 1
                cursor += 1

        # Convert old cursor -> pixel -> new cursor
        new_grid_blocks = {}  # new_cursor -> block_data
        for old_cursor, bi in old_positions:
            old_row = old_cursor // old_bpr
            old_col = old_cursor % old_bpr
            # Pixel position
            px = old_x1 + old_col * 4
            py = main_layer.offset_y1 + old_row * 4
            # New grid position
            new_col = (px - new_x1) // 4
            new_row = (py - main_layer.offset_y1) // 4
            new_cursor = new_row * new_bpr + new_col
            new_grid_blocks[new_cursor] = layer.blocks[bi]

        # Rebuild commands from new grid positions
        sorted_positions = sorted(new_grid_blocks.keys())
        if not sorted_positions:
            continue

        new_commands = []
        new_blocks = []
        prev_end = 0
        i = 0

        while i < len(sorted_positions):
            pos = sorted_positions[i]
            skip = pos - prev_end

            while skip > 255:
                new_commands.append((255, 0))
                skip -= 255

            # Find consecutive run
            run_start = i
            while (i < len(sorted_positions) - 1 and
                   sorted_positions[i + 1] == sorted_positions[i] + 1):
                i += 1
            run_len = i - run_start + 1

            drawn = 0
            while drawn < run_len:
                chunk = min(run_len - drawn, 255)
                new_commands.append((skip if drawn == 0 else 0, chunk))
                for j in range(run_start + drawn, run_start + drawn + chunk):
                    new_blocks.append(new_grid_blocks[sorted_positions[j]])
                drawn += chunk

            prev_end = sorted_positions[i] + 1
            i += 1

        layer.commands = new_commands
        layer.command_count = len(new_commands)
        layer.blocks = new_blocks

    # Update bounds
    main_layer.offset_x1 = new_x1
    main_layer.offset_x2 = new_x2


def ensure_layer_blocks(layer, frame, needed_grid_positions, default_block=None):
    """
    Ensure a layer has blocks at the given grid positions.

    Rewrites the layer's skip/draw commands and inserts default blocks
    at any needed positions that don't already have data.

    Args:
        layer: SLDLayer
        frame: SLDFrame (to get main layer bounds)
        needed_grid_positions: set of linear grid indices where blocks are needed
        default_block: 8-byte block data for new positions (default: all zeros)
    """
    if default_block is None:
        default_block = _ZERO_BLOCK

    main = get_layer(frame, LAYER_MAIN)
    if not main:
        return
    layer_w = main.offset_x2 - main.offset_x1
    blocks_per_row = (layer_w + 3) // 4

    # Build map: grid_position -> existing block index
    existing = {}
    block_idx = 0
    cursor = 0
    for skip, draw in layer.commands:
        cursor += skip
        for _ in range(draw):
            existing[cursor] = block_idx
            block_idx += 1
            cursor += 1

    # Merge needed positions into the drawn set
    all_drawn = set(existing.keys()) | needed_grid_positions
    sorted_positions = sorted(all_drawn)

    if not sorted_positions:
        return

    # Walk sorted positions and build (skip, draw) runs
    # Skip and draw values must fit in a byte (0-255)
    new_commands = []
    new_blocks = []
    i = 0
    prev_end = 0

    while i < len(sorted_positions):
        pos = sorted_positions[i]
        skip = pos - prev_end

        # Emit skip-only commands for large gaps (skip > 255)
        while skip > 255:
            new_commands.append((255, 0))
            skip -= 255

        # Find consecutive run starting at pos
        run_start = i
        while i < len(sorted_positions) - 1 and sorted_positions[i + 1] == sorted_positions[i] + 1:
            i += 1
        run_len = i - run_start + 1

        # Split long draw runs (draw > 255)
        drawn = 0
        while drawn < run_len:
            chunk = min(run_len - drawn, 255)
            new_commands.append((skip if drawn == 0 else 0, chunk))
            for j in range(run_start + drawn, run_start + drawn + chunk):
                gp = sorted_positions[j]
                if gp in existing:
                    new_blocks.append(layer.blocks[existing[gp]])
                else:
                    new_blocks.append(default_block)
            drawn += chunk

        prev_end = sorted_positions[i] + 1
        i += 1

    layer.commands = new_commands
    layer.command_count = len(new_commands)
    layer.blocks = new_blocks


def add_foundation_fill(frame, main_layer, margin, outline_value):
    """
    Add dithered team-color fill to empty areas inside the foundation diamond.

    Creates checkerboard DXT1 blocks in the main layer (half transparent, half white)
    and matching playercolor BC4 blocks so the opaque pixels show team color.
    """
    cx = frame.center_x
    cy = frame.center_y
    pc_layer = get_layer(frame, LAYER_PLAYERCOLOR)
    if pc_layer is None:
        return

    # Grid dimensions (from main layer)
    layer_w = main_layer.offset_x2 - main_layer.offset_x1
    layer_h = main_layer.offset_y2 - main_layer.offset_y1
    base_x = main_layer.offset_x1
    base_y = main_layer.offset_y1
    blocks_per_row = (layer_w + 3) // 4
    blocks_per_col = (layer_h + 3) // 4

    # Existing drawn positions
    drawn = set()
    for _, bx, by in get_block_positions(main_layer, frame):
        drawn.add((bx, by))

    # Find empty positions inside the foundation diamond
    fill_grid_positions = set()
    for row in range(blocks_per_col):
        for col in range(blocks_per_row):
            bx = base_x + col * 4
            by = base_y + row * 4
            if (bx, by) in drawn:
                continue
            # Check if block center is inside the diamond
            cpx = bx + 2
            cpy = by + 2
            dx = abs(cpx - cx)
            top_y = (cy - margin) + dx * 0.5
            bot_y = (cy + margin) - dx * 0.5
            if cpy >= top_y and cpy <= bot_y:
                fill_grid_positions.add(row * blocks_per_row + col)

    if not fill_grid_positions:
        return

    # Determine checkerboard phase from layer origin
    # All blocks share the same internal pattern since block coords are 4-aligned
    phase = (base_x + base_y) % 2

    # DXT1 checkerboard: white color, transparent mode (color0 == color1)
    # Phase 0: even pixels (col+row)%2==0 are opaque (index 0), odd are transparent (index 3)
    # Phase 1: flipped
    if phase == 0:
        dxt1_fill = b'\xFF\xFF\xFF\xFF\xCC\x33\xCC\x33'
    else:
        dxt1_fill = b'\xFF\xFF\xFF\xFF\x33\xCC\x33\xCC'

    # BC4 checkerboard: team color at opaque pixels, zero at transparent
    flat = np.zeros(16, dtype=np.uint8)
    for i in range(16):
        if ((i % 4 + i // 4) % 2 == 0) != (phase == 1):
            flat[i] = outline_value
    bc4_fill = encode_bc4_block_from_flat(flat)

    # Insert blocks into both layers
    ensure_layer_blocks(main_layer, frame, fill_grid_positions, dxt1_fill)
    ensure_layer_blocks(pc_layer, frame, fill_grid_positions, bc4_fill)


def process_frame(frame, tile_hh, tiles, outline_value=200,
                  edge_inset=3, gradient_height=0, outline_thickness=4,
                  outline_enabled=True, animation_protection=None,
                  full_positions=None, dither_intensity=8,
                  dither_bottom=False, contour_width=0,
                  contour_color='team', compound_offsets=None):
    """
    Process a single SLD frame, applying dithering to layers.

    Args:
        frame: SLDFrame object
        tile_hh: tile half-height in pixels (24 for x1, 48 for x2)
        tiles: (tiles_u, tiles_v) footprint tuple
        outline_value: brightness for foundation outline (0-255)
        edge_inset: pixels from building edge to keep opaque
        gradient_height: transition zone height above foundation line
        outline_thickness: outline band height in pixels
        outline_enabled: whether to draw foundation outline
        animation_protection: dict of (bx, by) -> uint16 mask for animated pixels to keep opaque
        full_positions: complete block positions from frame 0 (for stable edge
            protection in delta frames). If None, uses the current frame's positions.
    """
    cx = frame.center_x
    cy = frame.center_y

    # Get main graphic layer
    main_layer = get_layer(frame, LAYER_MAIN)
    if main_layer is None:
        return

    tiles_u, tiles_v = tiles
    margin_u = tiles_u * tile_hh
    margin_v = tiles_v * tile_hh

    # Calculate block positions for main layer
    main_positions = get_block_positions(main_layer, frame)
    if not main_positions:
        return

    # Vectorized mask computation for all main-layer blocks
    pos_array = np.array(main_positions, dtype=np.int32)
    block_idxs = pos_array[:, 0]
    block_xs = pos_array[:, 1]
    block_ys = pos_array[:, 2]

    if compound_offsets:
        masks = compute_compound_dither_masks(block_xs, block_ys, cx, cy,
                                              tile_hh, compound_offsets,
                                              gradient_height, dither_intensity,
                                              dither_bottom)
    else:
        masks = compute_dither_masks(block_xs, block_ys, cx, cy,
                                     margin_u, margin_v, gradient_height,
                                     dither_intensity, dither_bottom)

    if outline_enabled:
        if compound_offsets:
            outline_masks = compute_compound_outline_masks(
                block_xs, block_ys, cx, cy, tile_hh, compound_offsets,
                outline_thickness)
        else:
            outline_masks = compute_outline_masks(
                block_xs, block_ys, cx, cy, margin_u, margin_v, outline_thickness)
    else:
        outline_masks = np.zeros(len(main_positions), dtype=np.uint32)

    # Edge protection: keep outermost pixels opaque based on silhouette boundary
    # For delta frames, merge frame 0's positions with current frame's draw
    # blocks so edge protection follows both the static body AND animated parts
    edge_prot_cache = {}
    if edge_inset > 0:
        if full_positions and full_positions is not main_positions:
            # Merge: frame 0 silhouette + current frame's draw blocks
            full_set = {(bx, by) for _, bx, by in full_positions}
            merged = list(full_positions)
            for pos in main_positions:
                if (pos[1], pos[2]) not in full_set:
                    merged.append(pos)
            edge_positions = merged
        else:
            edge_positions = main_positions
        edge_prot_cache = compute_edge_protection(edge_positions, edge_inset, main_layer.blocks)
        if edge_prot_cache:
            prot_array = np.array(
                [edge_prot_cache.get((int(block_xs[i]), int(block_ys[i])), 0)
                 for i in range(len(main_positions))], dtype=np.uint32)
            masks &= ~prot_array

    # Animation protection: keep changing pixels opaque across animation frames
    if animation_protection:
        anim_array = np.array(
            [animation_protection.get((int(block_xs[i]), int(block_ys[i])), 0)
             for i in range(len(main_positions))], dtype=np.uint32)
        masks &= ~anim_array

    # Keep main layer opaque at outline pixels so team color shows through
    if outline_enabled:
        masks &= ~outline_masks

    # Contour: compute outer contour (pixels outside the silhouette)
    contour_pos_cache = {}
    contour_cache = {}
    if contour_width > 0:
        contour_cache = compute_outer_contour(main_positions, contour_width, main_layer.blocks) or {}

    # Build lookup dicts
    block_dither_masks = {}   # block_idx -> mask_bits (for main layer)
    pos_mask_cache = {}       # (block_x, block_y) -> dither mask_bits (for all layers)
    outline_pos_cache = {}    # (block_x, block_y) -> outline mask_bits
    any_dithered = False

    for i in range(len(main_positions)):
        m = int(masks[i])
        om = int(outline_masks[i])
        idx = int(block_idxs[i])
        key = (int(block_xs[i]), int(block_ys[i]))
        pos_mask_cache[key] = m
        outline_pos_cache[key] = om
        if m:
            block_dither_masks[idx] = m
            any_dithered = True

    if not any_dithered and not contour_cache:
        return

    # Apply dithering to main graphic layer (DXT1)
    if any_dithered:
        _apply_dxt1_masks_batch(main_layer, block_dither_masks)

    if outline_enabled:
        # Force outline pixels to opaque in main layer DXT1 blocks.
        block_outline_masks = {}
        for i in range(len(main_positions)):
            om = int(outline_masks[i])
            if om:
                block_outline_masks[int(block_idxs[i])] = om
        _force_opaque_dxt1_batch(main_layer, block_outline_masks)

    # Contour: create outer contour blocks (like outline, but around silhouette)
    if contour_cache:
        # Apply contour to existing blocks that overlap
        existing_pos_set = {(int(block_xs[i]), int(block_ys[i])): int(block_idxs[i])
                            for i in range(len(main_positions))}
        block_contour_masks = {}
        for (bx, by), cm in contour_cache.items():
            bi = existing_pos_set.get((bx, by))
            if bi is not None:
                block_contour_masks[bi] = cm
                contour_pos_cache[(bx, by)] = cm
        if block_contour_masks:
            _force_opaque_dxt1_batch(main_layer, block_contour_masks)

    # Create new outline blocks only for full frames (not deltas).
    # Delta frames inherit outline blocks from frame 0 via skip commands.
    # Adding blocks to delta frames would corrupt the delta encoding.
    is_delta = main_layer.flag1 & 0x80

    if outline_enabled and not is_delta:
        # Create playercolor layer if frame doesn't have one (e.g. palisade)
        # so outline can show team color
        if get_layer(frame, LAYER_PLAYERCOLOR) is None and any(om for om in outline_pos_cache.values()):
            from sld import SLDLayer
            pc_layer = SLDLayer()
            pc_layer.layer_type = LAYER_PLAYERCOLOR
            pc_layer.flag1 = 0
            pc_layer.unknown1 = 0
            pc_layer.commands = []
            pc_layer.command_count = 0
            pc_layer.blocks = []
            frame.frame_type |= LAYER_PLAYERCOLOR
            frame.layers.append(pc_layer)

        # --- Create new blocks at outline positions missing from the sprite ---
        # The foundation diamond may extend beyond drawn blocks (below the sprite,
        # at side corners, or in gaps). We need DXT1+BC4 blocks there so the
        # outline renders continuously around the entire diamond.

        # Extend horizontal bounds if diamond left/right corners exceed them
        if compound_offsets:
            # Compound: union of 1x1 diamonds at offset positions
            M = tile_hh
            diamond_left = min(cx + ox - 2 * M for ox, oy in compound_offsets)
            diamond_right = max(cx + ox + 2 * M for ox, oy in compound_offsets)
            bottom_edge_y = max(cy + oy + M for ox, oy in compound_offsets)
        else:
            diamond_left = cx - (margin_u + margin_v)
            diamond_right = cx + (margin_u + margin_v)
            bottom_edge_y = cy + max(margin_u, margin_v)
        new_x1 = min(main_layer.offset_x1, (diamond_left // 4) * 4)
        new_x2 = max(main_layer.offset_x2, ((diamond_right + 3) // 4) * 4)
        if new_x1 < main_layer.offset_x1 or new_x2 > main_layer.offset_x2:
            widen_layer_bounds(main_layer, frame, new_x1, new_x2)
            # Re-read positions after grid remap
            main_positions = get_block_positions(main_layer, frame)

        # Extend vertical bounds if bottom diamond edge exceeds them
        if bottom_edge_y > main_layer.offset_y2:
            main_layer.offset_y2 = ((bottom_edge_y + 3) // 4) * 4

        # Collect existing drawn block positions
        existing_pos = {(bx, by) for _, bx, by in main_positions}

        # Scan full grid for empty positions that need outline blocks
        layer_w = main_layer.offset_x2 - main_layer.offset_x1
        layer_h = main_layer.offset_y2 - main_layer.offset_y1
        bpr = (layer_w + 3) // 4
        bpc = (layer_h + 3) // 4
        base_x, base_y = main_layer.offset_x1, main_layer.offset_y1

        cand_bxs, cand_bys, cand_gps = [], [], []
        for row in range(bpc):
            for col in range(bpr):
                bx = base_x + col * 4
                by = base_y + row * 4
                if (bx, by) not in existing_pos:
                    cand_bxs.append(bx)
                    cand_bys.append(by)
                    cand_gps.append(row * bpr + col)

        if cand_bxs:
            arr_bxs = np.array(cand_bxs, dtype=np.int32)
            arr_bys = np.array(cand_bys, dtype=np.int32)
            if compound_offsets:
                cand_outlines = compute_compound_outline_masks(
                    arr_bxs, arr_bys, cx, cy, tile_hh, compound_offsets,
                    outline_thickness)
            else:
                cand_outlines = compute_outline_masks(
                    arr_bxs, arr_bys, cx, cy, margin_u, margin_v, outline_thickness)

            needed_gps = set()
            new_outline_info = []  # (bx, by, outline_mask)
            for i in range(len(cand_bxs)):
                om = int(cand_outlines[i])
                if om:
                    needed_gps.add(cand_gps[i])
                    new_outline_info.append((cand_bxs[i], cand_bys[i], om))
                    # Register dither=0 so the loop doesn't recompute masks
                    pos_mask_cache[(cand_bxs[i], cand_bys[i])] = 0

            if needed_gps:
                # Add placeholder blocks to main layer
                ensure_layer_blocks(main_layer, frame, needed_gps)

                # Patch new DXT1 blocks: gray base in transparent-mode.
                # c0=c1=0x52AA (RGB 82,85,82) matches the flat buildings mod's
                # proven gray that composites well with playercolor tinting.
                # Outline pixels = index 0 (opaque gray), rest = index 3 (transparent).
                new_positions = get_block_positions(main_layer, frame)
                pos_to_bi = {(bx, by): bi for bi, bx, by in new_positions}
                for bx, by, om in new_outline_info:
                    bi = pos_to_bi.get((bx, by))
                    if bi is not None:
                        non_outline = (~om) & 0xFFFF
                        tb = int(_MASK_EXPAND[non_outline])
                        main_layer.blocks[bi] = bytes([
                            0xAA, 0x52, 0xAA, 0x52,
                            tb & 0xFF, (tb >> 8) & 0xFF,
                            (tb >> 16) & 0xFF, (tb >> 24) & 0xFF])

                # Directly patch PC blocks for new outline positions
                pc_new = get_layer(frame, LAYER_PLAYERCOLOR)
                if pc_new is not None:
                    ensure_layer_blocks(pc_new, frame, needed_gps)
                    pc_positions = get_block_positions(pc_new, frame)
                    pc_pos_to_bi = {(bx, by): bi for bi, bx, by in pc_positions}
                    for bx, by, om in new_outline_info:
                        bi = pc_pos_to_bi.get((bx, by))
                        if bi is not None:
                            pc_new.blocks[bi] = inject_bc4_outline(
                                pc_new.blocks[bi], om, 255)

        # Ensure playercolor layer has blocks at all outline positions
        pc_layer = get_layer(frame, LAYER_PLAYERCOLOR)
        if pc_layer is not None:
            layer_w = main_layer.offset_x2 - main_layer.offset_x1
            blocks_per_row = (layer_w + 3) // 4
            base_x = main_layer.offset_x1
            base_y = main_layer.offset_y1

            needed = set()
            for (bx, by), om in outline_pos_cache.items():
                if om:
                    col = (bx - base_x) // 4
                    row = (by - base_y) // 4
                    needed.add(row * blocks_per_row + col)

            if needed:
                ensure_layer_blocks(pc_layer, frame, needed)

    # Create outer contour blocks at positions outside the silhouette
    if contour_cache and not is_delta:
        # Widen layer bounds to fit contour positions
        contour_bxs = [bx for bx, _ in contour_cache]
        contour_bys = [by for _, by in contour_cache]
        contour_x1 = min(contour_bxs)
        contour_x2 = max(contour_bxs) + 4
        contour_y2 = max(contour_bys) + 4

        new_x1 = min(main_layer.offset_x1, (contour_x1 // 4) * 4)
        new_x2 = max(main_layer.offset_x2, ((contour_x2 + 3) // 4) * 4)
        if new_x1 < main_layer.offset_x1 or new_x2 > main_layer.offset_x2:
            widen_layer_bounds(main_layer, frame, new_x1, new_x2)

        if contour_y2 > main_layer.offset_y2:
            main_layer.offset_y2 = ((contour_y2 + 3) // 4) * 4
        contour_y1 = min(contour_bys)
        if contour_y1 < main_layer.offset_y1:
            main_layer.offset_y1 = (contour_y1 // 4) * 4

        # Determine which contour positions need new blocks
        current_positions = get_block_positions(main_layer, frame)
        existing_pos = {(bx, by) for _, bx, by in current_positions}

        layer_w = main_layer.offset_x2 - main_layer.offset_x1
        blocks_per_row = (layer_w + 3) // 4
        base_x = main_layer.offset_x1
        base_y = main_layer.offset_y1

        needed_gps = set()
        new_contour_info = []
        for (bx, by), cm in contour_cache.items():
            if (bx, by) not in existing_pos:
                col = (bx - base_x) // 4
                row = (by - base_y) // 4
                needed_gps.add(row * blocks_per_row + col)
                new_contour_info.append((bx, by, cm))
            contour_pos_cache[(bx, by)] = cm

        if needed_gps:
            ensure_layer_blocks(main_layer, frame, needed_gps)

        # Patch DXT1 blocks for contour
        all_positions = get_block_positions(main_layer, frame)
        pos_to_bi = {(bx, by): bi for bi, bx, by in all_positions}

        # New blocks: opaque gray at contour pixels, transparent elsewhere
        new_contour_set = {(bx, by) for bx, by, _ in new_contour_info}
        for bx, by, cm in new_contour_info:
            bi = pos_to_bi.get((bx, by))
            if bi is not None:
                non_contour = (~cm) & 0xFFFF
                tb = int(_MASK_EXPAND[non_contour])
                main_layer.blocks[bi] = bytes([
                    0xAA, 0x52, 0xAA, 0x52,
                    tb & 0xFF, (tb >> 8) & 0xFF,
                    (tb >> 16) & 0xFF, (tb >> 24) & 0xFF])

        # Existing blocks: force contour pixels opaque
        existing_contour_masks = {}
        for (bx, by), cm in contour_cache.items():
            if (bx, by) not in new_contour_set:
                bi = pos_to_bi.get((bx, by))
                if bi is not None:
                    existing_contour_masks[bi] = cm
        if existing_contour_masks:
            _force_opaque_dxt1_batch(main_layer, existing_contour_masks)

        # Create playercolor layer if needed
        if get_layer(frame, LAYER_PLAYERCOLOR) is None:
            from sld import SLDLayer
            pc_layer = SLDLayer()
            pc_layer.layer_type = LAYER_PLAYERCOLOR
            pc_layer.flag1 = 0
            pc_layer.unknown1 = 0
            pc_layer.commands = []
            pc_layer.command_count = 0
            pc_layer.blocks = []
            frame.frame_type |= LAYER_PLAYERCOLOR
            frame.layers.append(pc_layer)

        # Set playercolor for contour pixels
        pc_layer = get_layer(frame, LAYER_PLAYERCOLOR)
        if pc_layer is not None:
            ensure_layer_blocks(pc_layer, frame, {
                ((by - base_y) // 4) * blocks_per_row + ((bx - base_x) // 4)
                for (bx, by) in contour_cache
            })
            pc_positions = get_block_positions(pc_layer, frame)
            pc_pos_to_bi = {(bx, by): bi for bi, bx, by in pc_positions}
            contour_pc_value = 255 if contour_color == 'team' else 0
            for (bx, by), cm in contour_cache.items():
                bi = pc_pos_to_bi.get((bx, by))
                if bi is not None:
                    if contour_color == 'team':
                        pc_layer.blocks[bi] = inject_bc4_outline(
                            pc_layer.blocks[bi], cm, contour_pc_value)
                    else:
                        pc_layer.blocks[bi] = zero_bc4_pixels(
                            pc_layer.blocks[bi], cm)

    # Apply to other layers, reusing cached masks by position
    for layer_type in (LAYER_SHADOW, LAYER_PLAYERCOLOR, LAYER_DAMAGE):
        layer = get_layer(frame, layer_type)
        if layer is None:
            continue

        positions = get_block_positions(layer, frame)
        is_dxt1 = layer_type in DXT1_LAYERS

        # Split into cached hits and uncached positions
        layer_masks = {}
        layer_outline_masks = {}  # only for player color
        uncached = []

        for block_idx, block_x, block_y in positions:
            key = (block_x, block_y)
            cached = pos_mask_cache.get(key)
            if cached is not None:
                if cached:
                    layer_masks[block_idx] = cached
                if layer_type == LAYER_PLAYERCOLOR:
                    om = outline_pos_cache.get(key, 0)
                    if om:
                        layer_outline_masks[block_idx] = om
            else:
                uncached.append((block_idx, block_x, block_y))

        # Batch-compute any uncached masks
        if uncached:
            u_array = np.array(uncached, dtype=np.int32)
            if compound_offsets:
                new_masks = compute_compound_dither_masks(
                    u_array[:, 1], u_array[:, 2], cx, cy, tile_hh,
                    compound_offsets, gradient_height)
            else:
                new_masks = compute_dither_masks(u_array[:, 1], u_array[:, 2],
                                                 cx, cy, margin_u, margin_v,
                                                 gradient_height)
            # Apply edge protection to uncached masks
            if edge_prot_cache:
                u_prot = np.array(
                    [edge_prot_cache.get((int(u_array[i, 1]), int(u_array[i, 2])), 0)
                     for i in range(len(uncached))], dtype=np.uint32)
                new_masks &= ~u_prot
            # Apply animation protection to uncached masks
            if animation_protection:
                u_anim = np.array(
                    [animation_protection.get((int(u_array[i, 1]), int(u_array[i, 2])), 0)
                     for i in range(len(uncached))], dtype=np.uint32)
                new_masks &= ~u_anim
            if layer_type == LAYER_PLAYERCOLOR:
                if compound_offsets:
                    new_outlines = compute_compound_outline_masks(
                        u_array[:, 1], u_array[:, 2], cx, cy, tile_hh,
                        compound_offsets, outline_thickness)
                else:
                    new_outlines = compute_outline_masks(
                        u_array[:, 1], u_array[:, 2], cx, cy,
                        margin_u, margin_v, outline_thickness)
                new_masks &= ~new_outlines
            for i in range(len(uncached)):
                m = int(new_masks[i])
                pos_mask_cache[(uncached[i][1], uncached[i][2])] = m
                if m:
                    layer_masks[uncached[i][0]] = m
                if layer_type == LAYER_PLAYERCOLOR:
                    om = int(new_outlines[i])
                    outline_pos_cache[(uncached[i][1], uncached[i][2])] = om
                    if om:
                        layer_outline_masks[uncached[i][0]] = om

        # Apply masks (batch for DXT1, per-block for BC4)
        if is_dxt1:
            _apply_dxt1_masks_batch(layer, layer_masks)
        elif layer_type == LAYER_PLAYERCOLOR:
            # Player color: zero dithered pixels, then inject outline + contour
            for block_idx, mask in layer_masks.items():
                layer.blocks[block_idx] = zero_bc4_pixels(
                    layer.blocks[block_idx], mask)
            for block_idx, omask in layer_outline_masks.items():
                layer.blocks[block_idx] = inject_bc4_outline(
                    layer.blocks[block_idx], omask, outline_value)
        else:
            for block_idx, mask in layer_masks.items():
                layer.blocks[block_idx] = zero_bc4_pixels(
                    layer.blocks[block_idx], mask)


def _apply_dxt1_masks_batch(layer, masks_dict):
    """Apply dither masks to all DXT1 blocks at once using numpy vectorization."""
    if not masks_dict:
        return

    block_indices = sorted(masks_dict.keys())
    mask_values = np.array([masks_dict[i] for i in block_indices], dtype=np.uint32)
    N = len(block_indices)

    # Handle all-transparent blocks (mask = 0xFFFF)
    # DXT1 transparent mode: c0 <= c1, index 3 = transparent
    # c0=0, c1=0, all indices = 3 (0xFF 0xFF 0xFF 0xFF)
    _ALL_TRANSPARENT = b'\x00\x00\x00\x00\xff\xff\xff\xff'
    all_trans = mask_values == 0xFFFF
    for i in np.flatnonzero(all_trans):
        layer.blocks[block_indices[i]] = _ALL_TRANSPARENT

    # Process partial transparency blocks in batch
    partial = ~all_trans
    if not np.any(partial):
        return

    p_indices = [block_indices[i] for i in np.flatnonzero(partial)]
    p_masks = mask_values[partial]
    M = len(p_indices)

    # Load block data into numpy array
    raw = b''.join(layer.blocks[i] for i in p_indices)
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(M, 8).copy()

    # Unpack little-endian: color0 (u16), color1 (u16), index_word (u32)
    c0 = blocks[:, 0].astype(np.uint32) | (blocks[:, 1].astype(np.uint32) << 8)
    c1 = blocks[:, 2].astype(np.uint32) | (blocks[:, 3].astype(np.uint32) << 8)
    idx = (blocks[:, 4].astype(np.uint32) |
           (blocks[:, 5].astype(np.uint32) << 8) |
           (blocks[:, 6].astype(np.uint32) << 16) |
           (blocks[:, 7].astype(np.uint32) << 24))

    # Handle opaque mode (c0 > c1): swap endpoints and remap indices
    # Remap: 0->1, 1->0, 2->2, 3->2 (bitwise, no Python loop)
    opaque = c0 > c1
    if np.any(opaque):
        c0_o, c1_o = c1[opaque].copy(), c0[opaque].copy()
        c0[opaque] = c0_o
        c1[opaque] = c1_o

        old = idx[opaque]
        low = old & np.uint32(0x55555555)
        high = (old >> np.uint32(1)) & np.uint32(0x55555555)
        not_high = ~high & np.uint32(0x55555555)
        idx3 = low & high
        idx[opaque] = (old ^ not_high) & ~idx3

    # Expand 16-bit dither masks to 32-bit index patterns and apply
    trans = _MASK_EXPAND[p_masks]
    idx = (idx & ~trans) | trans

    # Pack back into block bytes
    blocks[:, 0] = (c0 & 0xFF).astype(np.uint8)
    blocks[:, 1] = (c0 >> 8).astype(np.uint8)
    blocks[:, 2] = (c1 & 0xFF).astype(np.uint8)
    blocks[:, 3] = (c1 >> 8).astype(np.uint8)
    blocks[:, 4] = (idx & 0xFF).astype(np.uint8)
    blocks[:, 5] = ((idx >> 8) & 0xFF).astype(np.uint8)
    blocks[:, 6] = ((idx >> 16) & 0xFF).astype(np.uint8)
    blocks[:, 7] = ((idx >> 24) & 0xFF).astype(np.uint8)

    # Write back to layer
    for i in range(M):
        layer.blocks[p_indices[i]] = blocks[i].tobytes()


def _force_opaque_dxt1_batch(layer, masks_dict):
    """Force outline pixels to opaque (index 0) in transparent-mode DXT1 blocks.

    In transparent-mode DXT1 (c0 <= c1), index 3 = transparent. Original sprites
    may have transparent pixels at outline positions. This clears the index bits
    for those pixels, setting them to index 0 (color0 = opaque).

    Opaque-mode blocks (c0 > c1) are skipped since all indices are already opaque.
    """
    if not masks_dict:
        return

    block_indices = sorted(masks_dict.keys())
    mask_values = np.array([masks_dict[i] for i in block_indices], dtype=np.uint32)
    N = len(block_indices)

    raw = b''.join(layer.blocks[i] for i in block_indices)
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(N, 8).copy()

    c0 = blocks[:, 0].astype(np.uint32) | (blocks[:, 1].astype(np.uint32) << 8)
    c1 = blocks[:, 2].astype(np.uint32) | (blocks[:, 3].astype(np.uint32) << 8)
    idx = (blocks[:, 4].astype(np.uint32) |
           (blocks[:, 5].astype(np.uint32) << 8) |
           (blocks[:, 6].astype(np.uint32) << 16) |
           (blocks[:, 7].astype(np.uint32) << 24))

    # Only modify transparent-mode blocks (c0 <= c1) where index 3 = transparent
    trans_mode = c0 <= c1
    if not np.any(trans_mode):
        return

    # Clear both bits of 2-bit index for outline pixels -> index 0 (color0, opaque)
    clear_bits = _MASK_EXPAND[mask_values]
    clear_mask = np.where(trans_mode, clear_bits, np.uint32(0))
    idx &= ~clear_mask

    blocks[:, 4] = (idx & 0xFF).astype(np.uint8)
    blocks[:, 5] = ((idx >> 8) & 0xFF).astype(np.uint8)
    blocks[:, 6] = ((idx >> 16) & 0xFF).astype(np.uint8)
    blocks[:, 7] = ((idx >> 24) & 0xFF).astype(np.uint8)

    for i in np.flatnonzero(trans_mode):
        layer.blocks[block_indices[i]] = blocks[i].tobytes()


def process_file(input_path, output_path, tile_half_heights, tile_widths,
                 outline_value=200, edge_inset=3, gradient_height=0,
                 outline_thickness=4, no_outline=False,
                 dither_intensity=8, dither_bottom=False,
                 contour_width=0, contour_color='team'):
    """
    Process a single SLD file, applying transparency dithering.

    Args:
        input_path: Path to original SLD file
        output_path: Path to write modified SLD file
        tile_half_heights: dict {'x1': int, 'x2': int}
        tile_widths: dict {'x1': int, 'x2': int}
        outline_value: int, brightness for foundation outline (0-255)
        edge_inset: pixels from building edge to keep opaque
        gradient_height: transition zone height above foundation line
        outline_thickness: outline band height in pixels
        no_outline: disable foundation outline entirely
    """
    filename = os.path.basename(input_path)
    scale = 'x2' if '_x2.' in filename else 'x1'
    tile_hh = tile_half_heights[scale]
    tile_w = tile_widths[scale]

    with open(input_path, 'rb') as f:
        data = f.read()

    sld = parse_sld(data)

    # Determine building footprint from filename, with width-based fallback
    main_layer = get_layer(sld.frames[0], LAYER_MAIN) if sld.frames else None
    layer_w = (main_layer.offset_x2 - main_layer.offset_x1) if main_layer else 0
    tiles = get_building_tiles(filename, layer_w, tile_w)
    compound_offsets = get_gate_compound_offsets(filename, tile_hh)

    name_lower = filename.lower()
    outline_enabled = not no_outline

    # Town Center front/back variants: use normal dithering with
    # the standard TC footprint (units can walk on most of it).
    if 'town_center' in name_lower and ('_front' in name_lower or '_back' in name_lower):
        dither_bottom = True

    # Scale pixel-based parameters for UHD (x2) resolution
    scale_factor = 2 if scale == 'x2' else 1
    scaled_edge_inset = edge_inset * scale_factor
    scaled_outline_thickness = outline_thickness * scale_factor
    scaled_gradient_height = gradient_height * scale_factor

    # For animated sprites with delta encoding, get frame 0's full block
    # positions for stable edge protection across all delta frames.
    # Animation protection is not needed: delta encoding ensures dithering
    # consistency — frame 0's dithered body persists via skip commands,
    # and delta draw blocks get the same checkerboard pattern applied.
    frame0_positions = None
    if sld.num_frames > 1 and main_layer is not None:
        for f in sld.frames[1:]:
            l = get_layer(f, LAYER_MAIN)
            if l and (l.flag1 & 0x80):
                frame0_positions = get_block_positions(main_layer, sld.frames[0])
                break

    scaled_contour = contour_width * scale_factor

    for frame in sld.frames:
        process_frame(frame, tile_hh, tiles, outline_value, scaled_edge_inset,
                      scaled_gradient_height, scaled_outline_thickness,
                      outline_enabled=outline_enabled,
                      full_positions=frame0_positions,
                      dither_intensity=dither_intensity,
                      dither_bottom=dither_bottom,
                      contour_width=scaled_contour,
                      contour_color=contour_color,
                      compound_offsets=compound_offsets)

    output_data = write_sld(sld)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(output_data)

    return len(data), len(output_data)


def _process_file_worker(args):
    """Worker function for multiprocessing pool."""
    (input_path, output_path, tile_half_heights, tile_widths,
     outline_value, edge_inset, gradient_height, outline_thickness,
     no_outline, dither_intensity, dither_bottom, contour_width,
     contour_color) = args
    filename = os.path.basename(input_path)
    try:
        orig_size, new_size = process_file(input_path, output_path,
                                           tile_half_heights, tile_widths,
                                           outline_value, edge_inset,
                                           gradient_height, outline_thickness,
                                           no_outline, dither_intensity,
                                           dither_bottom, contour_width,
                                           contour_color)
        return (filename, orig_size, new_size, None)
    except Exception as e:
        return (filename, 0, 0, str(e))


DEFAULT_EXCLUDE = []

def find_building_files(exclude=None):
    """
    Find all building SLD files to process.

    Returns list of filenames (not full paths).
    Excludes destruction and rubble variants.

    Args:
        exclude: list of building type keywords to exclude (e.g. ['mill', 'monastery']).
                 Defaults to DEFAULT_EXCLUDE. Pass an empty list to include everything.
    """
    if exclude is None:
        exclude = DEFAULT_EXCLUDE

    pattern = os.path.join(get_graphics_dir(), "b_*.sld")
    all_files = glob.glob(pattern)

    result = []
    for filepath in all_files:
        name = os.path.basename(filepath)
        # Skip destruction, rubble, flags, and decorative side pieces
        if '_destruction_' in name or '_rubble_' in name:
            continue
        # Skip non-building overlays: gate/wall flags, waypoint flags, flagships,
        # lanterns, satrapy decorations, and decorative side pieces
        base = name.replace('_x1.sld', '').replace('_x2.sld', '')
        if base.endswith('_flag') or '_sides' in name:
            continue
        if '_waypoint_flag_' in name or '_flagship_' in name or '_lantern' in name:
            continue
        if '_satrapy' in name:
            continue
        # Skip non-competitive: scenario editor, foundations, fish traps,
        # mule carts, black tile, scenario towers
        if name.startswith('b_scen_') or '_cart_mule_' in name or '_fish_trap' in name:
            continue
        if 'black_tile' in name or '_foundation' in name or '_tower_scen' in name:
            continue
        # Skip Return of Rome architecture sets
        if any(prefix in name for prefix in (
            'b_archaic_', 'b_greek_', 'b_puru_', 'b_thracian',
            'b_spartans_', 'b_athenians_', 'b_macedonian_',
        )):
            continue
        # Skip user-specified building types
        if any(f'_{kw}_' in name or base.endswith(f'_{kw}') for kw in exclude):
            continue
        result.append(name)

    result.sort()
    return result


def main():
    parser = argparse.ArgumentParser(description="AoE2 DE Transparent Buildings Mod")
    parser.add_argument('--file', type=str, default=None,
                        help='Process a specific SLD file (filename only)')
    parser.add_argument('--tile-height', type=int, default=None,
                        help='Override tile half-height in pixels (default: 24 for x1, 48 for x2)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse and process but do not write output files')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--outline-value', type=int, default=200,
                        help='Brightness of foundation outline (0=off, 255=max, default: 200)')
    parser.add_argument('--edge-inset', type=int, default=3,
                        help='Pixels from building edge to keep opaque, auto-scaled 2x for UHD (default: 3)')
    parser.add_argument('--gradient-height', type=int, default=0,
                        help='Transition zone height above foundation in pixels (default: 0)')
    parser.add_argument('--outline-thickness', type=int, default=4,
                        help='Foundation outline band height in pixels (default: 4)')
    parser.add_argument('--no-outline', action='store_true',
                        help='Disable foundation outline entirely')
    parser.add_argument('--dither-intensity', type=int, default=8,
                        help='Bayer dither intensity 0-16 (0=opaque, 8=50%%, 16=fully transparent, default: 8)')
    parser.add_argument('--dither-bottom', action='store_true',
                        help='Also dither below the foundation line')
    parser.add_argument('--contour-width', type=int, default=0,
                        help='Contour width in pixels around building silhouette (default: 0 = off)')
    parser.add_argument('--contour-color', type=str, default='team', choices=['team', 'black'],
                        help='Contour color: team color or black (default: team)')
    parser.add_argument('--exclude', type=str, nargs='*', default=None,
                        help='Building types to exclude (default: mill). Pass without args to exclude nothing.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--combine-with', type=str, nargs='+', default=None,
                        help='Path(s) to other mod root directories to combine with. '
                             'Building SLDs are read from these mods in order (first match wins, '
                             'falling back to vanilla), and all resource files are copied through.')
    args = parser.parse_args()

    if args.tile_height is not None:
        TILE_HALF_HEIGHT['x1'] = args.tile_height
        TILE_HALF_HEIGHT['x2'] = args.tile_height * 2

    # Output directory (can be overridden)
    output_graphics_dir = args.output_dir if args.output_dir else get_mod_graphics_dir()

    # Create mod directory structure
    os.makedirs(output_graphics_dir, exist_ok=True)

    combine_mod_roots = args.combine_with or []
    for mr in combine_mod_roots:
        if not os.path.isdir(mr):
            print(f"Error: combine-with directory does not exist: {mr}")
            return 1

    # Derive graphics dirs from mod roots for building SLD resolution
    gfx_subpath = os.path.join("resources", "_common", "drs", "graphics")
    combine_gfx_dirs = []
    for mr in combine_mod_roots:
        gd = os.path.join(mr, gfx_subpath)
        if os.path.isdir(gd):
            combine_gfx_dirs.append(gd)

    exclude = args.exclude if args.exclude is not None else None
    if args.file:
        files = [args.file]
    else:
        files = find_building_files(exclude=exclude)

    print(f"Transparent Buildings Mod - Processing {len(files)} file(s)")
    graphics_dir = get_graphics_dir()
    print(f"  Input:  {graphics_dir}")
    for mr in combine_mod_roots:
        print(f"  Combine: {os.path.basename(mr)}")
    print(f"  Output: {output_graphics_dir}")
    print(f"  Tile half-height: {TILE_HALF_HEIGHT['x1']}px (x1), {TILE_HALF_HEIGHT['x2']}px (x2)")
    print(f"  Dither intensity: {args.dither_intensity}/16 ({round(args.dither_intensity * 100 / 16)}%)")
    print(f"  Dither bottom: {args.dither_bottom}")
    print(f"  Outline value: {args.outline_value}")
    print(f"  Edge inset: {args.edge_inset}px")
    print(f"  Gradient height: {args.gradient_height}px")
    print(f"  Outline thickness: {args.outline_thickness}px")
    print()

    success = 0
    errors = 0
    start_time = time.time()

    # Snapshot tile config (so workers on Windows get the right values)
    tile_hh = dict(TILE_HALF_HEIGHT)
    tile_w = dict(TILE_WIDTH)

    def _resolve_input(filename):
        """Resolve input path: check combine graphics dirs in order, fall back to vanilla."""
        for gd in combine_gfx_dirs:
            combined = os.path.join(gd, filename)
            if os.path.exists(combined):
                return combined
        return os.path.join(graphics_dir, filename)

    if args.dry_run:
        # Dry run: sequential, no output files
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Dry run...", total=len(files))
            for filename in files:
                input_path = _resolve_input(filename)
                if not os.path.exists(input_path):
                    progress.console.print(f"  [yellow]SKIP[/] {filename} (not found)")
                    errors += 1
                else:
                    try:
                        with open(input_path, 'rb') as f:
                            data = f.read()
                        sld = parse_sld(data)
                        progress.console.print(
                            f"  [green]OK[/] {filename}: "
                            f"{sld.num_frames} frames, {len(data):,} bytes")
                        success += 1
                    except Exception as e:
                        progress.console.print(f"  [red]ERROR[/] {filename}: {e}")
                        errors += 1
                progress.advance(task)
    else:
        # Build work items, skipping missing files
        work = []
        for filename in files:
            input_path = _resolve_input(filename)
            output_path = os.path.join(output_graphics_dir, filename)
            if not os.path.exists(input_path):
                print(f"  SKIP {filename} (not found)")
                errors += 1
                continue
            work.append((input_path, output_path, tile_hh, tile_w,
                         args.outline_value, args.edge_inset, args.gradient_height,
                         args.outline_thickness, args.no_outline,
                         args.dither_intensity, args.dither_bottom,
                         args.contour_width, args.contour_color))

        if len(work) == 1:
            # Single file: direct call, no pool overhead
            filename, orig_size, new_size, error = _process_file_worker(work[0])
            if error:
                print(f"  ERROR {filename}: {error}")
                errors += 1
            else:
                ratio = new_size / orig_size if orig_size > 0 else 0
                print(f"  OK {filename}: "
                      f"{orig_size:,} -> {new_size:,} bytes ({ratio:.1%})")
                success += 1
        elif work:
            # Parallel processing
            num_workers = args.workers or min(cpu_count(), len(work))
            print(f"  Workers: {num_workers}")
            print()

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Processing...", total=len(work))
                with Pool(num_workers) as pool:
                    for filename, orig_size, new_size, error in pool.imap_unordered(
                            _process_file_worker, work, chunksize=8):
                        if error:
                            progress.console.print(f"  [red]ERROR[/] {filename}: {error}")
                            errors += 1
                        else:
                            ratio = new_size / orig_size if orig_size > 0 else 0
                            progress.console.print(
                                f"  [green]OK[/] {filename}: "
                                f"{orig_size:,} -> {new_size:,} bytes ({ratio:.1%})")
                            success += 1
                        progress.advance(task)

    # Copy resources from combined mods (later mods don't overwrite earlier ones)
    copied = 0
    if combine_mod_roots and not args.dry_run:
        building_files = set(files)
        mod_dir = get_mod_dir()
        for mr in combine_mod_roots:
            src_res = os.path.join(mr, "resources")
            dst_res = os.path.join(mod_dir, "resources")
            if not os.path.isdir(src_res):
                continue
            for dirpath, dirnames, filenames in os.walk(src_res):
                rel = os.path.relpath(dirpath, src_res)
                dst_dir = os.path.join(dst_res, rel)
                os.makedirs(dst_dir, exist_ok=True)
                for fn in filenames:
                    dst_file = os.path.join(dst_dir, fn)
                    # Skip building files we already processed
                    if rel == os.path.join("_common", "drs", "graphics") and fn in building_files:
                        continue
                    # Don't overwrite files from earlier mods
                    if os.path.exists(dst_file):
                        continue
                    shutil.copy2(os.path.join(dirpath, fn), dst_file)
                    copied += 1

    elapsed = time.time() - start_time
    print()
    if copied > 0:
        print(f"Copied {copied} non-building file(s) from combined mod")
    print(f"Done in {elapsed:.1f}s: {success} succeeded, {errors} failed")

    if success > 0 and not args.dry_run:
        # Write info.json for mod
        import json
        mod_dir = get_mod_dir()
        info_path = os.path.join(mod_dir, "info.json")
        info = {
            "Title": "Transparent Buildings",
            "Description": (
                "See through buildings! The upper part of every building "
                "becomes semi-transparent using a checkerboard dither pattern, "
                "letting you spot units hiding behind them. The foundation "
                "stays solid with a team-color diamond outline so you can "
                "still see building footprints clearly. Works with all "
                "civilizations, animated buildings (mills, folwarks), and "
                "both standard and UHD graphics."
            ),
            "Author": "Yustee",
            "CacheStatus": 0,
        }
        with open(info_path, 'w') as f:
            json.dump(info, f)
        print(f"Wrote mod info: {info_path}")

        # Generate thumbnail if castle was built
        thumb_path = os.path.join(mod_dir, "thumbnail.png")
        showcase = "b_west_castle_age3_x1.sld"
        orig_file = os.path.join(graphics_dir, showcase)
        mod_file = os.path.join(output_graphics_dir, showcase)
        if os.path.exists(orig_file) and os.path.exists(mod_file):
            try:
                _generate_thumbnail(orig_file, mod_file, thumb_path)
                print(f"Wrote thumbnail: {thumb_path}")
            except Exception as e:
                print(f"Thumbnail generation failed: {e}")

    return 0 if errors == 0 else 1


def _generate_thumbnail(orig_path, mod_path, out_path, width=1280, height=720):
    """Generate a before/after thumbnail for the mod."""
    from sld import LAYER_MAIN
    from tools.sld_to_png import render_frame, save_png

    with open(orig_path, 'rb') as f:
        orig_sld = parse_sld(f.read())
    with open(mod_path, 'rb') as f:
        mod_sld = parse_sld(f.read())

    orig = render_frame(orig_sld.frames[0], LAYER_MAIN)
    modded = render_frame(mod_sld.frames[0], LAYER_MAIN)

    thumb = np.zeros((height, width, 4), dtype=np.uint8)
    thumb[:, :, 0] = 45
    thumb[:, :, 1] = 65
    thumb[:, :, 2] = 30
    thumb[:, :, 3] = 255

    # Divider
    thumb[:, width // 2 - 1:width // 2 + 1, :3] = 200

    half_w = width // 2
    for canvas, x_base in [(orig, 0), (modded, half_w)]:
        ch, cw = canvas.shape[:2]
        y0 = max(0, (height - ch) // 2 + 20)
        x0 = x_base + max(0, (half_w - cw) // 2)
        yend = min(y0 + ch, height)
        xend = min(x0 + cw, width)
        sh = yend - y0
        sw = xend - x0
        alpha = canvas[:sh, :sw, 3:4].astype(np.float32) / 255.0
        thumb[y0:yend, x0:xend, :3] = (
            canvas[:sh, :sw, :3] * alpha +
            thumb[y0:yend, x0:xend, :3] * (1 - alpha)
        ).astype(np.uint8)

    save_png(thumb, out_path, rgb=True)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    sys.exit(main())
