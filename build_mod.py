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
from scipy.ndimage import distance_transform_edt
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
            if direction in ('ne', 'n'):
                return (1, 2)
            else:  # 'se', 'e'
                return (2, 1)
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

# Precomputed: 16-bit dither mask -> 32-bit DXT1 index transparency pattern
# Each set bit i in the 16-bit mask becomes 0b11 at bits 2i+1:2i (DXT1 index 3 = transparent)
_MASK_EXPAND = np.zeros(65536, dtype=np.uint32)
for _j in range(16):
    _MASK_EXPAND += ((np.arange(65536, dtype=np.uint32) >> _j) & 1) * np.uint32(3 << (2 * _j))


def compute_dither_masks(block_xs, block_ys, center_x, center_y,
                         margin_u, margin_v, gradient_height=0):
    """
    Vectorized computation of 16-bit dither masks for N blocks at once.

    The isometric diamond is defined by two constraints in rotated space:
      |dx/2 + dy| <= margin_u  AND  |-dx/2 + dy| <= margin_v
    Top edge = max of both lower bounds, bottom edge = min of both upper bounds.
    For square NxN, margin_u == margin_v and this reduces to the simple formula.
    """
    # Expand to per-pixel coordinates: (N, 16)
    pixel_ys = block_ys[:, None] + _PIX_ROWS[None, :]
    pixel_xs = block_xs[:, None] + _PIX_COLS[None, :]

    # Isometric diamond edges from both constraints
    above_hotspot = pixel_ys < center_y
    dx = (pixel_xs - center_x).astype(np.float64)
    # Top edge: max of the two lower bounds
    top_a = -margin_u - dx * 0.5  # from |dx/2 + dy| <= margin_u
    top_b = -margin_v + dx * 0.5  # from |-dx/2 + dy| <= margin_v
    foundation_y = center_y + np.maximum(top_a, top_b)
    checkerboard = (pixel_xs + pixel_ys) % 2 == 0

    # Gradient zone: sparser dithering near the foundation line
    if gradient_height > 0:
        full_zone = pixel_ys < (foundation_y - gradient_height)
        in_gradient = ~full_zone & (pixel_ys < foundation_y)
        sparse_checkerboard = (pixel_xs % 2 == 0) & (pixel_ys % 2 == 0)
        dither = above_hotspot & (
            (full_zone & checkerboard) |
            (in_gradient & sparse_checkerboard)
        )
    else:
        above_foundation = pixel_ys < foundation_y
        dither = above_hotspot & above_foundation & checkerboard

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


def compute_edge_protection(positions, edge_inset):
    """
    Compute per-block 16-bit masks of pixels within edge_inset of the silhouette edge.

    Uses the drawn-block grid (from skip/draw commands) as the building silhouette,
    then erodes it to find pixels near the edge. These pixels should be kept opaque.

    Args:
        positions: list of (block_idx, block_x, block_y) from get_block_positions
        edge_inset: pixels from building edge to keep opaque

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
    for _, bx, by in positions:
        px = (bx - min_bx) + pad
        py = (by - min_by) + pad
        drawn[py:py+4, px:px+4] = True

    # Euclidean distance from each drawn pixel to the nearest non-drawn pixel
    dist = distance_transform_edt(drawn)

    # Boundary = drawn pixels within edge_inset of the silhouette edge
    boundary = drawn & (dist <= edge_inset)

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
                  edge_inset=0, gradient_height=0, outline_thickness=4,
                  outline_enabled=True):
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

    masks = compute_dither_masks(block_xs, block_ys, cx, cy,
                                 margin_u, margin_v, gradient_height)

    if outline_enabled:
        outline_masks = compute_outline_masks(
            block_xs, block_ys, cx, cy, margin_u, margin_v, outline_thickness)
    else:
        outline_masks = np.zeros(len(main_positions), dtype=np.uint32)

    # Edge protection: keep outermost pixels opaque based on silhouette boundary
    edge_prot_cache = {}
    if edge_inset > 0:
        edge_prot_cache = compute_edge_protection(main_positions, edge_inset)
        if edge_prot_cache:
            prot_array = np.array(
                [edge_prot_cache.get((int(block_xs[i]), int(block_ys[i])), 0)
                 for i in range(len(main_positions))], dtype=np.uint32)
            masks &= ~prot_array

    # Keep main layer opaque at outline pixels so team color shows through
    if outline_enabled:
        masks &= ~outline_masks

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

    if not any_dithered:
        return

    # Apply dithering to main graphic layer (DXT1)
    _apply_dxt1_masks_batch(main_layer, block_dither_masks)

    if outline_enabled:
        # Force outline pixels to opaque in main layer DXT1 blocks.
        block_outline_masks = {}
        for i in range(len(main_positions)):
            om = int(outline_masks[i])
            if om:
                block_outline_masks[int(block_idxs[i])] = om
        _force_opaque_dxt1_batch(main_layer, block_outline_masks)

    if outline_enabled:
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

        # Extend vertical bounds if bottom diamond edge exceeds them
        bottom_edge_y = cy + max(margin_u, margin_v)
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
            new_masks = compute_dither_masks(u_array[:, 1], u_array[:, 2],
                                             cx, cy, margin_u, margin_v,
                                             gradient_height)
            # Apply edge protection to uncached masks
            if edge_prot_cache:
                u_prot = np.array(
                    [edge_prot_cache.get((int(u_array[i, 1]), int(u_array[i, 2])), 0)
                     for i in range(len(uncached))], dtype=np.uint32)
                new_masks &= ~u_prot
            if layer_type == LAYER_PLAYERCOLOR:
                new_outlines = compute_outline_masks(
                    u_array[:, 1], u_array[:, 2], cx, cy,
                    margin_u, margin_v, outline_thickness,
            )
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
            # Player color: zero dithered pixels, then inject outline
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
    all_trans = mask_values == 0xFFFF
    for i in np.flatnonzero(all_trans):
        layer.blocks[block_indices[i]] = _ZERO_BLOCK

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
                 outline_value=200, edge_inset=0, gradient_height=0,
                 outline_thickness=4, no_outline=False):
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

    # Disable outline on directional gate files (the middle gate opening)
    name_lower = filename.lower()
    outline_enabled = not no_outline
    if outline_enabled and 'gate' in name_lower and _GATE_DIR_RE.search(name_lower):
        outline_enabled = False

    for frame in sld.frames:
        process_frame(frame, tile_hh, tiles, outline_value, edge_inset,
                      gradient_height, outline_thickness,
                      outline_enabled=outline_enabled)

    output_data = write_sld(sld)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(output_data)

    return len(data), len(output_data)


def _process_file_worker(args):
    """Worker function for multiprocessing pool."""
    (input_path, output_path, tile_half_heights, tile_widths,
     outline_value, edge_inset, gradient_height, outline_thickness,
     no_outline) = args
    filename = os.path.basename(input_path)
    try:
        orig_size, new_size = process_file(input_path, output_path,
                                           tile_half_heights, tile_widths,
                                           outline_value, edge_inset,
                                           gradient_height, outline_thickness,
                                           no_outline)
        return (filename, orig_size, new_size, None)
    except Exception as e:
        return (filename, 0, 0, str(e))


DEFAULT_EXCLUDE = ['mill']

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
        # Skip flag overlays (gate/wall/garrison flags) and decorative side pieces
        # Use endswith check to avoid matching 'flagship' or 'satrapy_flags'
        base = name.replace('_x1.sld', '').replace('_x2.sld', '')
        if base.endswith('_flag') or '_sides' in name:
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
    parser.add_argument('--edge-inset', type=int, default=0,
                        help='Pixels from building edge to keep opaque (default: 0)')
    parser.add_argument('--gradient-height', type=int, default=0,
                        help='Transition zone height above foundation in pixels (default: 0)')
    parser.add_argument('--outline-thickness', type=int, default=4,
                        help='Foundation outline band height in pixels (default: 4)')
    parser.add_argument('--no-outline', action='store_true',
                        help='Disable foundation outline entirely')
    parser.add_argument('--exclude', type=str, nargs='*', default=None,
                        help='Building types to exclude (default: mill). Pass without args to exclude nothing.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    args = parser.parse_args()

    if args.tile_height is not None:
        TILE_HALF_HEIGHT['x1'] = args.tile_height
        TILE_HALF_HEIGHT['x2'] = args.tile_height * 2

    # Output directory (can be overridden)
    output_graphics_dir = args.output_dir if args.output_dir else get_mod_graphics_dir()

    # Create mod directory structure
    os.makedirs(output_graphics_dir, exist_ok=True)

    exclude = args.exclude if args.exclude is not None else None
    if args.file:
        files = [args.file]
    else:
        files = find_building_files(exclude=exclude)

    print(f"Transparent Buildings Mod - Processing {len(files)} file(s)")
    graphics_dir = get_graphics_dir()
    print(f"  Input:  {graphics_dir}")
    print(f"  Output: {output_graphics_dir}")
    print(f"  Tile half-height: {TILE_HALF_HEIGHT['x1']}px (x1), {TILE_HALF_HEIGHT['x2']}px (x2)")
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
                input_path = os.path.join(graphics_dir, filename)
                if not os.path.exists(input_path):
                    progress.console.print(f"  [yellow]SKIP[/] {filename} (not found)")
                    errors += 1
                else:
                    try:
                        with open(input_path, 'rb') as f:
                            data = f.read()
                        sld = parse_sld(data)
                        success += 1
                    except Exception as e:
                        progress.console.print(f"  [red]ERROR[/] {filename}: {e}")
                        errors += 1
                progress.advance(task)
    else:
        # Build work items, skipping missing files
        work = []
        for filename in files:
            input_path = os.path.join(graphics_dir, filename)
            output_path = os.path.join(output_graphics_dir, filename)
            if not os.path.exists(input_path):
                print(f"  SKIP {filename} (not found)")
                errors += 1
                continue
            work.append((input_path, output_path, tile_hh, tile_w,
                         args.outline_value, args.edge_inset, args.gradient_height,
                         args.outline_thickness, args.no_outline))

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
                            success += 1
                        progress.advance(task)

    elapsed = time.time() - start_time
    print()
    print(f"Done in {elapsed:.1f}s: {success} succeeded, {errors} failed")

    if success > 0 and not args.dry_run:
        # Create info.json for mod
        info_path = os.path.join(get_mod_dir(), "info.json")
        if not os.path.exists(info_path):
            import json
            info = {
                "Title": "Transparent Buildings",
                "Description": "See through buildings! The upper part of every building becomes see-through so you can spot units and holes hiding behind them. The foundation stays solid.",
                "Author": "Yustee",
                "Version": "1.0"
            }
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            print(f"Created mod info: {info_path}")

    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
