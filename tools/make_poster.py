"""Generate before/after comparison poster images for the mod page.

Creates a grid of buildings showing original (left) vs modded (right)
side-by-side, suitable for uploading as a mod preview image.

Usage:
    uv run make-poster                        # Default showcase
    uv run make-poster --output poster.png    # Custom output name
    uv run make-poster --style west           # Only western buildings
    uv run make-poster --cols 3               # 3 comparison pairs per row
"""
import os
import sys
import struct
import argparse

import numpy as np

from paths import get_graphics_dir, get_mod_graphics_dir
from sld import parse_sld
from dxt import decode_dxt1_block
from tools.sld_to_png import render_frame, save_png, LAYER_BY_NAME

# Curated showcase buildings: (filename, display_name, terrain)
# terrain: 'grass' (default) or 'water'
# Grouped so similar buildings sit next to each other for comparison
SHOWCASE = [
    # Row 1: Castles - biggest selling point
    ('b_west_castle_age3_x1.sld', 'Castle', 'grass'),
    ('b_asia_castle_age3_x1.sld', 'Castle (East Asian)', 'grass'),
    ('b_east_castle_age3_x1.sld', 'Castle (Middle East)', 'grass'),
    # Row 2: Key buildings
    ('b_west_town_center_age4_x1.sld', 'Town Center', 'grass'),
    ('b_west_monastery_age3_x1.sld', 'Monastery', 'grass'),
    ('wall_gate', 'Wall + Gate', 'grass'),
    # Row 3: Military & economy
    ('b_west_barracks_age3_x1.sld', 'Barracks', 'grass'),
    ('b_west_market_age3_x1.sld', 'Market', 'grass'),
    ('b_west_dock_age3_x1.sld', 'Dock', 'shore'),
]

# Style-specific subsets
STYLE_SETS = {
    'west': [
        ('b_west_house_age2_x1.sld', 'House'),
        ('b_west_barracks_age3_x1.sld', 'Barracks'),
        ('b_west_archery_range_age3_x1.sld', 'Archery Range'),
        ('b_west_stable_age2_x1.sld', 'Stable'),
        ('b_west_monastery_age3_x1.sld', 'Monastery'),
        ('b_west_siege_workshop_age3_x1.sld', 'Siege Workshop'),
        ('b_west_blacksmith_age3_x1.sld', 'Blacksmith'),
        ('b_west_market_age3_x1.sld', 'Market'),
        ('b_west_university_age3_x1.sld', 'University'),
        ('b_west_town_center_age4_x1.sld', 'Town Center'),
        ('b_west_castle_age3_x1.sld', 'Castle'),
        ('b_west_wonder_britons_x1.sld', 'Wonder'),
    ],
    'asia': [
        ('b_asia_house_age2_x1.sld', 'House'),
        ('b_asia_barracks_age3_x1.sld', 'Barracks'),
        ('b_asia_town_center_age4_x1.sld', 'Town Center'),
        ('b_asia_castle_age3_x1.sld', 'Castle'),
        ('b_asia_monastery_age3_x1.sld', 'Monastery'),
        ('b_asia_market_age3_x1.sld', 'Market'),
    ],
    'east': [
        ('b_east_house_age2_x1.sld', 'House'),
        ('b_east_barracks_age3_x1.sld', 'Barracks'),
        ('b_east_town_center_age4_x1.sld', 'Town Center'),
        ('b_east_castle_age3_x1.sld', 'Castle'),
        ('b_east_monastery_age3_x1.sld', 'Monastery'),
        ('b_east_market_age3_x1.sld', 'Market'),
    ],
    'meso': [
        ('b_meso_house_age2_x1.sld', 'House'),
        ('b_meso_barracks_age3_x1.sld', 'Barracks'),
        ('b_meso_town_center_age4_x1.sld', 'Town Center'),
        ('b_meso_castle_age3_x1.sld', 'Castle'),
        ('b_meso_monastery_age3_x1.sld', 'Monastery'),
        ('b_meso_market_age3_x1.sld', 'Market'),
    ],
}

LABEL_HEIGHT = 0  # reserved for future text labels
GAP = 32  # gap between original and mod (room for arrow)
CELL_PAD = 4  # padding around each cell


def _decode_dds_dxt1(filepath):
    """Load a DXT1 DDS file and decode mip level 0 to RGBA."""
    with open(filepath, 'rb') as f:
        data = f.read()

    # Parse DDS header
    magic = data[:4]
    if magic != b'DDS ':
        raise ValueError(f"Not a DDS file: {magic!r}")

    _size, _flags, height, width = struct.unpack_from('<4I', data, 4)

    # DXT1: 8 bytes per 4x4 block
    blocks_w = (width + 3) // 4
    blocks_h = (height + 3) // 4
    pixel_data_offset = 4 + 124  # magic + header

    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    for by in range(blocks_h):
        for bx in range(blocks_w):
            block_idx = by * blocks_w + bx
            offset = pixel_data_offset + block_idx * 8
            block = data[offset:offset + 8]
            pixels = decode_dxt1_block(block)
            # Write 4x4 block to canvas
            py, px = by * 4, bx * 4
            rh = min(4, height - py)
            rw = min(4, width - px)
            canvas[py:py+rh, px:px+rw] = pixels[:rh, :rw]

    return canvas


def _downsample_2x(img):
    """Downsample RGBA image by 2x using box filter (average of 2x2 blocks)."""
    h, w = img.shape[:2]
    h2, w2 = h // 2, w // 2
    return (img[:h2*2, :w2*2].reshape(h2, 2, w2, 2, 4)
            .astype(np.uint16).mean(axis=(1, 3)).astype(np.uint8))


def _load_terrain_texture(terrain_type='grass', scale='x1'):
    """Load an in-game terrain texture from AoE2 DE files.

    Textures in 2x/ are UHD scale and get downsampled to match x1 sprites.
    For x2 scale, textures are used at native resolution.
    """
    game_dir = os.path.dirname(os.path.dirname(get_graphics_dir()))
    terrain_dir = os.path.join(game_dir, 'terrain', 'textures', '2x')

    if terrain_type == 'water':
        candidates = ['g_wtr.dds', 'g_sha.dds']
    else:
        candidates = ['g_grs.dds', 'g_gr2.dds', 'g_gr3.dds']

    for name in candidates:
        path = os.path.join(terrain_dir, name)
        if os.path.exists(path):
            tex = _decode_dds_dxt1(path)
            # Downsample from 2x to x1 scale only for x1
            if scale == 'x1' and tex.shape[0] > 1024:
                tex = _downsample_2x(tex)
            return tex

    return None


# Cache loaded terrain textures
_terrain_cache = {}


_terrain_scale = 'x1'


def _get_terrain(terrain_type='grass'):
    """Get terrain texture tile, cached."""
    key = (terrain_type, _terrain_scale)
    if key not in _terrain_cache:
        print(f"  Loading {terrain_type} texture ({_terrain_scale})...")
        tex = _load_terrain_texture(terrain_type, _terrain_scale)
        if tex is None:
            # Fallback solid color
            fallback_colors = {'grass': (87, 122, 52), 'water': (40, 70, 120)}
            color = fallback_colors.get(terrain_type, (30, 30, 30))
            tex = np.zeros((64, 64, 4), dtype=np.uint8)
            tex[:, :, :3] = color
            tex[:, :, 3] = 255
        _terrain_cache[key] = tex
    return _terrain_cache[key]


def _fill_terrain(canvas, x, y, w, h, terrain_type='grass'):
    """Fill a rectangular region of the canvas with tiled terrain.

    For 'shore' terrain, fills top portion with grass and bottom with water,
    with a diagonal transition matching the isometric perspective.
    """
    if terrain_type == 'shore':
        _fill_shore(canvas, x, y, w, h)
        return

    tile = _get_terrain(terrain_type)
    th, tw = tile.shape[:2]

    for ty in range(0, h, th):
        for tx in range(0, w, tw):
            src_h = min(th, h - ty)
            src_w = min(tw, w - tx)
            dy = y + ty
            dx = x + tx
            if dy + src_h > canvas.shape[0]:
                src_h = canvas.shape[0] - dy
            if dx + src_w > canvas.shape[1]:
                src_w = canvas.shape[1] - dx
            if src_h > 0 and src_w > 0:
                canvas[dy:dy+src_h, dx:dx+src_w] = tile[:src_h, :src_w]


def _fill_shore(canvas, x, y, w, h, building_bottom=None):
    """Fill a cell with grass in bottom-left, water in upper-right.

    The shoreline runs diagonally at isometric angle (slope = 0.5).
    The dock/building sits on the water side (upper-right), with
    land visible behind it (bottom-left).

    If building_bottom is given, the shore line is placed so it passes
    through the bottom edge of the building.
    """
    # First fill entirely with water
    _fill_terrain(canvas, x, y, w, h, 'water')

    grass = _get_terrain('grass')
    gth, gtw = grass.shape[:2]

    # Positive slope: shoreline goes from upper-left to lower-right
    # This puts grass in bottom-left, water in upper-right
    slope = 0.5
    if building_bottom is not None:
        # Position shore line so it crosses the center of the region
        # at the building's bottom edge
        center_x_rel = w / 2
        y_offset = building_bottom - slope * center_x_rel
    else:
        y_offset = y + h * 0.70

    beach_width = 40

    for py in range(y, min(y + h, canvas.shape[0])):
        for px in range(x, min(x + w, canvas.shape[1])):
            shore_y = slope * (px - x) + y_offset
            dist = py - shore_y  # positive = below shore = grass side

            if dist > beach_width:
                # Solid grass
                gx = (px - x) % gtw
                gy = (py - y) % gth
                canvas[py, px] = grass[gy, gx]
            elif dist > 0:
                # Beach to grass transition
                t = dist / beach_width
                beach = np.array([160, 150, 105, 255], dtype=np.float32)
                gx = (px - x) % gtw
                gy = (py - y) % gth
                grass_px = grass[gy, gx].astype(np.float32)
                canvas[py, px] = (beach * (1 - t) + grass_px * t).astype(np.uint8)
            elif dist > -beach_width:
                # Water to beach transition
                t = (dist + beach_width) / beach_width
                beach = np.array([160, 150, 105, 255], dtype=np.float32)
                water_px = canvas[py, px].astype(np.float32)
                canvas[py, px] = (water_px * (1 - t) + beach * t).astype(np.uint8)
            # else: already water


def render_building(filepath, layer_type=None, frame_idx=0):
    """Load and render a frame of an SLD file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        data = f.read()
    sld = parse_sld(data)
    if not sld.frames or frame_idx >= len(sld.frames):
        return None
    return render_frame(sld.frames[frame_idx], layer_type)


def render_wall_gate(gfx_dir, layer_type=None, scale='x1'):
    """Render a wall+gate+wall composite.

    Uses fortified wall frame 2 (tower) and fortified gate E (closed).
    Returns an RGBA canvas with the composite.
    """
    suffix = '_x1.sld' if scale == 'x1' else '_x2.sld'
    gate_path = os.path.join(gfx_dir, f'b_west_gate_fortified_e_closed{suffix}')
    wall_path = os.path.join(gfx_dir, f'b_west_wall_fortified{suffix}')

    gate = render_building(gate_path, layer_type, frame_idx=0)
    wall = render_building(wall_path, layer_type, frame_idx=2)
    if gate is None or wall is None:
        return None

    # Scale factor: x2 sprites are 2x larger
    s = 2 if scale == 'x2' else 1
    tile_dx = 140 * s
    tile_dy = -20 * s

    # Canvas large enough for wall + gate + wall
    cw = 800 * s
    ch = 400 * s
    cx, cy = cw // 2, ch // 2  # composite center

    canvas = np.zeros((ch, cw, 4), dtype=np.uint8)

    # Sprite centers (gate/wall canvases scale with sprite size)
    gate_cx = gate.shape[1] // 2
    gate_cy = gate.shape[0] // 2
    wall_cx = wall.shape[1] // 2
    wall_cy = wall.shape[0] // 2

    def place(sprite, sprite_cx, sprite_cy, dst_cx, dst_cy):
        """Place sprite on canvas aligning sprite center to dst center."""
        ox = dst_cx - sprite_cx
        oy = dst_cy - sprite_cy
        sh, sw = sprite.shape[:2]
        # Clip
        sx1 = max(0, -ox)
        sy1 = max(0, -oy)
        dx1 = max(0, ox)
        dy1 = max(0, oy)
        w = min(sw - sx1, cw - dx1)
        h = min(sh - sy1, ch - dy1)
        if w > 0 and h > 0:
            region = sprite[sy1:sy1+h, sx1:sx1+w]
            alpha = region[:, :, 3:4].astype(np.float32) / 255.0
            dst = canvas[dy1:dy1+h, dx1:dx1+w]
            dst[:, :, :3] = (region[:, :, :3].astype(np.float32) * alpha +
                             dst[:, :, :3].astype(np.float32) * (1 - alpha)).astype(np.uint8)
            dst[:, :, 3] = np.maximum(dst[:, :, 3], region[:, :, 3])

    # Place left wall, gate, right wall (back to front for proper overlap)
    # Gate raised to connect at tower side corners, not tower base
    gate_dy = -25
    place(wall, wall_cx, wall_cy, cx - tile_dx, cy + tile_dy)
    place(gate, gate_cx, gate_cy, cx, cy + gate_dy)
    place(wall, wall_cx, wall_cy, cx + tile_dx, cy + tile_dy)

    return canvas


def crop_to_content(canvas, pad=4):
    """Crop RGBA canvas to non-transparent content with padding."""
    alpha = canvas[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return canvas

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding
    rmin = max(0, rmin - pad)
    rmax = min(canvas.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(canvas.shape[1] - 1, cmax + pad)

    return canvas[rmin:rmax+1, cmin:cmax+1]


def _resize(img, target_h, target_w):
    """Resize RGBA image using block averaging for clean downscaling."""
    src_h, src_w = img.shape[:2]

    # Map each target pixel to its source range
    y_indices = np.linspace(0, src_h, target_h + 1).astype(int)
    x_indices = np.linspace(0, src_w, target_w + 1).astype(int)

    result = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    img_f = img.astype(np.float32)

    for ty in range(target_h):
        y0, y1 = y_indices[ty], max(y_indices[ty + 1], y_indices[ty] + 1)
        row_avg = img_f[y0:y1].mean(axis=0)  # (src_w, 4)
        for tx in range(target_w):
            x0, x1 = x_indices[tx], max(x_indices[tx + 1], x_indices[tx] + 1)
            result[ty, tx] = row_avg[x0:x1].mean(axis=0).astype(np.uint8)

    return result


def make_poster(buildings, cols=2, output='poster.png', layer='main', target_size=None, scale='x1', mod_dir=None):
    """Generate the poster image."""
    global _terrain_scale
    _terrain_scale = scale
    gfx = get_graphics_dir()
    if mod_dir:
        # Support both flat dirs and full mod structure
        test_path = os.path.join(mod_dir, 'resources', '_common', 'drs', 'graphics')
        mod_gfx = test_path if os.path.isdir(test_path) else mod_dir
    else:
        mod_gfx = get_mod_graphics_dir()
    layer_type = LAYER_BY_NAME[layer]

    # Render all pairs
    pairs = []
    for entry in buildings:
        if len(entry) == 3:
            filename, name, terrain = entry
        else:
            filename, name = entry[0], entry[1]
            terrain = 'grass'

        # Swap x1 → x2 filenames when rendering at x2 scale
        if scale == 'x2' and filename != 'wall_gate':
            filename = filename.replace('_x1.sld', '_x2.sld')

        if filename == 'wall_gate':
            orig = render_wall_gate(gfx, layer_type, scale)
            mod = render_wall_gate(mod_gfx, layer_type, scale)
        else:
            orig = render_building(os.path.join(gfx, filename), layer_type)
            mod = render_building(os.path.join(mod_gfx, filename), layer_type)
        if orig is None:
            print(f"  Skipping {filename} (not found in game files)")
            continue
        if mod is None:
            print(f"  Skipping {filename} (not found in mod files)")
            continue

        # Crop both to content
        orig_c = crop_to_content(orig)
        mod_c = crop_to_content(mod)

        # Make same size (use max dimensions)
        h = max(orig_c.shape[0], mod_c.shape[0])
        w = max(orig_c.shape[1], mod_c.shape[1])

        def center_on(canvas, target_h, target_w):
            result = np.zeros((target_h, target_w, 4), dtype=np.uint8)
            dy = (target_h - canvas.shape[0]) // 2
            dx = (target_w - canvas.shape[1]) // 2
            result[dy:dy+canvas.shape[0], dx:dx+canvas.shape[1]] = canvas
            return result

        orig_c = center_on(orig_c, h, w)
        mod_c = center_on(mod_c, h, w)

        pairs.append((name, orig_c, mod_c, h, w, terrain))
        print(f"  {name}: {filename} ({w}x{h})")

    if not pairs:
        print("No buildings rendered!")
        return

    # Lay out grid: each cell is [orig | gap | mod], cells arranged in cols columns
    rows = (len(pairs) + cols - 1) // cols

    # Find max cell dimensions per row/col for alignment
    cell_widths = []
    cell_heights = []
    for _, _, _, h, w, _terrain in pairs:
        cell_w = w * 2 + GAP  # orig + gap + mod
        cell_widths.append(cell_w)
        cell_heights.append(h)

    # Use uniform cell size based on max
    max_cell_w = max(cell_widths) + CELL_PAD * 2
    max_cell_h = max(cell_heights) + CELL_PAD * 2

    # Arrow/label column between orig and mod
    poster_w = cols * max_cell_w + CELL_PAD
    poster_h = rows * max_cell_h + CELL_PAD

    # Start with transparent canvas, fill terrain per cell
    poster = np.zeros((poster_h, poster_w, 4), dtype=np.uint8)

    for idx, (name, orig, mod, h, w, terrain) in enumerate(pairs):
        row = idx // cols
        col = idx % cols

        cell_x = col * max_cell_w + CELL_PAD
        cell_y = row * max_cell_h + CELL_PAD

        cell_w = max_cell_w - CELL_PAD
        cell_h = max_cell_h - CELL_PAD

        # Center the pair within the cell
        pair_w = w * 2 + GAP
        pair_h = h
        ox = cell_x + (cell_w - pair_w) // 2
        oy = cell_y + (cell_h - pair_h) // 2

        if terrain == 'shore':
            # Two separate shore backgrounds (one per building)
            # Shore line passes through the bottom of the building
            half_w = (cell_w - GAP) // 2
            left_x = cell_x
            right_x = cell_x + half_w + GAP
            # Shore line at ~85% of building height so dock base overlaps the beach
            building_shore = oy + int(h * 0.85)
            _fill_shore(poster, left_x, cell_y, half_w, cell_h, building_shore)
            _fill_shore(poster, right_x, cell_y, half_w, cell_h, building_shore)
            # Center each building within its own half
            ox_l = left_x + (half_w - w) // 2
            ox_r = right_x + (half_w - w) // 2
        else:
            _fill_terrain(poster, cell_x, cell_y, cell_w, cell_h, terrain)
            ox_l = ox
            ox_r = ox + w + GAP

        # Draw original (left)
        _composite(poster, orig, ox_l, oy)
        # Draw mod (right)
        _composite(poster, mod, ox_r, oy)

        # Draw arrow between orig and mod
        arrow_x = (ox_l + w + ox_r) // 2
        arrow_y = oy + h // 2
        _draw_arrow(poster, arrow_x, arrow_y)

    # Resize to target size (fit within bounds, no stretching)
    tw, th = target_size
    # Scale uniformly to fit, then crop/center to exact size
    scale = min(tw / poster_w, th / poster_h)
    scaled_w = int(poster_w * scale)
    scaled_h = int(poster_h * scale)
    poster = _resize(poster, scaled_h, scaled_w)

    # Center on target-sized canvas with grass background
    final = np.zeros((th, tw, 4), dtype=np.uint8)
    _fill_terrain(final, 0, 0, tw, th, 'grass')
    ox = (tw - scaled_w) // 2
    oy = (th - scaled_h) // 2
    final[oy:oy+scaled_h, ox:ox+scaled_w] = poster
    poster = final
    poster_w, poster_h = tw, th

    # Black bars at top and bottom
    bar_h = 32
    bar = np.zeros((bar_h, poster_w, 4), dtype=np.uint8)
    bar[:, :, 3] = 200  # semi-transparent black
    poster[:bar_h] = ((poster[:bar_h].astype(np.float32) * 0.2)).astype(np.uint8)
    poster[:bar_h, :, 3] = 255
    poster[-bar_h:] = ((poster[-bar_h:].astype(np.float32) * 0.2)).astype(np.uint8)
    poster[-bar_h:, :, 3] = 255

    if output.endswith('.png'):
        save_png(poster, output, rgb=True)
    else:
        from PIL import Image
        img = Image.fromarray(poster[:, :, :3])
        img.save(output, quality=90)
    print(f"\nSaved poster to {output} ({poster_w}x{poster_h}, {len(pairs)} buildings)")


def _composite(dst, src, x, y):
    """Alpha-composite src onto dst at position (x, y)."""
    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]

    # Clip
    sx1 = max(0, -x)
    sy1 = max(0, -y)
    dx1 = max(0, x)
    dy1 = max(0, y)
    w = min(sw - sx1, dw - dx1)
    h = min(sh - sy1, dh - dy1)
    if w <= 0 or h <= 0:
        return

    src_region = src[sy1:sy1+h, sx1:sx1+w]
    dst_region = dst[dy1:dy1+h, dx1:dx1+w]

    alpha = src_region[:, :, 3:4].astype(np.float32) / 255.0
    dst_region[:, :, :3] = (src_region[:, :, :3].astype(np.float32) * alpha +
                             dst_region[:, :, :3].astype(np.float32) * (1 - alpha)).astype(np.uint8)
    dst_region[:, :, 3] = np.maximum(dst_region[:, :, 3], src_region[:, :, 3])


def _draw_arrow(canvas, cx, cy):
    """Draw a '=>' arrow at (cx, cy) with outline for visibility on grass."""
    h, w = canvas.shape[:2]
    fill = np.array([255, 255, 255, 255], dtype=np.uint8)
    outline = np.array([30, 30, 30, 200], dtype=np.uint8)

    # Arrow shape: shaft (horizontal line) + head (triangle)
    shaft_len = 18
    head_size = 9
    thickness = 3

    def put(px, py, color):
        if 0 <= px < w and 0 <= py < h:
            canvas[py, px] = color

    # Draw outline first (1px border around everything)
    # Shaft outline
    for dx in range(-shaft_len, head_size + 2):
        for dy in range(-thickness - 1, thickness + 2):
            # Check if this pixel is on the outline border
            in_shaft = (-shaft_len <= dx <= 0) and (-thickness <= dy <= thickness)
            in_head = (0 < dx <= head_size) and abs(dy) <= head_size - dx + 1
            if not (in_shaft or in_head):
                # Check if adjacent to shape
                for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = dx + ddx, dy + ddy
                    s = (-shaft_len <= nx <= 0) and (-thickness <= ny <= thickness)
                    hd = (0 < nx <= head_size) and abs(ny) <= head_size - nx + 1
                    if s or hd:
                        put(cx + dx, cy + dy, outline)
                        break

    # Shaft fill
    for dx in range(-shaft_len, 1):
        for dy in range(-thickness, thickness + 1):
            put(cx + dx, cy + dy, fill)

    # Arrowhead fill (triangle pointing right)
    for dx in range(1, head_size + 1):
        span = head_size - dx + 1
        for dy in range(-span, span + 1):
            put(cx + dx, cy + dy, fill)


def main():
    parser = argparse.ArgumentParser(description="Generate before/after poster for mod page")
    parser.add_argument('--output', '-o', default='exports/poster.jpg', help='Output filename')
    parser.add_argument('--style', choices=list(STYLE_SETS.keys()),
                        help='Architecture style subset')
    parser.add_argument('--cols', type=int, default=3, help='Comparison pairs per row (default: 3)')
    parser.add_argument('--layer', choices=['main', 'shadow', 'damage', 'playercolor'],
                        default='main', help='Layer to render')
    parser.add_argument('--buildings', nargs='+',
                        help='Specific SLD filenames to include')
    parser.add_argument('--size', default='2560x1440',
                        help='Output size WxH (default: 2560x1440, 16:9)')
    parser.add_argument('--scale', choices=['x1', 'x2'], default='x1',
                        help='Sprite scale: x1 (standard) or x2 (UHD)')
    parser.add_argument('--mod-dir', help='Override mod directory path')
    args = parser.parse_args()

    parts = args.size.lower().split('x')
    target_size = (int(parts[0]), int(parts[1]))

    if args.buildings:
        buildings = [(f, os.path.splitext(f)[0], 'shore' if 'dock' in f.lower() else 'grass')
                     for f in args.buildings]
    elif args.style:
        buildings = STYLE_SETS[args.style]
    else:
        buildings = SHOWCASE

    output = args.output
    # Auto-name output for x2 scale
    if args.scale == 'x2' and output == 'exports/poster.jpg':
        output = 'exports/poster_x2.jpg'

    print(f"Generating {args.scale} poster with {len(buildings)} buildings...")
    make_poster(buildings, cols=args.cols, output=output, layer=args.layer,
                target_size=target_size, scale=args.scale, mod_dir=args.mod_dir)


if __name__ == '__main__':
    main()
