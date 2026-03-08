"""Convert SLD sprite files to PNG or animated GIF for visual inspection.

Usage:
    uv run sld-to-png b_west_house_age2_x1.sld              # First frame as PNG
    uv run sld-to-png b_west_mill_age2_x1.sld --frame 5     # Specific frame
    uv run sld-to-png b_west_mill_age2_x1.sld --all-frames  # All frames as separate PNGs
    uv run sld-to-png b_west_mill_age2_x1.sld --sheet       # All frames in a sprite sheet
    uv run sld-to-png b_west_mill_age2_x1.sld --gif         # Animated GIF
    uv run sld-to-png b_west_mill_age2_x1.sld --gif --fps 30
    uv run sld-to-png b_west_house_age2_x1.sld --layer shadow
    uv run sld-to-png b_west_house_age2_x1.sld --mod        # Read from mod output dir
    uv run sld-to-png b_west_house_age2_x1.sld --compare    # Side-by-side original vs mod
"""
import os
import sys
import argparse
import struct
import zlib

import numpy as np

from paths import get_graphics_dir, get_mod_graphics_dir
from sld import (
    parse_sld, get_layer, get_block_positions,
    LAYER_MAIN, LAYER_SHADOW, LAYER_DAMAGE, LAYER_PLAYERCOLOR,
)
from dxt import decode_dxt1_block, decode_bc4_block

LAYER_NAMES = {
    LAYER_MAIN: 'main',
    LAYER_SHADOW: 'shadow',
    LAYER_DAMAGE: 'damage',
    LAYER_PLAYERCOLOR: 'playercolor',
}

LAYER_BY_NAME = {v: k for k, v in LAYER_NAMES.items()}

# AoE2 DE runs at ~20fps for building animations
DEFAULT_FPS = 20


def render_frame(frame, layer_type=None):
    """Render a frame to an RGBA numpy array.

    For DXT1 layers (main, damage), renders full RGBA.
    For BC4 layers (shadow, playercolor), renders as grayscale with alpha.
    """
    if layer_type is None:
        layer_type = LAYER_MAIN

    layer = get_layer(frame, layer_type)
    if layer is None:
        return None

    w = frame.canvas_width
    h = frame.canvas_height
    is_dxt1 = layer_type in (LAYER_MAIN, LAYER_DAMAGE)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    positions = get_block_positions(layer, frame)
    for block_idx, bx, by in positions:
        if block_idx >= len(layer.blocks):
            break

        if is_dxt1:
            pixels = decode_dxt1_block(layer.blocks[block_idx])
            for r in range(4):
                for c in range(4):
                    py, px = by + r, bx + c
                    if 0 <= py < h and 0 <= px < w:
                        canvas[py, px] = pixels[r, c]
        else:
            values = decode_bc4_block(layer.blocks[block_idx])
            for r in range(4):
                for c in range(4):
                    py, px = by + r, bx + c
                    if 0 <= py < h and 0 <= px < w:
                        v = values[r, c]
                        canvas[py, px] = [v, v, v, 255 if v > 0 else 0]

    return canvas


# ---------------------------------------------------------------------------
# PNG writer (no PIL needed)
# ---------------------------------------------------------------------------

def _write_png_chunk(f, chunk_type, data):
    f.write(struct.pack('>I', len(data)))
    f.write(chunk_type)
    f.write(data)
    f.write(struct.pack('>I', zlib.crc32(chunk_type + data) & 0xFFFFFFFF))


def save_png(canvas, filepath, rgb=False):
    """Save numpy array as PNG. RGBA by default, RGB if rgb=True."""
    h, w = canvas.shape[:2]
    if rgb:
        # Color type 2 = RGB
        color_type = 2
        img = canvas[:, :, :3] if canvas.shape[2] >= 3 else canvas
    else:
        # Color type 6 = RGBA
        color_type = 6
        img = canvas
    with open(filepath, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        _write_png_chunk(f, b'IHDR', struct.pack('>IIBBBBB', w, h, 8, color_type, 0, 0, 0))
        raw = bytearray()
        for y in range(h):
            raw.append(0)
            raw.extend(img[y].tobytes())
        _write_png_chunk(f, b'IDAT', zlib.compress(bytes(raw), 9))
        _write_png_chunk(f, b'IEND', b'')


# ---------------------------------------------------------------------------
# GIF writer (no PIL needed) - supports animation and transparency
# ---------------------------------------------------------------------------

def _quantize_frame(rgba, bg_color=(34, 34, 34)):
    """Quantize RGBA canvas to 256-color palette with transparency.

    Uses popularity-based quantization: reserve index 0 for transparent,
    pick the 255 most frequently used colors (after reducing to 15-bit).
    Remaining colors map to closest palette entry.
    """
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    opaque = alpha > 127

    # Composite onto bg for transparent pixels
    composited = np.zeros((h, w, 3), dtype=np.uint8)
    composited[opaque] = rgba[opaque, :3]
    composited[~opaque] = bg_color

    # Reduce to 15-bit color (5-5-5) for grouping
    r5 = composited[:, :, 0] >> 3
    g5 = composited[:, :, 1] >> 3
    b5 = composited[:, :, 2] >> 3
    color_keys = (r5.astype(np.uint32) << 10) | (g5.astype(np.uint32) << 5) | b5.astype(np.uint32)

    # Find unique colors and pick 255 most popular
    flat_keys = color_keys.ravel()
    opaque_flat = opaque.ravel()
    opaque_keys = flat_keys[opaque_flat]

    if len(opaque_keys) == 0:
        # All transparent
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[0] = bg_color
        return np.zeros((h, w), dtype=np.uint8), palette

    unique_keys, counts = np.unique(opaque_keys, return_counts=True)
    # Sort by popularity (most common first)
    order = np.argsort(-counts)
    unique_keys = unique_keys[order]

    n_colors = min(len(unique_keys), 255)
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[0] = bg_color

    # Build palette from top N colors (expand 5-bit back to 8-bit)
    palette_rgb = np.zeros((n_colors, 3), dtype=np.uint8)
    for i in range(n_colors):
        k = int(unique_keys[i])
        r = ((k >> 10) & 0x1F) * 255 // 31
        g = ((k >> 5) & 0x1F) * 255 // 31
        b = (k & 0x1F) * 255 // 31
        palette[i + 1] = [r, g, b]
        palette_rgb[i] = [r, g, b]

    # Map all unique keys to palette indices
    # For top N: direct mapping. For overflow: find nearest palette entry.
    key_to_idx = {}
    for i in range(n_colors):
        key_to_idx[int(unique_keys[i])] = i + 1

    # For overflow colors, find nearest in palette by RGB distance
    if len(unique_keys) > 255:
        palette_arr = palette_rgb.astype(np.int32)
        for i in range(255, len(unique_keys)):
            k = int(unique_keys[i])
            r = ((k >> 10) & 0x1F) * 255 // 31
            g = ((k >> 5) & 0x1F) * 255 // 31
            b = (k & 0x1F) * 255 // 31
            dists = ((palette_arr[:, 0] - r) ** 2 +
                     (palette_arr[:, 1] - g) ** 2 +
                     (palette_arr[:, 2] - b) ** 2)
            key_to_idx[k] = int(np.argmin(dists)) + 1

    # Build index image
    indices = np.zeros((h, w), dtype=np.uint8)
    for k, idx in key_to_idx.items():
        indices[color_keys == k] = idx
    indices[~opaque] = 0

    return indices, palette


def save_gif(canvases, filepath, fps=DEFAULT_FPS, bg_color=(34, 34, 34)):
    """Save list of RGBA canvases as animated GIF with transparency."""
    if not canvases:
        return

    h, w = canvases[0].shape[:2]
    delay = max(1, round(100 / fps))  # GIF delay in centiseconds

    with open(filepath, 'wb') as f:
        # --- Header ---
        f.write(b'GIF89a')

        # Logical screen descriptor (no global color table)
        f.write(struct.pack('<HH', w, h))
        f.write(bytes([0x00, 0, 0]))  # no GCT, bg=0, aspect=0

        # Netscape looping extension (loop forever)
        f.write(b'\x21\xFF\x0BNETSCAPE2.0')
        f.write(b'\x03\x01')
        f.write(struct.pack('<H', 0))  # loop count 0 = infinite
        f.write(b'\x00')  # block terminator

        for canvas in canvases:
            # Ensure canvas matches expected dimensions
            ch, cw = canvas.shape[:2]
            if ch != h or cw != w:
                padded = np.zeros((h, w, 4), dtype=np.uint8)
                mh, mw = min(ch, h), min(cw, w)
                padded[:mh, :mw] = canvas[:mh, :mw]
                canvas = padded

            indices, palette = _quantize_frame(canvas, bg_color)

            # Graphic control extension (with transparency)
            f.write(b'\x21\xF9\x04')
            f.write(bytes([0x09]))  # dispose=restore to bg, transparent flag set
            f.write(struct.pack('<H', delay))
            f.write(bytes([0]))  # transparent index = 0
            f.write(b'\x00')

            # Image descriptor with local color table
            f.write(b'\x2C')
            f.write(struct.pack('<HHHH', 0, 0, w, h))
            f.write(bytes([0x87]))  # local color table, 256 entries (2^(7+1))

            # Local color table (256 * 3 bytes)
            f.write(palette.tobytes())

            # LZW-compressed image data
            min_code_size = 8
            f.write(bytes([min_code_size]))

            # LZW compress
            compressed = _lzw_compress(indices.ravel(), min_code_size)

            # Write in sub-blocks of max 255 bytes
            pos = 0
            while pos < len(compressed):
                chunk = compressed[pos:pos + 255]
                f.write(bytes([len(chunk)]))
                f.write(chunk)
                pos += 255
            f.write(b'\x00')  # block terminator

        # Trailer
        f.write(b'\x3B')


def _lzw_compress(pixels, min_code_size):
    """LZW compress pixel data for GIF."""
    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1

    code_size = min_code_size + 1
    next_code = eoi_code + 1
    max_code = (1 << code_size)

    # Initialize code table
    code_table = {}
    for i in range(clear_code):
        code_table[(i,)] = i

    output_bits = []

    def emit(code):
        output_bits.append((code, code_size))

    emit(clear_code)

    buffer = (int(pixels[0]),)
    for pixel in pixels[1:]:
        extended = buffer + (int(pixel),)
        if extended in code_table:
            buffer = extended
        else:
            emit(code_table[buffer])
            if next_code < 4096:
                code_table[extended] = next_code
                next_code += 1
                if next_code > max_code and code_size < 12:
                    code_size += 1
                    max_code = 1 << code_size
            else:
                # Table full, emit clear code and reset
                emit(clear_code)
                code_table = {}
                for i in range(clear_code):
                    code_table[(i,)] = i
                code_size = min_code_size + 1
                next_code = eoi_code + 1
                max_code = 1 << code_size
            buffer = (int(pixel),)

    emit(code_table[buffer])
    emit(eoi_code)

    # Pack bits into bytes (LSB first)
    result = bytearray()
    current_byte = 0
    bits_in_byte = 0
    for code, size in output_bits:
        current_byte |= (code << bits_in_byte)
        bits_in_byte += size
        while bits_in_byte >= 8:
            result.append(current_byte & 0xFF)
            current_byte >>= 8
            bits_in_byte -= 8
    if bits_in_byte > 0:
        result.append(current_byte & 0xFF)

    return bytes(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sheet(canvases, cols=10):
    """Combine multiple RGBA canvases into a sprite sheet."""
    if not canvases:
        return None
    h, w = canvases[0].shape[:2]
    rows = (len(canvases) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 4), dtype=np.uint8)
    for i, canvas in enumerate(canvases):
        r, c = divmod(i, cols)
        ch, cw = canvas.shape[:2]
        sheet[r*h:r*h+ch, c*w:c*w+cw] = canvas[:h, :w]
    return sheet


def render_all_frames(sld, layer_type, max_frames=None):
    """Render all (or up to max_frames) frames from an SLD."""
    limit = sld.num_frames
    if max_frames:
        limit = min(limit, max_frames)
    canvases = []
    for i in range(limit):
        canvas = render_frame(sld.frames[i], layer_type)
        if canvas is not None:
            canvases.append(canvas)
    return canvases


def side_by_side(canvas_a, canvas_b, gap=4, bg_color=(34, 34, 34)):
    """Place two RGBA canvases side by side with a gap."""
    ha, wa = canvas_a.shape[:2]
    hb, wb = canvas_b.shape[:2]
    h = max(ha, hb)
    w = wa + gap + wb
    combined = np.zeros((h, w, 4), dtype=np.uint8)
    # Fill gap with bg
    combined[:, wa:wa+gap, :3] = bg_color
    combined[:, wa:wa+gap, 3] = 255
    combined[:ha, :wa] = canvas_a
    combined[:hb, wa+gap:] = canvas_b
    return combined


def main():
    parser = argparse.ArgumentParser(description="Convert SLD sprites to PNG/GIF")
    parser.add_argument('file', help='SLD filename')
    parser.add_argument('--frame', type=int, default=0, help='Frame index (default: 0)')
    parser.add_argument('--all-frames', action='store_true', help='Export all frames as PNGs')
    parser.add_argument('--sheet', action='store_true', help='All frames in a sprite sheet')
    parser.add_argument('--sheet-cols', type=int, default=10, help='Columns in sprite sheet')
    parser.add_argument('--gif', action='store_true', help='Export as animated GIF')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS, help=f'GIF framerate (default: {DEFAULT_FPS})')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to export')
    parser.add_argument('--layer', choices=['main', 'shadow', 'damage', 'playercolor'],
                        default='main', help='Layer to render (default: main)')
    parser.add_argument('--output', '-o', help='Output path (default: auto)')
    parser.add_argument('--input', '-i', help='Custom input file path')
    parser.add_argument('--mod', action='store_true', help='Read from mod output directory')
    parser.add_argument('--compare', action='store_true',
                        help='Side-by-side original vs mod (PNG or GIF)')
    args = parser.parse_args()

    # Resolve input path
    if args.input:
        filepath = args.input
    elif args.mod:
        filepath = os.path.join(get_mod_graphics_dir(), args.file)
    else:
        filepath = os.path.join(get_graphics_dir(), args.file)

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    with open(filepath, 'rb') as f:
        data = f.read()

    sld = parse_sld(data)
    layer_type = LAYER_BY_NAME[args.layer]
    basename = os.path.splitext(args.file)[0]

    print(f"{args.file}: {sld.num_frames} frames")

    # --compare: side-by-side original vs mod
    if args.compare:
        orig_path = os.path.join(get_graphics_dir(), args.file)
        mod_path = os.path.join(get_mod_graphics_dir(), args.file)
        for p, label in [(orig_path, 'original'), (mod_path, 'mod')]:
            if not os.path.exists(p):
                print(f"{label} not found: {p}")
                sys.exit(1)

        with open(orig_path, 'rb') as f:
            orig_sld = parse_sld(f.read())
        with open(mod_path, 'rb') as f:
            mod_sld = parse_sld(f.read())

        if args.gif:
            orig_frames = render_all_frames(orig_sld, layer_type, args.max_frames)
            mod_frames = render_all_frames(mod_sld, layer_type, args.max_frames)
            n = min(len(orig_frames), len(mod_frames))
            combined = [side_by_side(orig_frames[i], mod_frames[i]) for i in range(n)]
            out = args.output or f"{basename}_compare.gif"
            save_gif(combined, out, fps=args.fps)
            print(f"Saved {n}-frame comparison GIF to {out}")
        else:
            fi = args.frame
            orig_canvas = render_frame(orig_sld.frames[fi], layer_type)
            mod_canvas = render_frame(mod_sld.frames[fi], layer_type)
            if orig_canvas is None or mod_canvas is None:
                print("Could not render frame.")
                sys.exit(1)
            combined = side_by_side(orig_canvas, mod_canvas)
            out = args.output or f"{basename}_compare_f{fi:03d}.png"
            save_png(combined, out)
            print(f"Saved comparison to {out} ({combined.shape[1]}x{combined.shape[0]})")
        return

    # --gif: animated GIF
    if args.gif:
        print(f"Rendering frames...", end=" ", flush=True)
        canvases = render_all_frames(sld, layer_type, args.max_frames)
        print(f"{len(canvases)} frames.")
        if not canvases:
            print("No frames rendered.")
            sys.exit(1)
        out = args.output or f"{basename}_{args.layer}.gif"
        print(f"Encoding GIF...", end=" ", flush=True)
        save_gif(canvases, out, fps=args.fps)
        print(f"done.")
        print(f"Saved {out} ({canvases[0].shape[1]}x{canvases[0].shape[0]}, {len(canvases)} frames, {args.fps}fps)")
        return

    # --sheet: sprite sheet
    if args.sheet:
        canvases = render_all_frames(sld, layer_type, args.max_frames)
        if not canvases:
            print("No frames rendered.")
            sys.exit(1)
        sheet = make_sheet(canvases, cols=args.sheet_cols)
        out = args.output or f"{basename}_{args.layer}_sheet.png"
        save_png(sheet, out)
        print(f"Saved {len(canvases)}-frame sheet to {out} ({sheet.shape[1]}x{sheet.shape[0]})")
        return

    # --all-frames: individual PNGs
    if args.all_frames:
        limit = args.max_frames or sld.num_frames
        for i in range(min(limit, sld.num_frames)):
            canvas = render_frame(sld.frames[i], layer_type)
            if canvas is not None:
                out = f"{basename}_{args.layer}_f{i:03d}.png"
                save_png(canvas, out)
                print(f"  Frame {i} -> {out}")
        print(f"Exported {min(limit, sld.num_frames)} frames.")
        return

    # Single frame PNG (default)
    if args.frame >= sld.num_frames:
        print(f"Frame {args.frame} out of range (0-{sld.num_frames-1})")
        sys.exit(1)
    canvas = render_frame(sld.frames[args.frame], layer_type)
    if canvas is None:
        sys.exit(1)
    out = args.output or f"{basename}_{args.layer}_f{args.frame:03d}.png"
    save_png(canvas, out)
    print(f"Saved {out} ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == '__main__':
    main()
