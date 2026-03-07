"""
DXT1 (BC1) and DXT4 (BC4) block compression codec for AoE2 DE SLD sprites.

DXT1: 8 bytes -> 4x4 RGBA pixels (main graphics + damage masks)
DXT4/BC4: 8 bytes -> 4x4 single-channel pixels (shadows + player color)

Includes fast in-place transparency injection for checkerboard dithering.
"""

import struct
import numpy as np

# Pre-allocated zero block (8 bytes of 0x00) shared across functions
_ZERO_BLOCK = b'\x00\x00\x00\x00\x00\x00\x00\x00'


def rgb565_to_rgb(c):
    """Decode RGB565 uint16 to (r, g, b) tuple in 0-255 range."""
    r = ((c >> 11) & 0x1F) * 255 // 31
    g = ((c >> 5) & 0x3F) * 255 // 63
    b = (c & 0x1F) * 255 // 31
    return (r, g, b)


def rgb_to_rgb565(r, g, b):
    """Encode (r, g, b) 0-255 to RGB565 uint16."""
    r5 = (r * 31 + 127) // 255
    g6 = (g * 63 + 127) // 255
    b5 = (b * 31 + 127) // 255
    return (r5 << 11) | (g6 << 5) | b5


# -------------------------------------------------------------------------
# Fast DXT1 transparency injection (no full decode/re-encode needed)
# -------------------------------------------------------------------------

# Index remapping table for when we swap color0 and color1 to enable
# transparent mode. In opaque mode (c0 > c1):
#   0 -> color0, 1 -> color1, 2 -> 2/3*c0 + 1/3*c1, 3 -> 1/3*c0 + 2/3*c1
# In transparent mode (c0 <= c1):
#   0 -> color0, 1 -> color1, 2 -> 1/2*(c0+c1), 3 -> transparent
# After swapping endpoints, old color0 -> new color1 and old color1 -> new color0
# So: old_idx 0 (was c0) -> new_idx 1 (now c1); old_idx 1 (was c1) -> new_idx 0 (now c0)
# For indices 2 and 3 (interpolated), the interpolation changes slightly
# (from 1/3,2/3 to 1/2), so we remap: old 2 -> new 2 (approximate), old 3 -> new 2
_OPAQUE_TO_TRANSPARENT_REMAP = [1, 0, 2, 2]


def inject_transparency_dxt1(block_data, mask_bits):
    """
    Inject transparency into a DXT1 block by modifying pixel indices.

    This is much faster than full decode/re-encode because it preserves
    the original color endpoints and only modifies the index bits.

    Args:
        block_data: bytes, 8-byte DXT1 block
        mask_bits: int, 16-bit bitmask where bit i = pixel i should be transparent

    Returns:
        bytes: modified 8-byte DXT1 block
    """
    # mask_bits == 0 should be filtered by caller, but guard anyway
    if mask_bits == 0:
        return block_data

    # All 16 pixels transparent
    if mask_bits == 0xFFFF:
        return _ZERO_BLOCK

    color0, color1, indices = struct.unpack_from('<HHI', block_data, 0)

    # Need to switch to transparent mode if currently in opaque mode
    if color0 > color1:
        # Currently opaque mode, need to swap to transparent mode
        color0, color1 = color1, color0
        # Remap all existing indices using lookup table
        # Build remapped indices in one pass using bit extraction
        new_indices = 0
        for i in range(16):
            old_idx = (indices >> (2 * i)) & 0x3
            new_indices |= (_OPAQUE_TO_TRANSPARENT_REMAP[old_idx] << (2 * i))
        indices = new_indices
    # else: already in transparent mode (color0 <= color1), indices are fine

    # Set index 3 (transparent) for all masked pixels
    # Build a 32-bit mask where each masked pixel gets 0b11 in its 2-bit slot
    trans_bits = 0
    m = mask_bits
    while m:
        i = (m & -m).bit_length() - 1  # lowest set bit
        trans_bits |= (0x3 << (2 * i))
        m &= m - 1  # clear lowest set bit

    # For each masked pixel: clear its 2-bit slot, then set to 0b11 (index 3)
    indices = (indices & ~trans_bits) | trans_bits

    return struct.pack('<HHI', color0, color1, indices)


def zero_bc4_pixels(block_data, mask_bits):
    """
    Zero out specific pixels in a BC4 block.

    For shadow/player color layers, we need to zero out pixels that
    correspond to transparent pixels in the main graphic layer.

    This does a full decode/re-encode since BC4 uses 3-bit indices
    with interpolation that can't be trivially patched.

    Args:
        block_data: bytes, 8-byte BC4 block
        mask_bits: int, 16-bit bitmask where bit i = zero this pixel

    Returns:
        bytes: modified 8-byte BC4 block
    """
    # mask_bits == 0 should be filtered by caller, but guard anyway
    if mask_bits == 0:
        return block_data

    # All 16 pixels zeroed
    if mask_bits == 0xFFFF:
        return _ZERO_BLOCK

    # Decode
    values = decode_bc4_block(block_data)
    flat = values.flatten()

    # Zero masked pixels
    m = mask_bits
    while m:
        i = (m & -m).bit_length() - 1
        flat[i] = 0
        m &= m - 1

    # Re-encode
    return encode_bc4_block_from_flat(flat)


# -------------------------------------------------------------------------
# Full decode/encode (used for BC4 and edge cases)
# -------------------------------------------------------------------------

def decode_dxt1_block(data):
    """Decode an 8-byte DXT1 block into a 4x4 RGBA numpy array.

    Returns np.ndarray of shape (4, 4, 4) dtype=uint8 (RGBA).
    """
    color0, color1, indices = struct.unpack_from('<HHI', data, 0)

    r0, g0, b0 = rgb565_to_rgb(color0)
    r1, g1, b1 = rgb565_to_rgb(color1)

    lookup = [(r0, g0, b0, 255),
              (r1, g1, b1, 255)]

    if color0 > color1:
        lookup.append((
            (2 * r0 + r1 + 1) // 3,
            (2 * g0 + g1 + 1) // 3,
            (2 * b0 + b1 + 1) // 3,
            255
        ))
        lookup.append((
            (r0 + 2 * r1 + 1) // 3,
            (g0 + 2 * g1 + 1) // 3,
            (b0 + 2 * b1 + 1) // 3,
            255
        ))
    else:
        lookup.append((
            (r0 + r1) // 2,
            (g0 + g1) // 2,
            (b0 + b1) // 2,
            255
        ))
        lookup.append((0, 0, 0, 0))

    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    for i in range(16):
        idx = (indices >> (2 * i)) & 0x3
        row = i // 4
        col = i % 4
        pixels[row, col] = lookup[idx]

    return pixels


def encode_dxt1_block(pixels):
    """Encode a 4x4 RGBA pixel array into an 8-byte DXT1 block.

    If all pixels are transparent, returns a zero block.
    If some pixels are transparent, uses transparent mode (color0 <= color1).
    """
    alpha = pixels[:, :, 3].flatten()
    opaque_mask = alpha > 127

    if not np.any(opaque_mask):
        return b'\x00' * 8

    if np.all(opaque_mask):
        return _encode_dxt1_opaque(pixels)
    else:
        return _encode_dxt1_transparent(pixels, opaque_mask)


def _find_endpoint_colors(pixels_flat, mask):
    """Find two endpoint colors from opaque pixels using bounding box approach."""
    opaque_pixels = pixels_flat[mask, :3].astype(np.int32)

    if len(opaque_pixels) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    if len(opaque_pixels) == 1:
        r, g, b = opaque_pixels[0]
        c = rgb_to_rgb565(r, g, b)
        r2, g2, b2 = rgb565_to_rgb(c)
        return c, c, r2, g2, b2, r2, g2, b2

    mins = opaque_pixels.min(axis=0)
    maxs = opaque_pixels.max(axis=0)
    ranges = maxs - mins
    axis = np.argmax(ranges)
    values = opaque_pixels[:, axis]
    idx_min = np.argmin(values)
    idx_max = np.argmax(values)

    ep0 = opaque_pixels[idx_max]
    ep1 = opaque_pixels[idx_min]

    c0 = rgb_to_rgb565(ep0[0], ep0[1], ep0[2])
    c1 = rgb_to_rgb565(ep1[0], ep1[1], ep1[2])

    r0, g0, b0 = rgb565_to_rgb(c0)
    r1, g1, b1 = rgb565_to_rgb(c1)

    return c0, c1, r0, g0, b0, r1, g1, b1


def _encode_dxt1_opaque(pixels):
    """Encode all-opaque 4x4 block in opaque mode (color0 > color1)."""
    flat = pixels.reshape(16, 4)
    mask = np.ones(16, dtype=bool)

    c0, c1, r0, g0, b0, r1, g1, b1 = _find_endpoint_colors(flat, mask)

    if c0 < c1:
        c0, c1 = c1, c0
        r0, g0, b0, r1, g1, b1 = r1, g1, b1, r0, g0, b0
    elif c0 == c1:
        if c0 < 0xFFFF:
            c0 = c0 + 1
            r0, g0, b0 = rgb565_to_rgb(c0)
        else:
            c1 = c1 - 1
            r1, g1, b1 = rgb565_to_rgb(c1)

    palette = np.array([
        [r0, g0, b0],
        [r1, g1, b1],
        [(2 * r0 + r1 + 1) // 3, (2 * g0 + g1 + 1) // 3, (2 * b0 + b1 + 1) // 3],
        [(r0 + 2 * r1 + 1) // 3, (g0 + 2 * g1 + 1) // 3, (b0 + 2 * b1 + 1) // 3],
    ], dtype=np.int32)

    indices = 0
    for i in range(16):
        pixel_rgb = flat[i, :3].astype(np.int32)
        dists = np.sum((palette - pixel_rgb) ** 2, axis=1)
        best = int(np.argmin(dists))
        indices |= (best << (2 * i))

    return struct.pack('<HHI', c0, c1, indices)


def _encode_dxt1_transparent(pixels, opaque_mask):
    """Encode mixed transparent/opaque block in transparent mode (color0 <= color1)."""
    flat = pixels.reshape(16, 4)

    c0, c1, r0, g0, b0, r1, g1, b1 = _find_endpoint_colors(flat, opaque_mask)

    if c0 > c1:
        c0, c1 = c1, c0
        r0, g0, b0, r1, g1, b1 = r1, g1, b1, r0, g0, b0

    palette = np.array([
        [r0, g0, b0],
        [r1, g1, b1],
        [(r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2],
    ], dtype=np.int32)

    indices = 0
    for i in range(16):
        if not opaque_mask[i]:
            indices |= (3 << (2 * i))
        else:
            pixel_rgb = flat[i, :3].astype(np.int32)
            dists = np.sum((palette - pixel_rgb) ** 2, axis=1)
            best = int(np.argmin(dists))
            indices |= (best << (2 * i))

    return struct.pack('<HHI', c0, c1, indices)


def decode_bc4_block(data):
    """Decode an 8-byte BC4 block into a 4x4 single-channel numpy array.

    Returns np.ndarray of shape (4, 4) dtype=uint8.
    """
    ref0 = data[0]
    ref1 = data[1]

    if ref0 > ref1:
        lookup = [ref0, ref1]
        for i in range(6):
            lookup.append(((6 - i) * ref0 + (1 + i) * ref1 + 3) // 7)
    else:
        lookup = [ref0, ref1]
        for i in range(4):
            lookup.append(((4 - i) * ref0 + (1 + i) * ref1 + 2) // 5)
        lookup.append(0)
        lookup.append(255)

    index_bytes = data[2:8]
    pixels = np.zeros((4, 4), dtype=np.uint8)

    for half in range(2):
        value = int.from_bytes(index_bytes[half * 3: half * 3 + 3], 'little')
        for p in range(8):
            idx = value & 0x7
            pixel_idx = half * 8 + p
            row = pixel_idx // 4
            col = pixel_idx % 4
            pixels[row, col] = lookup[idx]
            value >>= 3

    return pixels


def encode_bc4_block(values):
    """Encode a 4x4 single-channel array into an 8-byte BC4 block."""
    return encode_bc4_block_from_flat(values.flatten())


def encode_bc4_block_from_flat(flat):
    """Encode 16 single-channel values into an 8-byte BC4 block.

    Args:
        flat: array-like of 16 uint8 values
    """
    vmin = int(min(flat))
    vmax = int(max(flat))

    if vmax == vmin:
        return bytes([vmax, vmax, 0, 0, 0, 0, 0, 0])

    ref0 = vmax
    ref1 = vmin

    lookup = [ref0, ref1]
    for i in range(6):
        lookup.append(((6 - i) * ref0 + (1 + i) * ref1 + 3) // 7)

    indices = []
    for v in flat:
        best_idx = 0
        best_dist = abs(int(v) - lookup[0])
        for j in range(1, 8):
            d = abs(int(v) - lookup[j])
            if d < best_dist:
                best_dist = d
                best_idx = j
        indices.append(best_idx)

    index_bytes = bytearray(6)
    for half in range(2):
        value = 0
        for p in range(8):
            value |= (indices[half * 8 + p] << (3 * p))
        index_bytes[half * 3] = value & 0xFF
        index_bytes[half * 3 + 1] = (value >> 8) & 0xFF
        index_bytes[half * 3 + 2] = (value >> 16) & 0xFF

    return bytes([ref0, ref1]) + bytes(index_bytes)
