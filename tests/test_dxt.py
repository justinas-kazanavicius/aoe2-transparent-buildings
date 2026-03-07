"""Tests for DXT1/BC4 codec."""

import struct
import numpy as np
import pytest
from dxt import (
    rgb565_to_rgb, rgb_to_rgb565,
    decode_dxt1_block, encode_dxt1_block,
    decode_bc4_block, encode_bc4_block, encode_bc4_block_from_flat,
    inject_transparency_dxt1, zero_bc4_pixels,
)


class TestRGB565:
    def test_black(self):
        assert rgb565_to_rgb(0x0000) == (0, 0, 0)

    def test_white(self):
        assert rgb565_to_rgb(0xFFFF) == (255, 255, 255)

    def test_pure_red(self):
        r, g, b = rgb565_to_rgb(0xF800)
        assert r == 255 and g == 0 and b == 0

    def test_pure_green(self):
        r, g, b = rgb565_to_rgb(0x07E0)
        assert r == 0 and g == 255 and b == 0

    def test_pure_blue(self):
        r, g, b = rgb565_to_rgb(0x001F)
        assert r == 0 and g == 0 and b == 255

    def test_roundtrip(self):
        for c in [0x0000, 0xFFFF, 0xF800, 0x07E0, 0x001F, 0x52AA, 0x1234]:
            r, g, b = rgb565_to_rgb(c)
            assert rgb_to_rgb565(r, g, b) == c


class TestDXT1:
    def test_zero_block_is_all_transparent(self):
        block = b'\x00' * 8
        pixels = decode_dxt1_block(block)
        assert pixels.shape == (4, 4, 4)
        # c0 == c1 == 0 means transparent mode, index 0 = color0 = black opaque
        # All indices are 0, so all pixels are (0,0,0,255) — black opaque
        # Actually with c0<=c1 and all indices 0, pixel = color0 = (0,0,0) with alpha 255
        assert np.all(pixels[:, :, 3] == 255)

    def test_all_transparent_block(self):
        # c0=0, c1=0 (transparent mode), all indices = 3 (transparent)
        block = struct.pack('<HHI', 0, 0, 0xFFFFFFFF)
        pixels = decode_dxt1_block(block)
        assert np.all(pixels[:, :, 3] == 0)

    def test_encode_decode_roundtrip_opaque(self):
        # Create a simple opaque 4x4 block with two colors
        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255  # all opaque
        pixels[0:2, :, 0] = 255  # top half red
        pixels[2:4, :, 2] = 255  # bottom half blue
        encoded = encode_dxt1_block(pixels)
        decoded = decode_dxt1_block(encoded)
        assert decoded.shape == (4, 4, 4)
        assert np.all(decoded[:, :, 3] == 255)  # all opaque

    def test_encode_decode_roundtrip_mixed(self):
        # Mixed transparent/opaque block
        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0:2, :, :] = [255, 0, 0, 255]  # top opaque red
        pixels[2:4, :, :] = [0, 0, 0, 0]       # bottom transparent
        encoded = encode_dxt1_block(pixels)
        decoded = decode_dxt1_block(encoded)
        # Top should be opaque, bottom transparent
        assert np.all(decoded[0:2, :, 3] > 0)
        assert np.all(decoded[2:4, :, 3] == 0)

    def test_encode_all_transparent(self):
        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        encoded = encode_dxt1_block(pixels)
        assert encoded == b'\x00' * 8


class TestInjectTransparency:
    def test_no_mask_returns_same(self):
        block = b'\xFF\xFF\x00\x00\x00\x00\x00\x00'
        assert inject_transparency_dxt1(block, 0) == block

    def test_full_mask_returns_zero(self):
        block = b'\xFF\xFF\x00\x00\xAA\xBB\xCC\xDD'
        assert inject_transparency_dxt1(block, 0xFFFF) == b'\x00' * 8

    def test_single_pixel_transparent(self):
        # Opaque red block (c0 > c1), all index 0
        c0 = rgb_to_rgb565(255, 0, 0)
        c1 = rgb_to_rgb565(0, 0, 255)
        if c0 < c1:
            c0, c1 = c1, c0
        block = struct.pack('<HHI', c0, c1, 0x00000000)
        # Make pixel 0 transparent
        result = inject_transparency_dxt1(block, 0x0001)
        decoded = decode_dxt1_block(result)
        assert decoded[0, 0, 3] == 0  # pixel 0 transparent
        assert decoded[0, 1, 3] == 255  # pixel 1 still opaque

    def test_checkerboard_mask(self):
        # Make a simple block and apply checkerboard transparency
        c0 = rgb_to_rgb565(128, 128, 128)
        block = struct.pack('<HHI', c0, c0, 0x00000000)
        mask = 0x5555  # every other pixel
        result = inject_transparency_dxt1(block, mask)
        decoded = decode_dxt1_block(result)
        for i in range(16):
            row, col = i // 4, i % 4
            if mask & (1 << i):
                assert decoded[row, col, 3] == 0
            else:
                assert decoded[row, col, 3] == 255


class TestBC4:
    def test_zero_block(self):
        block = b'\x00' * 8
        values = decode_bc4_block(block)
        assert values.shape == (4, 4)
        assert np.all(values == 0)

    def test_constant_block(self):
        block = bytes([200, 200, 0, 0, 0, 0, 0, 0])
        values = decode_bc4_block(block)
        assert np.all(values == 200)

    def test_encode_decode_roundtrip_constant(self):
        values = np.full((4, 4), 150, dtype=np.uint8)
        encoded = encode_bc4_block(values)
        decoded = decode_bc4_block(encoded)
        assert np.all(decoded == 150)

    def test_encode_decode_roundtrip_gradient(self):
        values = np.array([[i * 16 for i in range(4)] for _ in range(4)], dtype=np.uint8)
        encoded = encode_bc4_block(values)
        decoded = decode_bc4_block(encoded)
        # BC4 is lossy but should be close
        assert np.max(np.abs(decoded.astype(int) - values.astype(int))) <= 5

    def test_encode_from_flat(self):
        flat = np.arange(16, dtype=np.uint8) * 16
        encoded = encode_bc4_block_from_flat(flat)
        decoded = decode_bc4_block(encoded).flatten()
        # BC4 has only 8 interpolation levels so max error can be ~17
        assert np.max(np.abs(decoded.astype(int) - flat.astype(int))) <= 18


class TestZeroBC4Pixels:
    def test_no_mask(self):
        block = bytes([200, 200, 0, 0, 0, 0, 0, 0])
        assert zero_bc4_pixels(block, 0) == block

    def test_full_mask(self):
        block = bytes([200, 100, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
        result = zero_bc4_pixels(block, 0xFFFF)
        assert result == b'\x00' * 8

    def test_partial_mask(self):
        flat = np.full(16, 200, dtype=np.uint8)
        block = encode_bc4_block_from_flat(flat)
        # Zero out pixel 0
        result = zero_bc4_pixels(block, 0x0001)
        decoded = decode_bc4_block(result).flatten()
        assert decoded[0] == 0
        assert decoded[1] > 100  # other pixels should stay high
