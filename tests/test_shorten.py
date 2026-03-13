"""Tests for building shortening."""

import struct
import numpy as np
import pytest

from dxt import decode_dxt1_block, encode_dxt1_block, decode_bc4_block, encode_bc4_block
from sld import (
    parse_sld, write_sld, get_layer, get_block_positions,
    LAYER_MAIN, LAYER_SHADOW, LAYER_DAMAGE, LAYER_PLAYERCOLOR,
)
from shorten import (
    shorten_sld, _shorten_pixels, _find_bounds, _encode_region,
    _resolve_frames, _overlay_blocks,
)


def _make_opaque_dxt1_block(r, g, b):
    """Create a DXT1 block where all 16 pixels are the given color."""
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[:, :, 0] = r
    pixels[:, :, 1] = g
    pixels[:, :, 2] = b
    pixels[:, :, 3] = 255
    return encode_dxt1_block(pixels)


def _make_bc4_block(value):
    """Create a BC4 block where all 16 pixels have the given value."""
    return encode_bc4_block(np.full((4, 4), value, dtype=np.uint8))


def _make_sld_bytes(canvas_w, canvas_h, cx, cy, layer_x1, layer_y1, layer_x2, layer_y2,
                    main_blocks, frame_type=LAYER_MAIN, extra_layers=None, num_frames=1,
                    delta_blocks_list=None):
    """Build SLD binary with configurable geometry."""
    buf = bytearray()
    buf += struct.pack('<4sHHHHI', b'SLDX', 4, num_frames, 0, 0x0010, 0x000000FF)

    for fi in range(num_frames):
        ft = frame_type
        buf += struct.pack('<HHhhBBH', canvas_w, canvas_h, cx, cy, ft, 0, fi)

        if fi == 0 or delta_blocks_list is None:
            blocks = main_blocks
            flag1 = 0
        else:
            blocks = delta_blocks_list[fi - 1]
            flag1 = 0x80

        cmd_count = 1
        skip = 0
        draw = len(blocks)
        header = struct.pack('<HHHHBB', layer_x1, layer_y1, layer_x2, layer_y2, flag1, 0)
        cmd_header = struct.pack('<H', cmd_count)
        cmd_data = struct.pack('<BB', skip, draw)
        block_data = b''.join(blocks)
        content = header + cmd_header + cmd_data + block_data
        buf += struct.pack('<I', len(content) + 4)
        buf += content
        while len(buf) % 4:
            buf += b'\x00'

        if extra_layers and fi == 0:
            for lt, eblocks in extra_layers:
                if lt == LAYER_SHADOW:
                    header = struct.pack('<HHHHBB', layer_x1, layer_y1, layer_x2, layer_y2, 0, 0)
                else:
                    header = struct.pack('<BB', 0, 0)
                cmd_header = struct.pack('<H', 1)
                cmd_data = struct.pack('<BB', 0, len(eblocks))
                block_data = b''.join(eblocks)
                content = header + cmd_header + cmd_data + block_data
                buf += struct.pack('<I', len(content) + 4)
                buf += content
                while len(buf) % 4:
                    buf += b'\x00'

    return bytes(buf)


class TestShortenPixels:
    def test_single_cut(self):
        """A single cut should reduce height by the cut's height."""
        h, w = 100, 20
        pixels = np.zeros((h, w, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255

        result = _shorten_pixels(pixels, cx=10, cy=80,
                                 margin_u=10, margin_v=10,
                                 cuts=[(4, 20)], total_remove=20)
        assert result.shape == (80, 20, 4)

    def test_multiple_cuts(self):
        """Two cuts should reduce height by the sum of both heights."""
        h, w = 200, 20
        pixels = np.zeros((h, w, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255

        cuts = [(8, 30), (100, 40)]
        result = _shorten_pixels(pixels, cx=10, cy=160,
                                 margin_u=20, margin_v=20,
                                 cuts=cuts, total_remove=70)
        assert result.shape == (130, 20, 4)

    def test_grayscale_collapse(self):
        """BC4 (single-channel) arrays should also work."""
        pixels = np.full((100, 20), 128, dtype=np.uint8)
        result = _shorten_pixels(pixels, 10, 80, 10, 10,
                                 cuts=[(4, 20)], total_remove=20)
        assert result.shape == (80, 20)

    def test_no_cuts(self):
        """Empty cuts list should return the input unchanged."""
        pixels = np.ones((50, 20, 4), dtype=np.uint8)
        # _shorten_pixels with empty cuts shouldn't be called,
        # but if it is, total_remove=0 means new_h=50
        result = _shorten_pixels(pixels, 10, 40, 10, 10,
                                 cuts=[], total_remove=0)
        assert result.shape == (50, 20, 4)

    def test_foundation_preserved(self):
        """Pixels near the foundation (within offset) should survive."""
        h, w = 100, 1
        pixels = np.zeros((h, w, 4), dtype=np.uint8)
        cy, margin = 80, 10
        ftop = 70  # foundation top y for center column
        # Mark 4 pixels just above foundation top
        for y in range(ftop - 4, ftop):
            pixels[y, 0] = [0, 255, 0, 255]

        result = _shorten_pixels(pixels, 0, cy, margin, margin,
                                 cuts=[(4, 20)], total_remove=20)

        green_rows = np.where(np.all(result[:, 0] == [0, 255, 0, 255], axis=-1))[0]
        assert len(green_rows) == 4

    def test_high_offset_cuts_near_top(self):
        """A large offset should cut near the top of the sprite."""
        h, w = 200, 1
        pixels = np.zeros((h, w, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255
        pixels[0:10, 0, 0] = 255  # top 10 rows red

        # foundation_top for center (dx=0) = 160 + (-40) = 120
        # cut_end = 120 - 100 = 20, cut_start = 20 - 30 = -10 -> clamped to 0,30
        result = _shorten_pixels(pixels, 0, 160, 40, 40,
                                 cuts=[(100, 30)], total_remove=30)

        assert result.shape == (170, 1, 4)
        # Top red pixels should be partially removed
        # (cut removes y=0..29, so first 10 red rows are gone)


class TestFindBounds:
    def test_empty_array(self):
        pixels = np.zeros((20, 20, 4), dtype=np.uint8)
        x1, y1, x2, y2 = _find_bounds(pixels, True)
        assert x2 > x1 and y2 > y1

    def test_single_pixel(self):
        pixels = np.zeros((20, 20, 4), dtype=np.uint8)
        pixels[5, 7, :] = [255, 0, 0, 255]
        x1, y1, x2, y2 = _find_bounds(pixels, True)
        assert x1 <= 7 and x2 > 7
        assert y1 <= 5 and y2 > 5
        assert x1 % 4 == 0 and x2 % 4 == 0
        assert y1 % 4 == 0 and y2 % 4 == 0

    def test_bc4_bounds(self):
        pixels = np.zeros((20, 20), dtype=np.uint8)
        pixels[8:12, 4:8] = 200
        x1, y1, x2, y2 = _find_bounds(pixels, False)
        assert x1 == 4 and y1 == 8 and x2 == 8 and y2 == 12


class TestEncodeRegion:
    def test_single_opaque_block(self):
        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[:, :] = [255, 0, 0, 255]
        cmds, blocks = _encode_region(pixels, 0, 0, 4, 4, True)
        assert len(blocks) == 1
        assert sum(d for _, d in cmds) == 1

    def test_empty_region(self):
        pixels = np.zeros((8, 8, 4), dtype=np.uint8)
        cmds, blocks = _encode_region(pixels, 0, 0, 8, 8, True)
        assert len(blocks) == 0

    def test_skip_and_draw(self):
        pixels = np.zeros((4, 8, 4), dtype=np.uint8)
        pixels[:, 4:8] = [0, 255, 0, 255]
        cmds, blocks = _encode_region(pixels, 0, 0, 8, 4, True)
        assert len(blocks) == 1
        assert cmds[0][0] == 1  # skip 1
        assert cmds[0][1] == 1  # draw 1

    def test_bc4_region(self):
        pixels = np.zeros((4, 4), dtype=np.uint8)
        pixels[:, :] = 200
        cmds, blocks = _encode_region(pixels, 0, 0, 4, 4, False)
        assert len(blocks) == 1
        decoded = decode_bc4_block(blocks[0])
        assert np.all(decoded == 200)


class TestShortenSLD:
    def test_single_cut(self):
        """Single cut should reduce canvas height."""
        blocks = [_make_opaque_dxt1_block(200, 100, 50)] * 16
        data = _make_sld_bytes(
            canvas_w=16, canvas_h=20, cx=8, cy=14,
            layer_x1=0, layer_y1=0, layer_x2=16, layer_y2=16,
            main_blocks=blocks,
        )
        sld = parse_sld(data)
        shorten_sld(sld, tile_hh=4, tiles_u=1, tiles_v=1,
                    cuts=[(2, 4)])
        assert sld.frames[0].canvas_height == 16
        assert sld.frames[0].center_y == 14 - 4

    def test_multiple_cuts(self):
        """Multiple cuts should reduce height by the sum."""
        blocks = [_make_opaque_dxt1_block(200, 100, 50)] * 16
        data = _make_sld_bytes(
            canvas_w=16, canvas_h=40, cx=8, cy=30,
            layer_x1=0, layer_y1=0, layer_x2=16, layer_y2=16,
            main_blocks=blocks,
        )
        sld = parse_sld(data)
        shorten_sld(sld, tile_hh=4, tiles_u=1, tiles_v=1,
                    cuts=[(2, 4), (12, 4)])
        assert sld.frames[0].canvas_height == 32
        assert sld.frames[0].center_y == 30 - 8

    def test_empty_cuts_is_noop(self):
        """Empty cuts list should not change the SLD."""
        blocks = [_make_opaque_dxt1_block(100, 100, 100)] * 4
        data = _make_sld_bytes(
            canvas_w=8, canvas_h=8, cx=4, cy=6,
            layer_x1=0, layer_y1=0, layer_x2=8, layer_y2=8,
            main_blocks=blocks,
        )
        sld = parse_sld(data)
        orig_h = sld.frames[0].canvas_height
        shorten_sld(sld, tile_hh=4, tiles_u=1, tiles_v=1, cuts=[])
        assert sld.frames[0].canvas_height == orig_h

    def test_zero_height_cut_is_noop(self):
        """Cuts with height=0 should be ignored."""
        blocks = [_make_opaque_dxt1_block(100, 100, 100)] * 4
        data = _make_sld_bytes(
            canvas_w=8, canvas_h=8, cx=4, cy=6,
            layer_x1=0, layer_y1=0, layer_x2=8, layer_y2=8,
            main_blocks=blocks,
        )
        sld = parse_sld(data)
        shorten_sld(sld, tile_hh=4, tiles_u=1, tiles_v=1,
                    cuts=[(8, 0), (20, 0)])
        assert sld.frames[0].canvas_height == 8

    def test_roundtrip_write(self):
        """Shortened SLD should write/parse without errors."""
        blocks = [_make_opaque_dxt1_block(150, 80, 40)] * 16
        data = _make_sld_bytes(
            canvas_w=16, canvas_h=24, cx=8, cy=18,
            layer_x1=0, layer_y1=0, layer_x2=16, layer_y2=16,
            main_blocks=blocks,
        )
        sld = parse_sld(data)
        shorten_sld(sld, tile_hh=4, tiles_u=1, tiles_v=1,
                    cuts=[(2, 4)])
        output = write_sld(sld)
        reparsed = parse_sld(output)
        assert reparsed.frames[0].canvas_height == 20
        assert len(get_layer(reparsed.frames[0], LAYER_MAIN).blocks) > 0

    def test_with_shadow_layer(self):
        """Shortening should also process shadow layers."""
        main_blocks = [_make_opaque_dxt1_block(200, 200, 200)] * 4
        shadow_blocks = [_make_bc4_block(128)] * 4
        frame_type = LAYER_MAIN | LAYER_SHADOW

        data = _make_sld_bytes(
            canvas_w=8, canvas_h=16, cx=4, cy=12,
            layer_x1=0, layer_y1=0, layer_x2=8, layer_y2=8,
            main_blocks=main_blocks,
            frame_type=frame_type,
            extra_layers=[(LAYER_SHADOW, shadow_blocks)],
        )
        sld = parse_sld(data)
        shorten_sld(sld, tile_hh=4, tiles_u=1, tiles_v=1,
                    cuts=[(2, 4)])
        assert sld.frames[0].canvas_height == 12
        assert get_layer(sld.frames[0], LAYER_SHADOW) is not None

    def test_with_playercolor_layer(self):
        """Shortening should also process playercolor layers."""
        main_blocks = [_make_opaque_dxt1_block(200, 200, 200)] * 4
        pc_blocks = [_make_bc4_block(200)] * 4
        frame_type = LAYER_MAIN | LAYER_PLAYERCOLOR

        data = _make_sld_bytes(
            canvas_w=8, canvas_h=16, cx=4, cy=12,
            layer_x1=0, layer_y1=0, layer_x2=8, layer_y2=8,
            main_blocks=main_blocks,
            frame_type=frame_type,
            extra_layers=[(LAYER_PLAYERCOLOR, pc_blocks)],
        )
        sld = parse_sld(data)
        shorten_sld(sld, tile_hh=4, tiles_u=1, tiles_v=1,
                    cuts=[(2, 4)])
        assert sld.frames[0].canvas_height == 12
        assert get_layer(sld.frames[0], LAYER_PLAYERCOLOR) is not None

    def test_high_offset_cut(self):
        """A large offset should cut near the top of the sprite."""
        blocks = [_make_opaque_dxt1_block(200, 100, 50)] * 16
        data = _make_sld_bytes(
            canvas_w=16, canvas_h=40, cx=8, cy=30,
            layer_x1=0, layer_y1=0, layer_x2=16, layer_y2=16,
            main_blocks=blocks,
        )
        sld = parse_sld(data)
        # offset=20: cut 20px above foundation top -> near sprite top
        shorten_sld(sld, tile_hh=4, tiles_u=1, tiles_v=1,
                    cuts=[(20, 8)])
        assert sld.frames[0].canvas_height == 32
