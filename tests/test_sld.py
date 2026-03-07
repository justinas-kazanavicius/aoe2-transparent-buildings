"""Tests for SLD parser/writer."""

import struct
import pytest
from sld import (
    SLDFile, SLDFrame, SLDLayer,
    parse_sld, write_sld, get_layer, get_block_positions,
    LAYER_MAIN, LAYER_SHADOW, LAYER_UNKNOWN, LAYER_DAMAGE, LAYER_PLAYERCOLOR,
    LAYER_ORDER,
)


def _make_sld_bytes(frames_data=None):
    """Build minimal valid SLD binary data.

    frames_data: list of (frame_type, layers) where layers is a list of
                 (layer_type, blocks) tuples. blocks is a list of 8-byte values.
    """
    buf = bytearray()
    num_frames = len(frames_data) if frames_data else 0
    # File header
    buf += struct.pack('<4sHHHHI', b'SLDX', 4, num_frames, 0, 0x0010, 0x000000FF)

    if frames_data:
        for fi, (frame_type, layers) in enumerate(frames_data):
            # Frame header: canvas 100x100, center at (50,50)
            buf += struct.pack('<HHhhBBH', 100, 100, 50, 50, frame_type, 0, fi)
            for layer_type, blocks in layers:
                if layer_type == LAYER_UNKNOWN:
                    # Unknown layer: just some raw bytes
                    raw = b'\xAB\xCD\xEF\x01'
                    content_length = len(raw) + 4
                    buf += struct.pack('<I', content_length)
                    buf += raw
                elif layer_type in (LAYER_MAIN, LAYER_SHADOW):
                    # 10-byte header + 2-byte cmd count + commands + blocks
                    cmd_count = 1
                    skip = 0
                    draw = len(blocks)
                    header = struct.pack('<HHHHBB', 0, 0, 100, 100, 0, 0)
                    cmd_header = struct.pack('<H', cmd_count)
                    cmd_data = struct.pack('<BB', skip, draw)
                    block_data = b''.join(blocks)
                    content_data = header + cmd_header + cmd_data + block_data
                    content_length = len(content_data) + 4
                    buf += struct.pack('<I', content_length)
                    buf += content_data
                else:
                    # Damage/playercolor: 2-byte header
                    cmd_count = 1
                    skip = 0
                    draw = len(blocks)
                    header = struct.pack('<BB', 0, 0)
                    cmd_header = struct.pack('<H', cmd_count)
                    cmd_data = struct.pack('<BB', skip, draw)
                    block_data = b''.join(blocks)
                    content_data = header + cmd_header + cmd_data + block_data
                    content_length = len(content_data) + 4
                    buf += struct.pack('<I', content_length)
                    buf += content_data
                # Pad to 4-byte alignment
                remainder = len(buf) % 4
                if remainder != 0:
                    buf += b'\x00' * (4 - remainder)

    return bytes(buf)


class TestParseSLD:
    def test_empty_file(self):
        data = _make_sld_bytes([])
        sld = parse_sld(data)
        assert sld.magic == b'SLDX'
        assert sld.version == 4
        assert sld.num_frames == 0
        assert sld.frames == []

    def test_invalid_magic(self):
        data = struct.pack('<4sHHHHI', b'NOPE', 4, 0, 0, 0, 0)
        with pytest.raises(ValueError, match="Invalid SLD magic"):
            parse_sld(data)

    def test_single_frame_main_layer(self):
        blocks = [b'\xAA' * 8, b'\xBB' * 8]
        data = _make_sld_bytes([(LAYER_MAIN, [(LAYER_MAIN, blocks)])])
        sld = parse_sld(data)
        assert sld.num_frames == 1
        frame = sld.frames[0]
        assert frame.canvas_width == 100
        assert frame.center_x == 50
        assert frame.frame_type == LAYER_MAIN
        assert len(frame.layers) == 1
        layer = frame.layers[0]
        assert layer.layer_type == LAYER_MAIN
        assert len(layer.blocks) == 2
        assert layer.blocks[0] == b'\xAA' * 8
        assert layer.blocks[1] == b'\xBB' * 8

    def test_multiple_layers(self):
        main_blocks = [b'\x11' * 8]
        shadow_blocks = [b'\x22' * 8]
        frame_type = LAYER_MAIN | LAYER_SHADOW
        layers = [(LAYER_MAIN, main_blocks), (LAYER_SHADOW, shadow_blocks)]
        data = _make_sld_bytes([(frame_type, layers)])
        sld = parse_sld(data)
        frame = sld.frames[0]
        assert len(frame.layers) == 2
        assert frame.layers[0].layer_type == LAYER_MAIN
        assert frame.layers[1].layer_type == LAYER_SHADOW

    def test_unknown_layer(self):
        frame_type = LAYER_MAIN | LAYER_UNKNOWN
        layers = [
            (LAYER_MAIN, [b'\x00' * 8]),
            (LAYER_UNKNOWN, []),
        ]
        data = _make_sld_bytes([(frame_type, layers)])
        sld = parse_sld(data)
        frame = sld.frames[0]
        assert len(frame.layers) == 2
        unknown = get_layer(frame, LAYER_UNKNOWN)
        assert unknown is not None
        assert unknown.raw_content == b'\xAB\xCD\xEF\x01'

    def test_multiple_frames(self):
        blocks = [b'\x00' * 8]
        frames = [
            (LAYER_MAIN, [(LAYER_MAIN, blocks)]),
            (LAYER_MAIN, [(LAYER_MAIN, blocks)]),
            (LAYER_MAIN, [(LAYER_MAIN, blocks)]),
        ]
        data = _make_sld_bytes(frames)
        sld = parse_sld(data)
        assert sld.num_frames == 3
        assert len(sld.frames) == 3


class TestWriteSLD:
    def test_roundtrip_empty(self):
        data = _make_sld_bytes([])
        sld = parse_sld(data)
        output = write_sld(sld)
        assert output == data

    def test_roundtrip_single_frame(self):
        blocks = [b'\xAA' * 8, b'\xBB' * 8, b'\xCC' * 8]
        data = _make_sld_bytes([(LAYER_MAIN, [(LAYER_MAIN, blocks)])])
        sld = parse_sld(data)
        output = write_sld(sld)
        reparsed = parse_sld(output)
        assert reparsed.num_frames == 1
        assert len(reparsed.frames[0].layers[0].blocks) == 3
        assert reparsed.frames[0].layers[0].blocks[0] == b'\xAA' * 8

    def test_roundtrip_multiple_layers(self):
        frame_type = LAYER_MAIN | LAYER_SHADOW | LAYER_PLAYERCOLOR
        layers = [
            (LAYER_MAIN, [b'\x11' * 8, b'\x22' * 8]),
            (LAYER_SHADOW, [b'\x33' * 8]),
            (LAYER_PLAYERCOLOR, [b'\x44' * 8, b'\x55' * 8]),
        ]
        data = _make_sld_bytes([(frame_type, layers)])
        sld = parse_sld(data)
        output = write_sld(sld)
        reparsed = parse_sld(output)
        assert len(reparsed.frames[0].layers) == 3
        assert reparsed.frames[0].layers[2].blocks[1] == b'\x55' * 8


class TestGetLayer:
    def test_found(self):
        blocks = [b'\x00' * 8]
        frame_type = LAYER_MAIN | LAYER_SHADOW
        layers = [(LAYER_MAIN, blocks), (LAYER_SHADOW, blocks)]
        data = _make_sld_bytes([(frame_type, layers)])
        frame = parse_sld(data).frames[0]
        assert get_layer(frame, LAYER_MAIN) is not None
        assert get_layer(frame, LAYER_SHADOW) is not None

    def test_not_found(self):
        blocks = [b'\x00' * 8]
        data = _make_sld_bytes([(LAYER_MAIN, [(LAYER_MAIN, blocks)])])
        frame = parse_sld(data).frames[0]
        assert get_layer(frame, LAYER_PLAYERCOLOR) is None


class TestGetBlockPositions:
    def test_single_draw_command(self):
        blocks = [b'\x00' * 8, b'\x00' * 8, b'\x00' * 8]
        data = _make_sld_bytes([(LAYER_MAIN, [(LAYER_MAIN, blocks)])])
        frame = parse_sld(data).frames[0]
        layer = get_layer(frame, LAYER_MAIN)
        positions = get_block_positions(layer, frame)
        assert len(positions) == 3
        # All blocks should be consecutive starting from layer origin
        for i, (idx, x, y) in enumerate(positions):
            assert idx == i

    def test_skip_and_draw(self):
        # Manually build a layer with skip+draw
        data = _make_sld_bytes([(LAYER_MAIN, [(LAYER_MAIN, [b'\x00' * 8])])])
        sld = parse_sld(data)
        layer = sld.frames[0].layers[0]
        # Override commands: skip 2 blocks, then draw 1
        layer.commands = [(2, 1)]
        layer.command_count = 1
        positions = get_block_positions(layer, sld.frames[0])
        assert len(positions) == 1
        idx, x, y = positions[0]
        assert idx == 0
        # Block at grid position 2 (0-indexed), so col=2, row=0
        base_x = layer.offset_x1
        assert x == base_x + 2 * 4
