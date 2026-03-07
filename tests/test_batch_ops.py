"""Tests for batch DXT1 operations and layer block manipulation."""

import struct
import numpy as np
import pytest
from dxt import rgb_to_rgb565, rgb565_to_rgb, decode_dxt1_block, decode_bc4_block
from sld import SLDFile, SLDFrame, SLDLayer, LAYER_MAIN, LAYER_SHADOW, LAYER_PLAYERCOLOR, get_layer, get_block_positions
from build_mod import (
    _apply_dxt1_masks_batch,
    _force_opaque_dxt1_batch,
    ensure_layer_blocks,
    inject_bc4_outline,
    compute_edge_protection,
    process_frame,
    find_building_files,
)


def _make_dxt1_block(c0, c1, indices=0):
    """Helper: build an 8-byte DXT1 block."""
    return struct.pack('<HHI', c0, c1, indices)


def _make_frame_with_main(blocks, width=100, height=100, cx=50, cy=50,
                          skip=0, extra_layers=None):
    """Helper: build a frame with a main layer containing the given blocks.

    Returns (frame, main_layer).
    """
    frame = SLDFrame()
    frame.canvas_width = width
    frame.canvas_height = height
    frame.center_x = cx
    frame.center_y = cy
    frame.frame_type = LAYER_MAIN
    frame.unknown = 0
    frame.frame_index = 0
    frame.layers = []

    main = SLDLayer()
    main.layer_type = LAYER_MAIN
    main.offset_x1 = 0
    main.offset_y1 = 0
    main.offset_x2 = width
    main.offset_y2 = height
    main.flag1 = 0
    main.unknown1 = 0
    main.commands = [(skip, len(blocks))]
    main.command_count = 1
    main.blocks = list(blocks)
    main.raw_content = b''
    main.padding = b''
    frame.layers.append(main)

    if extra_layers:
        for layer in extra_layers:
            frame.frame_type |= layer.layer_type
            frame.layers.append(layer)

    return frame, main


class TestApplyDXT1MasksBatch:
    def test_empty_dict(self):
        _, layer = _make_frame_with_main([b'\xFF' * 8])
        _apply_dxt1_masks_batch(layer, {})
        assert layer.blocks[0] == b'\xFF' * 8  # unchanged

    def test_full_transparency_mask(self):
        block = _make_dxt1_block(0xFFFF, 0x0000, 0x00000000)  # opaque mode
        _, layer = _make_frame_with_main([block])
        _apply_dxt1_masks_batch(layer, {0: 0xFFFF})
        assert layer.blocks[0] == b'\x00' * 8

    def test_partial_mask_transparent_mode(self):
        # Already in transparent mode (c0 <= c1)
        c0 = rgb_to_rgb565(100, 100, 100)
        c1 = rgb_to_rgb565(200, 200, 200)
        if c0 > c1:
            c0, c1 = c1, c0
        block = _make_dxt1_block(c0, c1, 0x00000000)  # all index 0
        _, layer = _make_frame_with_main([block])
        # Make pixel 0 transparent
        _apply_dxt1_masks_batch(layer, {0: 0x0001})
        decoded = decode_dxt1_block(layer.blocks[0])
        assert decoded[0, 0, 3] == 0    # pixel 0 transparent
        assert decoded[0, 1, 3] == 255  # pixel 1 opaque

    def test_opaque_mode_gets_swapped(self):
        # Opaque mode (c0 > c1), should be swapped to transparent mode
        c0 = rgb_to_rgb565(255, 0, 0)
        c1 = rgb_to_rgb565(0, 0, 255)
        if c0 < c1:
            c0, c1 = c1, c0
        block = _make_dxt1_block(c0, c1, 0x00000000)
        _, layer = _make_frame_with_main([block])
        _apply_dxt1_masks_batch(layer, {0: 0x0001})
        decoded = decode_dxt1_block(layer.blocks[0])
        assert decoded[0, 0, 3] == 0  # pixel 0 transparent
        # Remaining pixels should still be opaque
        assert decoded[0, 1, 3] == 255

    def test_multiple_blocks(self):
        blocks = [
            _make_dxt1_block(0x0000, 0x0000, 0x00000000),
            _make_dxt1_block(0x0000, 0x0000, 0x00000000),
            _make_dxt1_block(0x0000, 0x0000, 0x00000000),
        ]
        _, layer = _make_frame_with_main(blocks)
        # Dither blocks 0 and 2, leave block 1 untouched
        _apply_dxt1_masks_batch(layer, {0: 0x5555, 2: 0xAAAA})
        decoded0 = decode_dxt1_block(layer.blocks[0])
        decoded1 = decode_dxt1_block(layer.blocks[1])
        decoded2 = decode_dxt1_block(layer.blocks[2])
        # Block 0: every other pixel transparent
        assert decoded0[0, 0, 3] == 0  # bit 0 set
        assert decoded0[0, 1, 3] == 255
        # Block 1: untouched (all opaque, c0==c1==0 means color is black)
        assert np.all(decoded1[:, :, 3] == 255)
        # Block 2: opposite checkerboard
        assert decoded2[0, 0, 3] == 255
        assert decoded2[0, 1, 3] == 0  # bit 1 set


class TestForceOpaqueDXT1Batch:
    def test_empty_dict(self):
        block = _make_dxt1_block(0, 0, 0xFFFFFFFF)  # all transparent
        _, layer = _make_frame_with_main([block])
        _force_opaque_dxt1_batch(layer, {})
        assert layer.blocks[0] == block  # unchanged

    def test_forces_transparent_pixels_opaque(self):
        # Transparent mode with all pixels transparent (index 3)
        c0 = rgb_to_rgb565(128, 128, 128)
        block = _make_dxt1_block(c0, c0, 0xFFFFFFFF)
        _, layer = _make_frame_with_main([block])
        # Force pixel 0 opaque
        _force_opaque_dxt1_batch(layer, {0: 0x0001})
        decoded = decode_dxt1_block(layer.blocks[0])
        assert decoded[0, 0, 3] == 255  # forced opaque
        assert decoded[0, 1, 3] == 0    # still transparent

    def test_skips_opaque_mode_blocks(self):
        # Opaque mode (c0 > c1) — no index 3 = transparent, so nothing to do
        c0 = rgb_to_rgb565(255, 0, 0)
        c1 = rgb_to_rgb565(0, 0, 255)
        if c0 < c1:
            c0, c1 = c1, c0
        block = _make_dxt1_block(c0, c1, 0xFFFFFFFF)
        _, layer = _make_frame_with_main([block])
        original = layer.blocks[0]
        _force_opaque_dxt1_batch(layer, {0: 0x0001})
        # Block should be unchanged (opaque mode skipped)
        assert layer.blocks[0] == original


class TestEnsureLayerBlocks:
    def test_add_blocks_at_new_positions(self):
        # Layer with 1 block at grid position 0
        block = b'\xAA' * 8
        frame, main = _make_frame_with_main([block], width=20, height=4)
        # blocks_per_row = 20/4 = 5, so grid positions 0-4 for row 0
        # Add block at grid position 2
        ensure_layer_blocks(main, frame, {2})
        assert len(main.blocks) == 2
        assert main.blocks[0] == block  # original at pos 0
        assert main.blocks[1] == b'\x00' * 8  # new default at pos 2

    def test_existing_blocks_preserved(self):
        blocks = [b'\x11' * 8, b'\x22' * 8]
        frame, main = _make_frame_with_main(blocks, width=20, height=4)
        # Original: skip=0, draw=2 → positions 0,1
        # Request position 1 (already exists) and position 3 (new)
        ensure_layer_blocks(main, frame, {1, 3})
        assert len(main.blocks) == 3  # pos 0, 1, 3
        assert main.blocks[0] == b'\x11' * 8  # pos 0
        assert main.blocks[1] == b'\x22' * 8  # pos 1 (existing)
        assert main.blocks[2] == b'\x00' * 8  # pos 3 (new default)

    def test_custom_default_block(self):
        frame, main = _make_frame_with_main([b'\xAA' * 8], width=20, height=4)
        custom = b'\xFF' * 8
        ensure_layer_blocks(main, frame, {3}, default_block=custom)
        assert main.blocks[1] == custom

    def test_commands_updated_correctly(self):
        frame, main = _make_frame_with_main([b'\x00' * 8], width=20, height=4)
        # Original: skip=0, draw=1 → pos 0
        # Add at pos 4 (with gap)
        ensure_layer_blocks(main, frame, {4})
        # Should have skip=0,draw=1 then skip=3,draw=1
        total_draw = sum(d for _, d in main.commands)
        assert total_draw == 2
        assert len(main.blocks) == 2


class TestInjectBC4Outline:
    def test_zero_mask_returns_same(self):
        block = b'\xC8\xC8\x00\x00\x00\x00\x00\x00'  # constant 200
        assert inject_bc4_outline(block, 0) == block

    def test_sets_pixels_to_value(self):
        block = b'\x00\x00\x00\x00\x00\x00\x00\x00'  # all zeros
        result = inject_bc4_outline(block, 0x0001, value=255)
        decoded = decode_bc4_block(result).flatten()
        assert decoded[0] == 255
        # Other pixels should still be 0 or very close
        assert decoded[1] == 0

    def test_custom_value(self):
        block = b'\x00\x00\x00\x00\x00\x00\x00\x00'
        result = inject_bc4_outline(block, 0x0003, value=180)
        decoded = decode_bc4_block(result).flatten()
        assert decoded[0] == 180
        assert decoded[1] == 180

    def test_multiple_pixels(self):
        block = b'\x00\x00\x00\x00\x00\x00\x00\x00'
        # Set pixels 0, 4, 8, 12 (one per row)
        mask = (1 << 0) | (1 << 4) | (1 << 8) | (1 << 12)
        result = inject_bc4_outline(block, mask, value=200)
        decoded = decode_bc4_block(result)
        assert decoded[0, 0] == 200
        assert decoded[1, 0] == 200
        assert decoded[2, 0] == 200
        assert decoded[3, 0] == 200
        assert decoded[0, 1] == 0


class TestComputeEdgeProtection:
    def test_empty_positions(self):
        assert compute_edge_protection([], 2) == {}

    def test_zero_inset(self):
        positions = [(0, 0, 0)]
        assert compute_edge_protection(positions, 0) == {}

    def test_single_block_all_edge(self):
        # A single 4x4 block — all pixels are on the edge
        positions = [(0, 0, 0)]
        prot = compute_edge_protection(positions, 2)
        assert (0, 0) in prot
        # With inset=2, all 16 pixels of a lone block should be protected
        assert prot[(0, 0)] == 0xFFFF

    def test_interior_block_not_protected(self):
        # 3x3 grid of blocks — the center block should have no edge pixels
        # (with small inset)
        positions = []
        idx = 0
        for row in range(3):
            for col in range(3):
                positions.append((idx, col * 4, row * 4))
                idx += 1
        prot = compute_edge_protection(positions, 1)
        # Center block (4,4) should have no protection (all pixels interior)
        center_mask = prot.get((4, 4), 0)
        assert center_mask == 0

    def test_edge_block_has_protection(self):
        # 3x3 grid, corner block should have edge protection
        positions = []
        idx = 0
        for row in range(3):
            for col in range(3):
                positions.append((idx, col * 4, row * 4))
                idx += 1
        prot = compute_edge_protection(positions, 1)
        # Corner block (0,0) should have some protected pixels
        assert prot.get((0, 0), 0) > 0


class TestProcessFrame:
    def _make_test_frame(self, num_blocks_w=10, num_blocks_h=10):
        """Build a frame with a grid of opaque DXT1 blocks."""
        w = num_blocks_w * 4
        h = num_blocks_h * 4
        cx = w // 2
        cy = h // 2

        # Create uniform gray opaque blocks
        c = rgb_to_rgb565(128, 128, 128)
        # Transparent mode (c0 <= c1) with all index 0 (opaque)
        block = _make_dxt1_block(c, c, 0x00000000)

        total = num_blocks_w * num_blocks_h
        blocks = [block] * total
        frame, main = _make_frame_with_main(blocks, w, h, cx, cy)
        # Set commands to draw all blocks
        main.commands = [(0, min(total, 255))]
        if total > 255:
            main.commands = [(0, 255), (0, total - 255)]
        main.command_count = len(main.commands)
        return frame

    def test_blocks_above_hotspot_get_dithered(self):
        frame = self._make_test_frame(10, 20)
        original_blocks = [b[:] for b in frame.layers[0].blocks]

        process_frame(frame, tile_hh=24, tiles=(2, 2), outline_enabled=False)

        main = get_layer(frame, LAYER_MAIN)
        # At least some blocks above the hotspot should be modified
        changed = 0
        for i, (orig, new) in enumerate(zip(original_blocks, main.blocks)):
            if orig != new:
                changed += 1
        assert changed > 0

    def test_blocks_below_hotspot_unchanged(self):
        frame = self._make_test_frame(10, 20)
        cx = frame.center_x
        cy = frame.center_y
        main = get_layer(frame, LAYER_MAIN)
        original_blocks = [b[:] for b in main.blocks]

        process_frame(frame, tile_hh=24, tiles=(2, 2), outline_enabled=False)

        # Check that blocks well below hotspot + margin are unchanged
        positions = get_block_positions(main, frame)
        for idx, bx, by in positions:
            if by > cy + 30:  # well below foundation
                assert main.blocks[idx] == original_blocks[idx], \
                    f"Block at ({bx},{by}) below hotspot was modified"

    def test_no_main_layer_is_noop(self):
        frame = SLDFrame()
        frame.canvas_width = 100
        frame.canvas_height = 100
        frame.center_x = 50
        frame.center_y = 50
        frame.frame_type = 0
        frame.layers = []
        # Should not crash
        process_frame(frame, tile_hh=24, tiles=(2, 2))

    def test_dithered_pixels_are_transparent(self):
        frame = self._make_test_frame(10, 20)
        process_frame(frame, tile_hh=24, tiles=(2, 2), outline_enabled=False)

        main = get_layer(frame, LAYER_MAIN)
        positions = get_block_positions(main, frame)
        cy = frame.center_y

        found_transparent = False
        for idx, bx, by in positions:
            if by < cy - 30:  # well above hotspot
                decoded = decode_dxt1_block(main.blocks[idx])
                if np.any(decoded[:, :, 3] == 0):
                    found_transparent = True
                    break
        assert found_transparent, "No transparent pixels found above hotspot"

    def test_with_outline_enabled(self):
        # Use a larger frame so the foundation diamond (at hotspot +/- margin)
        # falls within the drawn block grid. With tiles=(2,2) and tile_hh=24,
        # margin=48. Hotspot at center means outline at cy-48 to cy+48.
        # Need enough height so blocks exist around the foundation line.
        frame = self._make_test_frame(num_blocks_w=30, num_blocks_h=30)

        # Add a playercolor layer
        main = get_layer(frame, LAYER_MAIN)
        pc = SLDLayer()
        pc.layer_type = LAYER_PLAYERCOLOR
        pc.flag1 = 0
        pc.unknown1 = 0
        pc.commands = list(main.commands)
        pc.command_count = main.command_count
        pc.blocks = [b'\x00' * 8] * len(main.blocks)
        pc.raw_content = b''
        pc.padding = b''
        frame.frame_type |= LAYER_PLAYERCOLOR
        frame.layers.append(pc)

        process_frame(frame, tile_hh=24, tiles=(2, 2),
                      outline_enabled=True, outline_thickness=4, outline_value=200)

        # Some playercolor blocks near the foundation should have non-zero values
        pc_layer = get_layer(frame, LAYER_PLAYERCOLOR)
        has_outline = False
        for block in pc_layer.blocks:
            vals = decode_bc4_block(block).flatten()
            if np.any(vals > 100):
                has_outline = True
                break
        assert has_outline, "No outline pixels found in playercolor layer"

    def test_with_shadow_layer(self):
        frame = self._make_test_frame(10, 20)

        # Add shadow layer
        main = get_layer(frame, LAYER_MAIN)
        shadow = SLDLayer()
        shadow.layer_type = LAYER_SHADOW
        shadow.offset_x1 = main.offset_x1
        shadow.offset_y1 = main.offset_y1
        shadow.offset_x2 = main.offset_x2
        shadow.offset_y2 = main.offset_y2
        shadow.flag1 = 0
        shadow.unknown1 = 0
        shadow.commands = list(main.commands)
        shadow.command_count = main.command_count
        # Shadow blocks are BC4 with some nonzero value
        from dxt import encode_bc4_block_from_flat
        shadow_block = encode_bc4_block_from_flat(np.full(16, 128, dtype=np.uint8))
        shadow.blocks = [shadow_block] * len(main.blocks)
        shadow.raw_content = b''
        shadow.padding = b''
        frame.frame_type |= LAYER_SHADOW
        frame.layers.append(shadow)

        original_shadow = [b[:] for b in shadow.blocks]
        process_frame(frame, tile_hh=24, tiles=(2, 2), outline_enabled=False)

        # Shadow blocks above hotspot should be modified (zeroed where dithered)
        shadow_layer = get_layer(frame, LAYER_SHADOW)
        changed = sum(1 for a, b in zip(original_shadow, shadow_layer.blocks) if a != b)
        assert changed > 0, "Shadow layer blocks were not modified"


class TestFindBuildingFiles:
    def test_filters_destruction(self, tmp_path, monkeypatch):
        # Create fake SLD files
        (tmp_path / "b_west_house_age2_x1.sld").touch()
        (tmp_path / "b_west_house_destruction_age2_x1.sld").touch()
        (tmp_path / "b_west_house_rubble_age2_x1.sld").touch()
        monkeypatch.setattr("build_mod.get_graphics_dir", lambda: str(tmp_path))
        files = find_building_files()
        assert "b_west_house_age2_x1.sld" in files
        assert "b_west_house_destruction_age2_x1.sld" not in files
        assert "b_west_house_rubble_age2_x1.sld" not in files

    def test_filters_flags(self, tmp_path, monkeypatch):
        (tmp_path / "b_west_house_age2_x1.sld").touch()
        (tmp_path / "b_west_gate_flag_x1.sld").touch()
        monkeypatch.setattr("build_mod.get_graphics_dir", lambda: str(tmp_path))
        files = find_building_files()
        assert "b_west_house_age2_x1.sld" in files
        assert "b_west_gate_flag_x1.sld" not in files

    def test_filters_mills_by_default(self, tmp_path, monkeypatch):
        (tmp_path / "b_west_house_age2_x1.sld").touch()
        (tmp_path / "b_west_mill_age2_x1.sld").touch()
        monkeypatch.setattr("build_mod.get_graphics_dir", lambda: str(tmp_path))
        files = find_building_files()
        assert "b_west_house_age2_x1.sld" in files
        assert "b_west_mill_age2_x1.sld" not in files

    def test_exclude_custom_building(self, tmp_path, monkeypatch):
        (tmp_path / "b_west_house_age2_x1.sld").touch()
        (tmp_path / "b_west_monastery_age3_x1.sld").touch()
        (tmp_path / "b_west_mill_age2_x1.sld").touch()
        monkeypatch.setattr("build_mod.get_graphics_dir", lambda: str(tmp_path))
        files = find_building_files(exclude=['monastery'])
        assert "b_west_house_age2_x1.sld" in files
        assert "b_west_monastery_age3_x1.sld" not in files
        # mill is NOT excluded when custom list is provided
        assert "b_west_mill_age2_x1.sld" in files

    def test_exclude_empty_includes_all(self, tmp_path, monkeypatch):
        (tmp_path / "b_west_house_age2_x1.sld").touch()
        (tmp_path / "b_west_mill_age2_x1.sld").touch()
        monkeypatch.setattr("build_mod.get_graphics_dir", lambda: str(tmp_path))
        files = find_building_files(exclude=[])
        assert "b_west_mill_age2_x1.sld" in files

    def test_filters_sides(self, tmp_path, monkeypatch):
        (tmp_path / "b_west_house_age2_x1.sld").touch()
        (tmp_path / "b_west_castle_sides_x1.sld").touch()
        monkeypatch.setattr("build_mod.get_graphics_dir", lambda: str(tmp_path))
        files = find_building_files()
        assert "b_west_house_age2_x1.sld" in files
        assert "b_west_castle_sides_x1.sld" not in files

    def test_returns_sorted(self, tmp_path, monkeypatch):
        (tmp_path / "b_west_house_age2_x1.sld").touch()
        (tmp_path / "b_east_barracks_age2_x1.sld").touch()
        (tmp_path / "b_afri_castle_age4_x1.sld").touch()
        monkeypatch.setattr("build_mod.get_graphics_dir", lambda: str(tmp_path))
        files = find_building_files()
        assert files == sorted(files)
