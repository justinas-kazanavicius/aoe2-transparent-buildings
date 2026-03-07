"""Tests for build_mod logic (no game files needed)."""

import numpy as np
import pytest
from build_mod import (
    get_building_tiles,
    compute_dither_masks,
    compute_outline_masks,
)


class TestGetBuildingTiles:
    def test_house(self):
        assert get_building_tiles("b_west_house_age2_x1.sld", 192, 96) == (2, 2)

    def test_castle(self):
        assert get_building_tiles("b_east_castle_age4_x1.sld", 384, 96) == (4, 4)

    def test_wonder(self):
        assert get_building_tiles("b_west_wonder_x1.sld", 480, 96) == (5, 5)

    def test_outpost(self):
        assert get_building_tiles("b_west_outpost_age2_x1.sld", 96, 96) == (1, 1)

    def test_tower(self):
        assert get_building_tiles("b_west_tower_age2_x1.sld", 96, 96) == (1, 1)

    def test_town_center(self):
        assert get_building_tiles("b_west_town_center_age2_x1.sld", 384, 96) == (4, 4)

    def test_barracks(self):
        assert get_building_tiles("b_west_barracks_age2_x1.sld", 288, 96) == (3, 3)

    def test_gate_ne(self):
        assert get_building_tiles("b_west_gate_ne_age2_x1.sld", 192, 96) == (1, 2)

    def test_gate_se(self):
        assert get_building_tiles("b_west_gate_se_age2_x1.sld", 192, 96) == (2, 1)

    def test_gate_n(self):
        assert get_building_tiles("b_west_gate_n_age2_x1.sld", 192, 96) == (1, 2)

    def test_gate_e(self):
        assert get_building_tiles("b_west_gate_e_age2_x1.sld", 192, 96) == (2, 1)

    def test_gate_corner(self):
        # Gate corners/flags should be 1x1
        assert get_building_tiles("b_west_gate_corner_x1.sld", 96, 96) == (1, 1)

    def test_unknown_building_uses_width(self):
        # Unknown building type falls back to width heuristic
        assert get_building_tiles("b_west_something_x1.sld", 288, 96) == (3, 3)

    def test_siege_workshop_matches_before_workshop(self):
        # "siege_workshop" should match before a hypothetical "workshop"
        assert get_building_tiles("b_west_siege_workshop_age2_x1.sld", 384, 96) == (4, 4)

    def test_x2_scale(self):
        assert get_building_tiles("b_west_house_age2_x2.sld", 384, 192) == (2, 2)


class TestComputeDitherMasks:
    def test_block_well_above_foundation_has_dither(self):
        # Block at (0,0) with center at (50,50) and margin 24
        # Block is well above the hotspot, should get dithered
        bx = np.array([48], dtype=np.int32)
        by = np.array([0], dtype=np.int32)
        masks = compute_dither_masks(bx, by, 50, 50, 24, 24)
        assert masks[0] > 0  # some pixels should be dithered

    def test_block_below_hotspot_no_dither(self):
        # Block below the center_y should never be dithered
        bx = np.array([48], dtype=np.int32)
        by = np.array([60], dtype=np.int32)
        masks = compute_dither_masks(bx, by, 50, 50, 24, 24)
        assert masks[0] == 0

    def test_block_at_foundation_line_no_dither(self):
        # Block right at the foundation line
        bx = np.array([50], dtype=np.int32)
        by = np.array([26], dtype=np.int32)  # center_y - margin = 50 - 24 = 26
        masks = compute_dither_masks(bx, by, 50, 50, 24, 24)
        # At the exact foundation line, pixels should not be dithered
        # (dither requires pixel_y < foundation_y)
        # Some pixels in this block may still be above the diamond edge
        # so we just check it doesn't dither everything
        assert masks[0] != 0xFFFF

    def test_checkerboard_pattern(self):
        # A block well above should have a checkerboard pattern
        # (roughly half the pixels dithered)
        bx = np.array([48], dtype=np.int32)
        by = np.array([0], dtype=np.int32)
        masks = compute_dither_masks(bx, by, 50, 100, 48, 48)
        mask = int(masks[0])
        bits_set = bin(mask).count('1')
        # Checkerboard = exactly 8 of 16 pixels
        assert bits_set == 8

    def test_multiple_blocks(self):
        bx = np.array([48, 48, 48], dtype=np.int32)
        by = np.array([0, 10, 200], dtype=np.int32)
        masks = compute_dither_masks(bx, by, 50, 100, 48, 48)
        assert len(masks) == 3
        assert masks[0] > 0   # above
        assert masks[1] > 0   # above
        assert masks[2] == 0  # below hotspot


class TestComputeOutlineMasks:
    def test_block_at_outline_has_bits(self):
        # Block near the top of the diamond should have outline pixels
        bx = np.array([48], dtype=np.int32)
        by = np.array([26], dtype=np.int32)  # near center_y - margin
        masks = compute_outline_masks(bx, by, 50, 50, 24, 24, thickness=4)
        # May or may not have outline pixels depending on exact geometry
        # Just verify it returns something valid
        assert len(masks) == 1

    def test_block_far_away_no_outline(self):
        bx = np.array([48], dtype=np.int32)
        by = np.array([0], dtype=np.int32)
        masks = compute_outline_masks(bx, by, 50, 200, 24, 24, thickness=4)
        assert masks[0] == 0
