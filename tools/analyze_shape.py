"""Quick analysis: measure opaque pixel profile of a building sprite."""
import os
import sys
import numpy as np
from sld import parse_sld, get_layer, get_block_positions, LAYER_MAIN
from dxt import decode_dxt1_block
from paths import get_graphics_dir

def analyze_building(filename):
    path = os.path.join(get_graphics_dir(), filename)
    with open(path, 'rb') as f:
        data = f.read()

    sld = parse_sld(data)
    frame = sld.frames[0]  # First frame (idle)
    main = get_layer(frame, LAYER_MAIN)
    if not main:
        print("No main layer!")
        return

    positions = get_block_positions(main, frame)

    # Build full pixel map
    # Find canvas bounds from layer offsets
    min_x = main.offset_x1
    min_y = main.offset_y1
    max_x = main.offset_x2
    max_y = main.offset_y2

    w = max_x - min_x
    h = max_y - min_y

    # Create alpha map
    alpha_map = np.zeros((h, w), dtype=np.uint8)

    for block_idx, canvas_x, canvas_y in positions:
        block_data = main.blocks[block_idx]
        pixels = decode_dxt1_block(block_data)  # 4x4x4 RGBA
        for row in range(4):
            for col in range(4):
                py = canvas_y - min_y + row
                px = canvas_x - min_x + col
                if 0 <= py < h and 0 <= px < w:
                    alpha_map[py, px] = pixels[row, col, 3]

    cx = frame.center_x
    cy = frame.center_y

    print(f"File: {filename}")
    print(f"Canvas: {frame.canvas_width}x{frame.canvas_height}")
    print(f"Hotspot: ({cx}, {cy})")
    print(f"Layer bounds: x=[{min_x},{max_x}] y=[{min_y},{max_y}] size={w}x{h}")
    print()

    # For each row, find leftmost and rightmost opaque pixel
    print("Row profile (relative to hotspot):")
    print(f"{'dy':>6}  {'left_dx':>8}  {'right_dx':>8}  {'width':>6}  {'visual':>40}")

    hotspot_row = cy - min_y

    # Sample every 4th row to keep output manageable
    for y in range(0, h, 4):
        row_alpha = alpha_map[y, :]
        opaque = np.where(row_alpha > 127)[0]
        if len(opaque) == 0:
            continue
        left = opaque[0] + min_x
        right = opaque[-1] + min_x
        width = right - left + 1
        dy = (y + min_y) - cy
        left_dx = left - cx
        right_dx = right - cx

        # Simple ASCII visualization
        bar_left = int((left_dx + 300) * 40 / 600)
        bar_right = int((right_dx + 300) * 40 / 600)
        bar_left = max(0, min(39, bar_left))
        bar_right = max(0, min(39, bar_right))
        visual = '.' * bar_left + '#' * (bar_right - bar_left + 1) + '.' * (39 - bar_right)

        marker = " <-- HOTSPOT" if abs(dy) < 4 else ""
        print(f"{dy:>6}  {left_dx:>8}  {right_dx:>8}  {width:>6}  {visual}{marker}")

    # Measure width at hotspot level
    y_hotspot = cy - min_y
    if 0 <= y_hotspot < h:
        row_alpha = alpha_map[y_hotspot, :]
        opaque = np.where(row_alpha > 127)[0]
        if len(opaque) > 0:
            left = opaque[0] + min_x
            right = opaque[-1] + min_x
            print(f"\nAt hotspot y: opaque from x={left} to x={right}, width={right-left+1}")
            print(f"  left_dx={left-cx}, right_dx={right-cx}")

    # Also measure at a few key levels
    for offset in [-20, -40, -60, -80, -100, -120]:
        y = cy + offset - min_y
        if 0 <= y < h:
            row_alpha = alpha_map[y, :]
            opaque = np.where(row_alpha > 127)[0]
            if len(opaque) > 0:
                left = opaque[0] + min_x
                right = opaque[-1] + min_x
                print(f"  At dy={offset}: opaque width={right-left+1}, dx=[{left-cx},{right-cx}]")

def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "b_west_house_age2_x1.sld", "b_west_house_age2_x2.sld",
        "b_east_house_age2_x1.sld", "b_afri_house_age2_x1.sld",
    ]
    for f in files:
        try:
            analyze_building(f)
            print("\n" + "="*80 + "\n")
        except Exception as e:
            print(f"Error with {f}: {e}\n")


if __name__ == '__main__':
    main()
