"""Test roundtrip: parse -> write -> compare to verify SLD parser/writer correctness."""

import os
import sys

from sld import parse_sld, write_sld, get_layer, get_block_positions, LAYER_MAIN, LAYER_SHADOW, LAYER_DAMAGE, LAYER_PLAYERCOLOR, LAYER_ORDER, LAYER_UNKNOWN
from paths import get_graphics_dir


def test_roundtrip(filename):
    filepath = os.path.join(get_graphics_dir(), filename)
    with open(filepath, 'rb') as f:
        original = f.read()

    sld = parse_sld(original)
    output = write_sld(sld)

    print(f"File: {filename}")
    print(f"  Original: {len(original):,} bytes")
    print(f"  Output:   {len(output):,} bytes")

    if original == output:
        print("  MATCH: Perfect roundtrip!")
        return True
    else:
        # Find first difference
        min_len = min(len(original), len(output))
        for i in range(min_len):
            if original[i] != output[i]:
                print(f"  MISMATCH at byte {i} (0x{i:X}): original=0x{original[i]:02X}, output=0x{output[i]:02X}")
                # Show context
                start = max(0, i - 8)
                end = min(min_len, i + 8)
                print(f"    Original [{start}:{end}]: {original[start:end].hex()}")
                print(f"    Output   [{start}:{end}]: {output[start:end].hex()}")
                break
        if len(original) != len(output):
            print(f"  Size difference: {len(output) - len(original):+d} bytes")
        return False


def inspect_frames(filename):
    filepath = os.path.join(get_graphics_dir(), filename)
    with open(filepath, 'rb') as f:
        data = f.read()

    sld = parse_sld(data)
    print(f"\nFrame details for {filename}:")
    print(f"  Version: {sld.version}, Frames: {sld.num_frames}")
    print(f"  Unknown1: {sld.unknown1}, Unknown2: 0x{sld.unknown2:04X}, Unknown3: 0x{sld.unknown3:08X}")

    for fi, frame in enumerate(sld.frames):
        print(f"\n  Frame {fi}:")
        print(f"    Canvas: {frame.canvas_width}x{frame.canvas_height}")
        print(f"    Center: ({frame.center_x}, {frame.center_y})")
        print(f"    Type: 0x{frame.frame_type:02X} ({frame.frame_type:08b})")
        print(f"    Unknown: {frame.unknown}, Index: {frame.frame_index}")

        for layer in frame.layers:
            layer_names = {0x01: "Main", 0x02: "Shadow", 0x04: "Unknown",
                          0x08: "Damage", 0x10: "PlayerColor"}
            name = layer_names.get(layer.layer_type, f"0x{layer.layer_type:02X}")
            print(f"    Layer {name}:")
            print(f"      Content length: {layer.content_length}")

            if layer.layer_type != LAYER_UNKNOWN:
                if layer.layer_type in (LAYER_MAIN, LAYER_SHADOW):
                    print(f"      Offsets: ({layer.offset_x1},{layer.offset_y1}) to ({layer.offset_x2},{layer.offset_y2})")
                    w = layer.offset_x2 - layer.offset_x1
                    h = layer.offset_y2 - layer.offset_y1
                    print(f"      Layer size: {w}x{h} pixels ({(w+3)//4}x{(h+3)//4} blocks)")
                print(f"      Flag1: {layer.flag1}, Unknown1: {layer.unknown1}")
                print(f"      Commands: {layer.command_count}")
                print(f"      Blocks: {len(layer.blocks)}")
                if layer.commands:
                    total_draw = sum(d for _, d in layer.commands)
                    total_skip = sum(s for s, _ in layer.commands)
                    print(f"      Total skip: {total_skip}, Total draw: {total_draw}")
            if layer.padding:
                print(f"      Padding: {len(layer.padding)} bytes")


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "b_west_house_age2_x1.sld"
    inspect_frames(filename)
    print()
    test_roundtrip(filename)


if __name__ == '__main__':
    main()
