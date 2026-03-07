"""Diagnostic script to inspect raw SLD file structure."""

import struct
import sys
import os

from paths import get_graphics_dir
from sld import LAYER_MAIN, LAYER_SHADOW, LAYER_UNKNOWN, LAYER_DAMAGE, LAYER_PLAYERCOLOR, LAYER_ORDER

LAYER_NAMES = {
    LAYER_MAIN: "Main (DXT1)",
    LAYER_SHADOW: "Shadow (BC4)",
    LAYER_UNKNOWN: "Unknown",
    LAYER_DAMAGE: "Damage (DXT1)",
    LAYER_PLAYERCOLOR: "PlayerColor (BC4)",
}


def inspect(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()

    print(f"File: {filepath}")
    print(f"Total size: {len(data):,} bytes")
    print()

    # File header
    magic, version, num_frames, unk1, unk2, unk3 = struct.unpack_from('<4s4HI', data, 0)
    print(f"Header: magic={magic!r}, version={version}, num_frames={num_frames}")
    print(f"  unknown1={unk1}, unknown2={unk2}, unknown3={unk3}")
    print()

    pos = 16  # After file header

    for fi in range(min(num_frames, 3)):  # Inspect first 3 frames
        print(f"=== Frame {fi} at offset {pos} ===")

        # Frame header: 12 bytes
        if pos + 12 > len(data):
            print("  ERROR: not enough data for frame header")
            break

        (cw, ch, cx, cy, ft, unk, fidx) = struct.unpack_from('<HHhhBBH', data, pos)
        print(f"  Canvas: {cw}x{ch}, Center: ({cx},{cy})")
        print(f"  Frame type: 0x{ft:02X} (binary: {ft:08b})")
        print(f"  Unknown: {unk}, Frame index: {fidx}")
        pos += 12

        # Parse layers
        for layer_bit in LAYER_ORDER:
            if ft & layer_bit:
                name = LAYER_NAMES.get(layer_bit, f"0x{layer_bit:02X}")
                print(f"\n  --- Layer: {name} at offset {pos} ---")

                if pos + 4 > len(data):
                    print("    ERROR: not enough data for content length")
                    break

                content_length = struct.unpack_from('<I', data, pos)[0]
                pos += 4
                print(f"    Content length: {content_length}")

                if layer_bit == LAYER_UNKNOWN:
                    print(f"    [Skipping unknown layer content]")
                    pos += content_length - 4
                else:
                    # Dump raw first 64 bytes of layer content for inspection
                    layer_start = pos
                    preview = data[pos:pos+min(64, content_length)]
                    print(f"    Raw first {len(preview)} bytes: {preview.hex()}")

                    # Try to interpret as layer header
                    # Try different header sizes
                    for hdr_size in [8, 12, 16, 20, 24, 32]:
                        if content_length >= hdr_size:
                            hdr = data[pos:pos+hdr_size]
                            if hdr_size == 8:
                                vals = struct.unpack_from('<4h', hdr)
                                print(f"    As 4xint16 ({hdr_size}B): {vals}")
                            elif hdr_size == 12:
                                vals = struct.unpack_from('<4hI', hdr)
                                print(f"    As 4xint16+uint32 ({hdr_size}B): offsets=({vals[0]},{vals[1]},{vals[2]},{vals[3]}), val5={vals[4]}")
                            elif hdr_size == 16:
                                vals = struct.unpack_from('<4hII', hdr)
                                print(f"    As 4xint16+2xuint32 ({hdr_size}B): offsets=({vals[0]},{vals[1]},{vals[2]},{vals[3]}), flags={vals[4]}, cmd_count={vals[5]}")

                    # Try to figure out valid command_count by working backwards
                    # If header is 16 bytes (4h+II), command array follows
                    if content_length >= 16:
                        vals = struct.unpack_from('<4hII', data, pos)
                        ox1, oy1, ox2, oy2, flags, cmd_count = vals

                        # Sanity check: cmd_count should be reasonable
                        cmd_bytes = cmd_count * 2
                        remaining_after_cmds = content_length - 16 - cmd_bytes
                        if 0 <= remaining_after_cmds and remaining_after_cmds % 8 == 0:
                            total_blocks = remaining_after_cmds // 8
                            print(f"    >> 16B header plausible: {cmd_count} cmds, {cmd_bytes} cmd bytes, {total_blocks} blocks")

                            # Verify by reading commands and summing draw counts
                            if cmd_bytes + 16 <= content_length:
                                cmd_start = pos + 16
                                total_draw = 0
                                for ci in range(cmd_count):
                                    skip, draw = struct.unpack_from('<BB', data, cmd_start + ci * 2)
                                    total_draw += draw
                                    if ci < 5:
                                        print(f"      cmd[{ci}]: skip={skip}, draw={draw}")
                                if cmd_count > 5:
                                    print(f"      ... ({cmd_count - 5} more commands)")
                                print(f"    >> Total draw blocks: {total_draw}, expected blocks from size: {total_blocks}")
                        else:
                            print(f"    >> 16B header: cmd_count={cmd_count}, remaining={content_length - 16 - cmd_bytes} NOT divisible by 8")

                    # Try 12-byte header variant
                    if content_length >= 12:
                        vals = struct.unpack_from('<4hHH', data, pos)
                        ox1, oy1, ox2, oy2, flags_or_cmd, cmd_or_other = vals

                        for hdr_try in [12]:
                            test_cmd_count = struct.unpack_from('<H', data, pos + hdr_try - 2)[0]
                            cmd_bytes_t = test_cmd_count * 2
                            remaining_t = content_length - hdr_try - cmd_bytes_t
                            if 0 <= remaining_t and remaining_t % 8 == 0:
                                total_blocks_t = remaining_t // 8
                                print(f"    >> {hdr_try}B header plausible: cmd_count={test_cmd_count}, blocks={total_blocks_t}")

                    pos = layer_start + content_length - 4

                # Alignment padding
                total_so_far = pos  # position after content
                remainder = pos % 4
                if remainder != 0:
                    pad = 4 - remainder
                    print(f"    Padding: {pad} bytes at offset {pos}")
                    pos += pad

        print()


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "b_west_house_age2_x1.sld"
    filepath = os.path.join(get_graphics_dir(), filename)
    inspect(filepath)


if __name__ == '__main__':
    main()
