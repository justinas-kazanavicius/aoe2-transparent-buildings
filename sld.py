"""
SLD file format parser and writer for AoE2 DE sprites.

SLD Format (since update 66692):
- File header: 16 bytes (magic "SLDX", version, num_frames, unknowns)
- Per frame: 12-byte frame header + N layers
- Layers use DXT1 (main graphics, damage mask) or BC4 (shadow, player color)
- Each layer: 4-byte content_length + layer header + command array + DXT blocks + padding

Reference: https://github.com/SFTtech/openage/blob/master/doc/media/sld-files.md

IMPORTANT: The openage spec labels bits 7-3 for layers, but the actual bit layout
in files uses bits 0-4 (verified from real data + the spec's own 0x17 example).
"""

import struct
import io


# Frame type bit flags (which layers are present)
# Verified mapping from real SLD files:
LAYER_MAIN        = 0x01  # bit 0: Main graphic (DXT1)
LAYER_SHADOW      = 0x02  # bit 1: Shadow (BC4)
LAYER_UNKNOWN     = 0x04  # bit 2: Unknown/outline layer
LAYER_DAMAGE      = 0x08  # bit 3: Damage mask (DXT1)
LAYER_PLAYERCOLOR = 0x10  # bit 4: Player color mask (BC4)

# Layer types for determining codec
DXT1_LAYERS = {LAYER_MAIN, LAYER_DAMAGE}
BC4_LAYERS = {LAYER_SHADOW, LAYER_PLAYERCOLOR}

# Ordered list of layer bits (low to high, matching file order)
LAYER_ORDER = [LAYER_MAIN, LAYER_SHADOW, LAYER_UNKNOWN, LAYER_DAMAGE, LAYER_PLAYERCOLOR]


class SLDLayer:
    """Represents a single layer within an SLD frame."""
    __slots__ = [
        'layer_type', 'content_length',
        'offset_x1', 'offset_y1', 'offset_x2', 'offset_y2',
        'flag1', 'unknown1',
        'command_count', 'commands', 'blocks',
        'raw_content', 'padding',
    ]

    def __init__(self):
        self.layer_type = 0
        self.content_length = 0
        self.offset_x1 = 0
        self.offset_y1 = 0
        self.offset_x2 = 0
        self.offset_y2 = 0
        self.flag1 = 0
        self.unknown1 = 0
        self.command_count = 0
        self.commands = []       # List of (skip_count, draw_count) tuples
        self.blocks = []         # List of 8-byte DXT block data (bytes)
        self.raw_content = b''   # Full raw content for unknown layers
        self.padding = b''       # Alignment padding bytes


class SLDFrame:
    """Represents a single frame in an SLD file."""
    __slots__ = [
        'canvas_width', 'canvas_height',
        'center_x', 'center_y',
        'frame_type', 'unknown', 'frame_index',
        'layers',
    ]

    def __init__(self):
        self.canvas_width = 0
        self.canvas_height = 0
        self.center_x = 0
        self.center_y = 0
        self.frame_type = 0
        self.unknown = 0
        self.frame_index = 0
        self.layers = []


class SLDFile:
    """Represents an entire SLD file."""
    __slots__ = ['magic', 'version', 'num_frames',
                 'unknown1', 'unknown2', 'unknown3', 'frames']

    def __init__(self):
        self.magic = b'SLDX'
        self.version = 4
        self.num_frames = 0
        self.unknown1 = 0
        self.unknown2 = 0x0010
        self.unknown3 = 0x000000FF
        self.frames = []


def parse_sld(data):
    """Parse an SLD file from bytes into an SLDFile object."""
    pos = 0

    sld = SLDFile()

    # File header: 4s + HHHH + I = 16 bytes
    magic, ver, num_frames, unk1, unk2, unk3 = struct.unpack_from('<4sHHHHI', data, pos)
    pos += 16

    if magic != b'SLDX':
        raise ValueError(f"Invalid SLD magic: {magic!r}")

    sld.magic = magic
    sld.version = ver
    sld.num_frames = num_frames
    sld.unknown1 = unk1
    sld.unknown2 = unk2
    sld.unknown3 = unk3
    sld.frames = []

    for _ in range(num_frames):
        frame, pos = _parse_frame(data, pos)
        sld.frames.append(frame)

    return sld


def _parse_frame(data, pos):
    """Parse a single frame starting at pos. Returns (frame, new_pos)."""
    frame = SLDFrame()

    # Frame header: HH hh BB H = 12 bytes
    (frame.canvas_width, frame.canvas_height,
     frame.center_x, frame.center_y,
     frame.frame_type, frame.unknown,
     frame.frame_index) = struct.unpack_from('<HHhhBBH', data, pos)
    pos += 12

    frame.layers = []

    for layer_bit in LAYER_ORDER:
        if frame.frame_type & layer_bit:
            layer, pos = _parse_layer(data, pos, layer_bit)
            frame.layers.append(layer)

    return frame, pos


def _parse_layer(data, pos, layer_type):
    """Parse a single layer starting at pos. Returns (layer, new_pos)."""
    layer = SLDLayer()
    layer.layer_type = layer_type
    layer_start = pos

    # Content length (4 bytes) - includes itself in the count
    layer.content_length = struct.unpack_from('<I', data, pos)[0]
    pos += 4

    if layer_type == LAYER_UNKNOWN:
        # Unknown layer - store raw content (content_length includes the 4-byte field)
        raw_size = layer.content_length - 4
        layer.raw_content = data[pos:pos + raw_size]
        pos += raw_size
    elif layer_type in (LAYER_MAIN, LAYER_SHADOW):
        # 10-byte header: 4 uint16 offsets + 2 uint8 flags
        (layer.offset_x1, layer.offset_y1,
         layer.offset_x2, layer.offset_y2,
         layer.flag1, layer.unknown1) = struct.unpack_from('<HHHHBB', data, pos)
        pos += 10
        pos = _parse_commands_and_blocks(data, pos, layer)
    elif layer_type in (LAYER_DAMAGE, LAYER_PLAYERCOLOR):
        # 2-byte header: flag1 + unknown
        layer.flag1, layer.unknown1 = struct.unpack_from('<BB', data, pos)
        pos += 2
        pos = _parse_commands_and_blocks(data, pos, layer)

    # Pad to 4-byte alignment
    # content_length includes the 4 bytes of itself, so total on-disk = content_length padded up to multiple of 4
    actual_length = layer.content_length + ((4 - layer.content_length) % 4)
    padded_end = layer_start + actual_length
    pad_size = padded_end - pos
    if pad_size > 0:
        layer.padding = data[pos:pos + pad_size]
        pos = padded_end
    else:
        layer.padding = b''

    return layer, pos


def _parse_commands_and_blocks(data, pos, layer):
    """Parse command array + DXT block array. Returns new pos."""
    # Command array length (2 bytes)
    layer.command_count = struct.unpack_from('<H', data, pos)[0]
    pos += 2

    # Commands: 2 bytes each (skip, draw)
    layer.commands = []
    for _ in range(layer.command_count):
        skip, draw = struct.unpack_from('<BB', data, pos)
        pos += 2
        layer.commands.append((skip, draw))

    # DXT blocks: 8 bytes each, total = sum of all draw counts
    total_blocks = sum(draw for _, draw in layer.commands)
    layer.blocks = []
    for _ in range(total_blocks):
        layer.blocks.append(data[pos:pos + 8])
        pos += 8

    return pos


def write_sld(sld):
    """Write an SLDFile object to bytes."""
    f = io.BytesIO()

    # File header: 16 bytes
    f.write(struct.pack('<4sHHHHI',
                        sld.magic, sld.version, sld.num_frames,
                        sld.unknown1, sld.unknown2, sld.unknown3))

    for frame in sld.frames:
        # Frame header: 12 bytes
        f.write(struct.pack('<HHhhBBH',
                            frame.canvas_width, frame.canvas_height,
                            frame.center_x, frame.center_y,
                            frame.frame_type, frame.unknown,
                            frame.frame_index))

        for layer in frame.layers:
            _write_layer(f, layer)

    return f.getvalue()


def _write_layer(f, layer):
    """Write a single layer to the BytesIO stream."""
    if layer.layer_type == LAYER_UNKNOWN:
        f.write(struct.pack('<I', layer.content_length))
        f.write(layer.raw_content)
    else:
        # Build layer content
        if layer.layer_type in (LAYER_MAIN, LAYER_SHADOW):
            header = struct.pack('<HHHHBB',
                                 layer.offset_x1, layer.offset_y1,
                                 layer.offset_x2, layer.offset_y2,
                                 layer.flag1, layer.unknown1)
        else:
            header = struct.pack('<BB', layer.flag1, layer.unknown1)

        cmd_header = struct.pack('<H', layer.command_count)
        cmd_data = b''.join(struct.pack('<BB', s, d) for s, d in layer.commands)
        block_data = b''.join(layer.blocks)

        content_data = header + cmd_header + cmd_data + block_data
        content_length = len(content_data) + 4  # includes the 4-byte length field
        f.write(struct.pack('<I', content_length))
        f.write(content_data)

    # Pad to 4-byte alignment
    pos = f.tell()
    remainder = pos % 4
    if remainder != 0:
        f.write(b'\x00' * (4 - remainder))


def get_layer(frame, layer_type):
    """Get a specific layer from a frame by its type flag, or None."""
    for layer in frame.layers:
        if layer.layer_type == layer_type:
            return layer
    return None


def get_block_positions(layer, frame):
    """
    Calculate the canvas position of each drawn DXT block.

    Blocks are 4x4 pixels. The command array specifies skip+draw counts
    that fill a grid of blocks from left to right, wrapping at the layer width.

    Returns list of (block_index, canvas_x, canvas_y) where canvas_x/y
    are the top-left pixel coords of each 4x4 block on the canvas.
    """
    # For main/shadow layers, use their own offsets
    # For damage/playercolor layers, they share the main layer's dimensions
    if layer.layer_type in (LAYER_MAIN, LAYER_SHADOW):
        layer_w = layer.offset_x2 - layer.offset_x1
        layer_h = layer.offset_y2 - layer.offset_y1
        base_x = layer.offset_x1
        base_y = layer.offset_y1
    else:
        # Damage and playercolor use the main layer's bounds
        main = get_layer(frame, LAYER_MAIN)
        if main:
            layer_w = main.offset_x2 - main.offset_x1
            layer_h = main.offset_y2 - main.offset_y1
            base_x = main.offset_x1
            base_y = main.offset_y1
        else:
            return []

    blocks_per_row = (layer_w + 3) // 4  # Ceiling division

    positions = []
    block_idx = 0
    cursor = 0  # Linear block position in the grid

    for skip, draw in layer.commands:
        cursor += skip
        for _ in range(draw):
            row = cursor // blocks_per_row
            col = cursor % blocks_per_row
            canvas_x = base_x + col * 4
            canvas_y = base_y + row * 4
            positions.append((block_idx, canvas_x, canvas_y))
            block_idx += 1
            cursor += 1

    return positions
