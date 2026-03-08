# SLD Delta Encoding (Animated Sprites)

SLD files for animated buildings (mills, folwarks, etc.) use **delta encoding**
to avoid storing redundant data across frames.

## How it works

- **Frame 0** is always a full frame (`flag1 = 0x00`). All blocks are stored
  and draw commands render the complete sprite.
- **Frames 1+** are delta frames (`flag1 = 0x80`, bit 7 set). Only the blocks
  that **changed** from the previous frame are stored as draw commands. Skip
  commands mean "copy the block from the previous frame's output" instead of
  the usual "transparent."

This applies per-layer: main, shadow, damage, and playercolor layers each
independently set `flag1` bit 7.

### Example: Western Mill (age3, x1)

| Frame | flag1  | Main blocks | Notes                                |
|-------|--------|-------------|--------------------------------------|
| 0     | `0x00` | 1848        | Full frame, all blocks stored        |
| 1     | `0x80` | 1374        | Delta: only sail movement blocks     |
| 2     | `0x80` | 1533        | Delta: only sail movement blocks     |

The ~474 missing blocks in frame 1 are the static building body, carried
forward from frame 0 by the game engine.

## flag1 bit layout

From the [openage SLD spec](https://github.com/SFTtech/openage/blob/master/doc/media/sld-files.md):

| Bit | Mask   | Meaning                                       |
|-----|--------|-----------------------------------------------|
| 7   | `0x80` | Delta mode: skip = copy from previous frame   |
| 0   | `0x01` | Set in playercolor/shadow layers              |
| 1-6 |        | Unknown                                       |

## Implications for modding

### Rendering

When decoding animated sprites for visualization (GIF export, etc.), each
frame must be rendered by **accumulating** onto the previous frame's canvas:

```python
prev = None
for frame in sld.frames:
    layer = get_layer(frame, LAYER_MAIN)
    is_delta = prev is not None and (layer.flag1 & 0x80)
    canvas = prev.copy() if is_delta else blank_canvas()
    # draw only the blocks present in this frame onto canvas
    for block in draw_blocks:
        canvas[block.pos] = decode(block)
    prev = canvas
```

Without this, delta frames appear to have missing/transparent blocks where
the static building should be.

### Dithering animated buildings

Delta encoding gives us animation detection for free. In delta frames, only
the **changed** blocks have draw commands. Blocks drawn frequently across
delta frames are the continuously animated parts (e.g. mill sails rotating).

To keep animated parts opaque while dithering the static building body:

1. Count how often each `(bx, by)` position appears as a draw block across
   all delta frames.
2. Positions drawn in more than 50% of delta frames are animated — protect
   them from dithering with a full `0xFFFF` mask.
3. Positions rarely or never drawn in delta frames are static body — dither
   them normally.

This is fast (no pixel decoding needed) and directly leverages the SLD
structure.

### Modifying delta frames

When applying dithering to an animated SLD:

- **Frame 0**: dither normally. The dithered static blocks become the
  baseline that the game carries forward through delta frames.
- **Delta frames**: only the draw blocks (changed/animated blocks) are
  present. Apply dithering to these too, but animation protection masks
  keep the animated pixels opaque. Static body blocks are not present
  in the frame data — they're carried forward from frame 0 by the engine.

The game engine handles the composition: dithered body from frame 0 persists
via skip commands, while animated blocks (sails, etc.) are drawn fresh each
frame and kept opaque by the protection mask.
