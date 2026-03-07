"""Check which buildings get wrong tile count vs known footprint sizes."""
import sys, os, glob, re
from sld import parse_sld, get_layer, LAYER_MAIN
from paths import get_graphics_dir

# Known building footprint sizes (NxN tiles)
KNOWN_SIZES = {
    'house': 2,
    'barracks': 3,
    'archery_range': 3,
    'stable': 3,
    'siege_workshop': 4,
    'blacksmith': 3,
    'market': 4,
    'monastery': 3,
    'university': 4,
    'town_center': 4,
    'castle': 4,
    'wonder': 5,
    'lumber_camp': 2,
    'mining_camp': 2,
    'mill': 2,
    'dock': 3,
    'tower': 1,
    'outpost': 1,
    'krepost': 3,
    'donjon': 2,
    'mule_cart': 1,
}

def main():
    gfx = get_graphics_dir()
    files = sorted(glob.glob(os.path.join(gfx, 'b_*_x1.sld')))
    files = [f for f in files if '_destruction_' not in f and '_rubble_' not in f
             and '_foundation_' not in f and '_shadow' not in f
             and '_front' not in f and '_back' not in f and '_center' not in f and '_main' not in f]

    wrong = []
    matched = 0
    for path in files:
        name = os.path.basename(path)
        m = re.match(r'b_\w+?_(town_center|archery_range|siege_workshop|lumber_camp|mining_camp|mule_cart|[a-z]+)(?:_\d+)?_age\d+_x1\.sld', name)
        if not m:
            continue
        btype = m.group(1)
        if btype not in KNOWN_SIZES:
            continue

        expected = KNOWN_SIZES[btype]
        with open(path, 'rb') as f:
            data = f.read()
        sld = parse_sld(data)
        frame = sld.frames[0]
        main_layer = get_layer(frame, LAYER_MAIN)
        if not main_layer:
            continue
        w = main_layer.offset_x2 - main_layer.offset_x1
        computed = round(w / 96)
        matched += 1
        if computed != expected:
            wrong.append((name, btype, expected, computed, w))

    print(f"Checked {matched} buildings, {len(wrong)} have wrong tile count:")
    for name, btype, expected, computed, w in wrong:
        print(f"  {name}: type={btype} expected={expected} got={computed} (w={w}, w/96={w/96:.2f})")


if __name__ == '__main__':
    main()
