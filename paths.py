"""Auto-detect AoE2 DE install and mod directories.

Paths can be overridden with environment variables:
  AOE2_GRAPHICS_DIR  - path to the game's graphics directory
  AOE2_MOD_DIR       - path to the local mod directory (TransparentBuildings root)
"""

import os
import glob


def _find_steam_graphics():
    """Try common Steam install locations for AoE2 DE graphics."""
    candidates = [
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Steam", "steamapps", "common", "AoE2DE"),
        os.path.join(os.environ.get("PROGRAMFILES", ""), "Steam", "steamapps", "common", "AoE2DE"),
        os.path.join("D:", os.sep, "Steam", "steamapps", "common", "AoE2DE"),
        os.path.join("D:", os.sep, "SteamLibrary", "steamapps", "common", "AoE2DE"),
        os.path.join("E:", os.sep, "SteamLibrary", "steamapps", "common", "AoE2DE"),
    ]
    for base in candidates:
        gfx = os.path.join(base, "resources", "_common", "drs", "graphics")
        if os.path.isdir(gfx):
            return gfx
    return None


def _find_mod_dir():
    """Try to find the AoE2 DE user profile and return the mod root."""
    userprofile = os.environ.get("USERPROFILE", "")
    aoe2_base = os.path.join(userprofile, "Games", "Age of Empires 2 DE")
    if os.path.isdir(aoe2_base):
        # Find the first Steam ID subdirectory
        entries = glob.glob(os.path.join(aoe2_base, "*"))
        # Prefer real Steam ID profiles (long numeric) over "0".
        # When multiple profiles exist, pick the one with a savegame dir
        # containing the most files (most likely the active profile).
        numeric_dirs = [e for e in entries
                        if os.path.isdir(e) and os.path.basename(e).isdigit()
                        and os.path.basename(e) != "0"]
        if not numeric_dirs:
            numeric_dirs = [e for e in entries
                            if os.path.isdir(e) and os.path.basename(e) == "0"]
        def _profile_weight(d):
            sg = os.path.join(d, "savegame")
            try:
                return len(os.listdir(sg))
            except OSError:
                return 0
        numeric_dirs.sort(key=_profile_weight, reverse=True)
        for entry in numeric_dirs:
            return os.path.join(entry, "mods", "local", "TransparentBuildings")
    return None


def get_graphics_dir():
    """Return the AoE2 DE graphics directory path."""
    env = os.environ.get("AOE2_GRAPHICS_DIR")
    if env:
        return env
    detected = _find_steam_graphics()
    if detected:
        return detected
    raise FileNotFoundError(
        "Could not find AoE2 DE graphics directory. "
        "Set the AOE2_GRAPHICS_DIR environment variable to the path containing SLD files, e.g.:\n"
        "  set AOE2_GRAPHICS_DIR=C:\\Program Files (x86)\\Steam\\steamapps\\common\\AoE2DE\\resources\\_common\\drs\\graphics"
    )


def get_mod_dir():
    """Return the TransparentBuildings mod root directory path."""
    env = os.environ.get("AOE2_MOD_DIR")
    if env:
        return env
    detected = _find_mod_dir()
    if detected:
        return detected
    raise FileNotFoundError(
        "Could not find AoE2 DE mod directory. "
        "Set the AOE2_MOD_DIR environment variable, e.g.:\n"
        "  set AOE2_MOD_DIR=C:\\Users\\YourName\\Games\\Age of Empires 2 DE\\<steam_id>\\mods\\local\\TransparentBuildings"
    )


def get_mod_graphics_dir():
    """Return the mod's graphics output directory path."""
    return os.path.join(get_mod_dir(), "resources", "_common", "drs", "graphics")


def get_mods_root():
    """Return the mods root directory (parent of local/ and subscribed/)."""
    mod_dir = get_mod_dir()
    # mod_dir is .../mods/local/TransparentBuildings, go up two levels
    return os.path.dirname(os.path.dirname(mod_dir))


def list_available_mods():
    """List all available mods (local and subscribed), excluding TransparentBuildings.

    Returns list of (display_name, mod_root) tuples for mods that have a
    resources/ directory with at least one file. Includes mods with SLD,
    SMX, or any other resource files.
    Display name strips the leading numeric ID prefix from subscribed mods
    (e.g. "469704_Transparent Buildings" -> "Transparent Buildings").
    """
    try:
        mods_root = get_mods_root()
    except FileNotFoundError:
        return []

    mods = []
    for subdir in ('local', 'subscribed'):
        parent = os.path.join(mods_root, subdir)
        if not os.path.isdir(parent):
            continue
        for entry in sorted(os.listdir(parent)):
            if entry == "TransparentBuildings":
                continue
            mod_root = os.path.join(parent, entry)
            res_dir = os.path.join(mod_root, "resources")
            if not os.path.isdir(res_dir):
                continue
            # Strip leading "12345_" ID prefix from subscribed mod names
            display = entry.split('_', 1)[1] if '_' in entry and entry.split('_', 1)[0].isdigit() else entry
            mods.append((display, mod_root))

    # Sort by display name
    mods.sort(key=lambda m: m[0].lower())
    return mods
