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
        for entry in sorted(entries):
            if os.path.isdir(entry) and os.path.basename(entry).isdigit():
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
