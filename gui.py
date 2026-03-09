"""
Transparent Buildings Mod - GUI

A tkinter GUI for configuring and building the mod without touching the command line.
Includes a live preview panel showing before/after rendering of a selected building.
Supports building groups with per-group settings and named presets.
"""

import os
import sys
import re
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Defaults (mirroring build_mod.py CLI defaults)
# ---------------------------------------------------------------------------

GROUP_SETTINGS_KEYS = [
    'transparency', 'dither_bottom', 'edge_inset', 'gradient_height',
    'contour_width', 'contour_color', 'no_outline', 'outline_value',
    'outline_thickness',
]

DEFAULT_GROUP_SETTINGS = {
    'transparency': 50,
    'dither_bottom': False,
    'edge_inset': 3,
    'gradient_height': 0,
    'contour_width': 0,
    'contour_color': 'Team color',
    'no_outline': False,
    'outline_value': 200,
    'outline_thickness': 4,
}

DEFAULTS = {
    'edge_inset': 3,
    'outline_value': 200,
    'outline_thickness': 4,
    'gradient_height': 0,
    'no_outline': False,
    'workers': cpu_count(),
    'dither_intensity': 8,
    'dither_bottom': False,
}

PRESETS_FILE = Path(__file__).parent / 'presets.json'

BUILDING_TYPES = [
    'house', 'outpost', 'tower', 'mill', 'lumber_camp', 'mining_camp',
    'mule_cart', 'donjon', 'folwark', 'dock', 'shipyard', 'barracks',
    'archery_range', 'stable', 'blacksmith', 'monastery', 'krepost',
    'settlement', 'siege_workshop', 'market', 'university', 'town_center',
    'castle', 'caravanserai', 'trade_workshop', 'wonder', 'wall',
]

TEAM_COLORS = {
    'Blue':   (0, 0, 255),
    'Red':    (255, 0, 0),
    'Green':  (0, 255, 0),
    'Yellow': (255, 255, 0),
    'Cyan':   (0, 255, 255),
    'Purple': (255, 0, 255),
    'Grey':   (128, 128, 128),
    'Orange': (255, 128, 0),
}

BACKGROUNDS = {
    'Grass':          ('terrain', 'grass'),
    'Water':          ('terrain', 'water'),
    'Dark (#222)':    ('solid', (0x22, 0x22, 0x22)),
    'Light (#ccc)':   ('solid', (0xcc, 0xcc, 0xcc)),
    'White (#fff)':   ('solid', (0xff, 0xff, 0xff)),
}

# Bayer intensity levels: 17 discrete steps (0-16)
# Map slider 0-100% to nearest Bayer level
BAYER_LEVELS = 17  # 0 through 16


def _pct_to_bayer(pct):
    """Convert 0-100 percentage to Bayer threshold 0-16."""
    return round(pct * 16 / 100)


def _bayer_to_pct(level):
    """Convert Bayer threshold 0-16 to display percentage."""
    return round(level * 100 / 16)


def _load_presets():
    """Load presets from disk."""
    if PRESETS_FILE.exists():
        try:
            with open(PRESETS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_presets(presets):
    """Save presets to disk."""
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=2)


def _filename_to_type(filename):
    """Determine which BUILDING_TYPE a filename belongs to.

    Returns the matching type string or None.
    """
    base = filename.replace('_x1.sld', '').replace('_x2.sld', '').replace('.sld', '')
    for btype in BUILDING_TYPES:
        if f'_{btype}_' in base or base.endswith(f'_{btype}'):
            return btype
    return None


def _building_display_name(filename):
    """Convert SLD filename to human-readable name.

    b_west_house_age2_x1.sld -> House (West, Age 2)
    """
    base = filename.replace('.sld', '')
    # Remove b_ prefix
    if base.startswith('b_'):
        base = base[2:]

    # Extract scale
    scale = ''
    for s in ('_x1', '_x2'):
        if base.endswith(s):
            base = base[:-3]
            scale = s[1:]
            break

    # Extract style (first segment)
    parts = base.split('_', 1)
    if len(parts) == 2:
        style, rest = parts
    else:
        style, rest = '', parts[0]

    # Extract age
    age_match = re.search(r'_age(\d)$', rest)
    age = ''
    if age_match:
        age = f'Age {age_match.group(1)}'
        rest = rest[:age_match.start()]

    # Build display name
    building = rest.replace('_', ' ').title()
    meta = ', '.join(x for x in [style.title(), age] if x)
    return f"{building} ({meta})" if meta else building


class TransparentBuildingsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AoE2 Transparent Buildings")
        self.root.resizable(True, True)

        self.build_thread = None
        self._preview_pending = None  # ID from root.after()
        self._sld_cache = {}  # filename -> bytes
        self._cached_source = None  # PIL Image of side-by-side composite
        self._cached_bg_tile = None
        self._loading_group = False  # flag to prevent save-while-loading

        # Groups: list of dicts, each with 'name', 'buildings' (list of types), 'settings'
        # "Default" group always exists and catches unassigned buildings
        self._groups = [{'name': 'Default', 'buildings': [], 'settings': dict(DEFAULT_GROUP_SETTINGS)}]
        self._current_group_idx = 0

        # Try to detect paths up front
        self._detect_paths()
        self._find_preview_buildings()

        self._build_ui()

        # Initial preview after UI is built
        self._update_building_index_label()
        self._update_group_buildings_label()
        self.root.after(200, self._render_preview)

    def _detect_paths(self):
        """Try to auto-detect game and mod paths."""
        self.graphics_dir = None
        self.mod_graphics_dir = None
        try:
            from paths import get_graphics_dir, get_mod_graphics_dir
            self.graphics_dir = get_graphics_dir()
            self.mod_graphics_dir = get_mod_graphics_dir()
        except Exception:
            pass

    def _find_preview_buildings(self):
        """Find x1 building files for the preview dropdown."""
        self._building_files = []
        self._building_names = []
        if not self.graphics_dir:
            return
        try:
            from build_mod import find_building_files
            all_files = find_building_files(exclude=[])
            # Filter to x1 only for preview
            x1_files = [f for f in all_files if '_x1.' in f]
            self._building_files = x1_files
            self._building_names = [_building_display_name(f) for f in x1_files]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 4}

        # Top-level resizable two-panel layout
        paned = ttk.PanedWindow(self.root, orient='horizontal')
        paned.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left panel: settings
        left = ttk.Frame(paned, width=400)

        # Right panel: preview (flexible)
        right = ttk.Frame(paned)

        paned.add(left, weight=0)
        paned.add(right, weight=1)

        # Make left panel scrollable
        left_canvas = tk.Canvas(left, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left, orient='vertical', command=left_canvas.yview)
        self._left_inner = ttk.Frame(left_canvas)
        self._left_inner.bind('<Configure>',
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox('all')))
        self._left_win_id = left_canvas.create_window((0, 0), window=self._left_inner, anchor='nw')
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.pack(side='left', fill='both', expand=True)
        left_scroll.pack(side='right', fill='y')

        # Make inner frame stretch to canvas width + update wraplengths
        self._wraplength_labels = []  # populated during UI build
        def _on_canvas_resize(event):
            left_canvas.itemconfigure(self._left_win_id, width=event.width)
            wl = max(100, event.width - 30)
            for lbl in self._wraplength_labels:
                try:
                    lbl.configure(wraplength=wl)
                except Exception:
                    pass
        left_canvas.bind('<Configure>', _on_canvas_resize)

        # Mouse wheel scrolling on left panel
        self._left_canvas = left_canvas
        def _on_left_scroll(event):
            left_canvas.yview_scroll(int(-event.delta / 120), 'units')
            return 'break'
        left_canvas.bind('<MouseWheel>', _on_left_scroll)
        self._left_inner.bind('<MouseWheel>', _on_left_scroll)
        self._left_scroll_handler = _on_left_scroll

        left = self._left_inner

        # ================= LEFT PANEL =================
        row = 0

        # Make slider column expand
        left.columnconfigure(1, weight=1)

        # --- Title ---
        title = ttk.Label(left, text="Transparent Buildings", font=('Segoe UI', 16, 'bold'))
        title.grid(row=row, column=0, columnspan=3, **pad, sticky='w')
        row += 1

        subtitle = ttk.Label(left, text="Configure settings and hit Build.",
                             foreground='#666666')
        subtitle.grid(row=row, column=0, columnspan=3, **pad, sticky='w')
        row += 1

        # --- Presets section ---
        row = self._section(left, row, "Presets")

        preset_frame = ttk.Frame(left)
        preset_frame.grid(row=row, column=0, columnspan=3, **pad, sticky='ew')
        ttk.Label(preset_frame, text="Preset:").pack(side='left', padx=(0, 4))
        self._preset_var = tk.StringVar(value='Default')
        self._preset_combo = ttk.Combobox(preset_frame, textvariable=self._preset_var,
                                          state='readonly', width=20)
        self._preset_combo.pack(side='left', padx=(0, 4))
        self._preset_combo.bind('<<ComboboxSelected>>', lambda e: self._load_preset())
        ttk.Button(preset_frame, text="Save", width=5,
                   command=self._save_preset).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="Delete", width=6,
                   command=self._delete_preset).pack(side='left', padx=2)
        self._refresh_preset_list()
        row += 1

        # --- Groups section ---
        row = self._section(left, row, "Settings Group")

        group_frame = ttk.Frame(left)
        group_frame.grid(row=row, column=0, columnspan=3, **pad, sticky='ew')
        ttk.Label(group_frame, text="Group:").pack(side='left', padx=(0, 4))
        self._group_var = tk.StringVar(value='Default')
        self._group_combo = ttk.Combobox(group_frame, textvariable=self._group_var,
                                         state='readonly', width=16)
        self._group_combo.pack(side='left', padx=(0, 4))
        self._group_combo.bind('<<ComboboxSelected>>', lambda e: self._on_group_selected())
        ttk.Button(group_frame, text="+", width=2,
                   command=self._add_group).pack(side='left', padx=2)
        ttk.Button(group_frame, text="-", width=2,
                   command=self._delete_group).pack(side='left', padx=2)
        ttk.Button(group_frame, text="\u25b2", width=2,
                   command=self._move_group_up).pack(side='left', padx=1)
        ttk.Button(group_frame, text="\u25bc", width=2,
                   command=self._move_group_down).pack(side='left', padx=1)
        ttk.Button(group_frame, text="Rename", width=6,
                   command=self._rename_group).pack(side='left', padx=2)
        self._refresh_group_list()
        row += 1

        # Building types assigned to this group
        self._group_buildings_label = ttk.Label(left, text="Buildings: All unassigned",
                                                 foreground='#666666', wraplength=350)
        self._group_buildings_label.grid(row=row, column=0, columnspan=3, **pad, sticky='ew')
        self._wraplength_labels.append(self._group_buildings_label)
        row += 1

        assign_frame = ttk.Frame(left)
        assign_frame.grid(row=row, column=0, columnspan=3, **pad, sticky='ew')
        ttk.Button(assign_frame, text="Assign Buildings...",
                   command=self._assign_buildings_dialog).pack(side='left')
        row += 1

        ttk.Separator(left, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky='ew', pady=8)
        row += 1

        # --- Dithering section ---
        row = self._section(left, row, "Dithering")

        row = self._slider(left, row, "Transparency",
                           "Pixel density of dithering pattern (0% = opaque, 100% = fully transparent)",
                           'transparency', 0, 100,
                           _bayer_to_pct(DEFAULTS['dither_intensity']),
                           preview=True, snap_bayer=True)

        self.dither_bottom_var = tk.BooleanVar(value=DEFAULTS['dither_bottom'])
        cb = ttk.Checkbutton(left, text="Dither below foundation",
                             variable=self.dither_bottom_var,
                             command=self._schedule_preview)
        cb.grid(row=row, column=0, columnspan=3, **pad, sticky='w')
        tip = ttk.Label(left, text="Also apply dithering below the foundation line",
                        foreground='#999999', font=('Segoe UI', 8))
        tip.grid(row=row + 1, column=1, columnspan=2, padx=8, sticky='w')
        row += 2

        row = self._slider(left, row, "Edge inset (px)",
                           "Pixels from building edge kept opaque (auto-scaled for UHD)",
                           'edge_inset', 0, 10, DEFAULTS['edge_inset'], preview=True)

        row = self._slider(left, row, "Gradient height (px)",
                           "Transition zone above foundation (0 = sharp cutoff)",
                           'gradient_height', 0, 32, DEFAULTS['gradient_height'],
                           preview=True)

        row = self._slider(left, row, "Contour (px)",
                           "Outline around building silhouette (0 = off)",
                           'contour_width', 0, 6, 0, preview=True)

        # Contour color
        contour_color_frame = ttk.Frame(left)
        contour_color_frame.grid(row=row, column=0, columnspan=3, padx=8, pady=2, sticky='w')
        ttk.Label(contour_color_frame, text="Contour color:").pack(side='left', padx=(0, 8))
        self.contour_color_var = tk.StringVar(value='Team color')
        for opt in ('Team color', 'Black'):
            ttk.Radiobutton(contour_color_frame, text=opt, variable=self.contour_color_var,
                           value=opt, command=self._schedule_preview).pack(side='left', padx=4)
        row += 1

        # --- Outline section ---
        row = self._section(left, row, "Foundation Outline")

        self.no_outline_var = tk.BooleanVar(value=DEFAULTS['no_outline'])
        cb = ttk.Checkbutton(left, text="Disable outline", variable=self.no_outline_var,
                             command=self._on_toggle_outline)
        cb.grid(row=row, column=0, columnspan=3, **pad, sticky='w')
        row += 1

        row = self._slider(left, row, "Outline brightness",
                           "0 = black, 255 = white",
                           'outline_value', 0, 255, DEFAULTS['outline_value'],
                           preview=True)

        row = self._slider(left, row, "Outline thickness (px)",
                           "Height of the outline band",
                           'outline_thickness', 1, 12, DEFAULTS['outline_thickness'],
                           preview=True)

        # Inclusion set: all x1 filenames included by default
        # (managed via Default group's Assign Buildings dialog)
        self._included_files = set(self._building_files)

        # --- Scale section ---
        row = self._section(left, row, "Build Scale")

        scale_frame = ttk.Frame(left)
        scale_frame.grid(row=row, column=0, columnspan=3, **pad, sticky='w')
        self.build_x1_var = tk.BooleanVar(value=True)
        self.build_x2_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scale_frame, text="x1 (Standard)",
                        variable=self.build_x1_var).pack(side='left', padx=(0, 16))
        ttk.Checkbutton(scale_frame, text="x2 (UHD)",
                        variable=self.build_x2_var).pack(side='left')
        row += 1

        # --- Performance section ---
        row = self._section(left, row, "Performance")

        row = self._slider(left, row, "Workers",
                           "Parallel workers for processing",
                           'workers', 1, cpu_count(), DEFAULTS['workers'])

        # --- Output dir ---
        ttk.Separator(left, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky='ew', pady=8)
        row += 1

        ttk.Label(left, text="Output directory:").grid(row=row, column=0, **pad, sticky='w')
        self.output_var = tk.StringVar(value=self.mod_graphics_dir or '')
        out_entry = ttk.Entry(left, textvariable=self.output_var)
        out_entry.grid(row=row, column=1, **pad, sticky='ew')
        ttk.Button(left, text="Browse...", command=self._browse_output).grid(
            row=row, column=2, **pad)
        row += 1

        # --- Combine with other mods ---
        ttk.Separator(left, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky='ew', pady=8)
        row += 1

        combine_header = ttk.Frame(left)
        combine_header.grid(row=row, column=0, columnspan=3, **pad, sticky='ew')
        ttk.Label(combine_header, text="Combine with mods:").pack(side='left')
        ttk.Button(combine_header, text="Refresh", command=self._refresh_combine_list).pack(side='right')
        row += 1

        self._local_mods = []  # list of (name, mod_root)
        self._selected_combine_mods = []  # ordered list of (name, mod_root)
        self._refresh_local_mods()

        combine_outer = ttk.Frame(left)
        combine_outer.grid(row=row, column=0, columnspan=3, **pad, sticky='ew')

        # Available mods (left)
        avail_frame = ttk.LabelFrame(combine_outer, text="Available")
        avail_frame.pack(side='left', fill='both', expand=True)
        self.combine_avail_listbox = tk.Listbox(avail_frame, height=5,
                                                exportselection=False)
        self.combine_avail_listbox.pack(side='left', fill='both', expand=True)
        avail_scroll = ttk.Scrollbar(avail_frame, orient='vertical',
                                     command=self.combine_avail_listbox.yview)
        avail_scroll.pack(side='right', fill='y')
        self.combine_avail_listbox.configure(yscrollcommand=avail_scroll.set)

        # Add/Remove buttons (center)
        btn_mid = ttk.Frame(combine_outer)
        btn_mid.pack(side='left', padx=4)
        ttk.Button(btn_mid, text="Add >", width=8, command=self._combine_add).pack(pady=2)
        ttk.Button(btn_mid, text="< Remove", width=8, command=self._combine_remove).pack(pady=2)

        # Selected mods with priority (right)
        sel_frame = ttk.LabelFrame(combine_outer, text="Selected (priority order)")
        sel_frame.pack(side='left', fill='both', expand=True)
        self.combine_sel_listbox = tk.Listbox(sel_frame, height=5,
                                              exportselection=False)
        self.combine_sel_listbox.pack(side='left', fill='both', expand=True)
        sel_scroll = ttk.Scrollbar(sel_frame, orient='vertical',
                                   command=self.combine_sel_listbox.yview)
        sel_scroll.pack(side='right', fill='y')
        self.combine_sel_listbox.configure(yscrollcommand=sel_scroll.set)

        # Up/Down buttons for priority
        btn_prio = ttk.Frame(combine_outer)
        btn_prio.pack(side='left', padx=4)
        ttk.Button(btn_prio, text="Up", width=5, command=self._combine_move_up).pack(pady=2)
        ttk.Button(btn_prio, text="Down", width=5, command=self._combine_move_down).pack(pady=2)

        self._populate_combine_available()
        row += 1

        self.combine_info = ttk.Label(left, text="Higher priority mods override lower ones",
                                      foreground='#666666', wraplength=350)
        self.combine_info.grid(row=row, column=0, columnspan=3, **pad, sticky='w')
        self._wraplength_labels.append(self.combine_info)
        row += 1

        # --- Status label ---
        if not self.graphics_dir:
            status_text = "Warning: Could not detect AoE2 graphics directory."
            status_color = '#cc4400'
        else:
            status_text = f"Game graphics: {self.graphics_dir}"
            status_color = '#006600'

        self.path_status = ttk.Label(left, text=status_text, foreground=status_color,
                                     wraplength=350)
        self.path_status.grid(row=row, column=0, columnspan=3, **pad, sticky='ew')
        self._wraplength_labels.append(self.path_status)
        row += 1

        # --- Build button & progress ---
        ttk.Separator(left, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky='ew', pady=8)
        row += 1

        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=row, column=0, columnspan=3, **pad, sticky='ew')

        self.build_btn = ttk.Button(btn_frame, text="Build Mod", command=self._on_build)
        self.build_btn.pack(side='left')

        self.status_label = ttk.Label(btn_frame, text="", foreground='#666666')
        self.status_label.pack(side='left', padx=12)
        row += 1

        self.progress = ttk.Progressbar(left, mode='determinate')
        self.progress.grid(row=row, column=0, columnspan=3, **pad, sticky='ew')
        row += 1

        # --- Log area ---
        self.log_text = tk.Text(left, height=8, state='disabled',
                                font=('Consolas', 9), bg='#1e1e1e', fg='#cccccc')
        self.log_text.grid(row=row, column=0, columnspan=3, **pad, sticky='ew')
        row += 1

        # ================= RIGHT PANEL (Preview) =================
        self._build_preview_panel(right)

        # Bind scroll to all left panel children
        self._bind_scroll_to_children(self._left_inner)

    def _bind_scroll_to_children(self, widget):
        """Recursively bind mousewheel scrolling to all children of the left panel."""
        for child in widget.winfo_children():
            child.bind('<MouseWheel>', self._left_scroll_handler)
            self._bind_scroll_to_children(child)

    def _build_preview_panel(self, parent):
        """Build the preview panel on the right side."""
        pad = {'padx': 8, 'pady': 4}

        ttk.Label(parent, text="Preview", font=('Segoe UI', 14, 'bold')).grid(
            row=0, column=0, columnspan=3, **pad, sticky='w')

        # Preview controls row
        ctrl = ttk.Frame(parent)
        ctrl.grid(row=1, column=0, columnspan=3, **pad, sticky='ew')

        # Building picker with prev/next buttons
        ttk.Label(ctrl, text="Building:").pack(side='left', padx=(0, 4))

        ttk.Button(ctrl, text="<", width=2,
                   command=self._prev_building).pack(side='left')

        self.building_var = tk.StringVar()
        self._building_combo = ttk.Combobox(ctrl, textvariable=self.building_var,
                                            values=self._building_names,
                                            width=30)
        self._building_combo.pack(side='left', padx=(2, 2))
        self._building_combo.bind('<<ComboboxSelected>>', lambda e: (self._update_building_index_label(), self._schedule_preview()))
        self._building_combo.bind('<Return>', lambda e: self._on_building_typed())
        self._building_combo.bind('<KeyRelease>', lambda e: self._filter_buildings(e))

        ttk.Button(ctrl, text=">", width=2,
                   command=self._next_building).pack(side='left', padx=(0, 12))

        # Set default to house if available
        default_file = 'b_west_house_age2_x1.sld'
        if default_file in self._building_files:
            idx = self._building_files.index(default_file)
            self._building_combo.current(idx)
        elif self._building_names:
            self._building_combo.current(0)

        # Keyboard shortcuts: Left/Right arrow to cycle buildings
        # (only when not focused on an entry/combobox)
        def _nav_key(e, fn):
            if not isinstance(e.widget, (ttk.Entry, ttk.Combobox, tk.Entry)):
                fn()
        self.root.bind('<Left>', lambda e: _nav_key(e, self._prev_building))
        self.root.bind('<Right>', lambda e: _nav_key(e, self._next_building))

        # Team color
        ttk.Label(ctrl, text="Color:").pack(side='left', padx=(0, 4))
        self.team_color_var = tk.StringVar(value='Blue')
        color_combo = ttk.Combobox(ctrl, textvariable=self.team_color_var,
                                   values=list(TEAM_COLORS.keys()),
                                   state='readonly', width=8)
        color_combo.pack(side='left', padx=(0, 12))
        color_combo.bind('<<ComboboxSelected>>', lambda e: self._schedule_preview())

        # Background terrain
        ttk.Label(ctrl, text="Terrain:").pack(side='left', padx=(0, 4))
        self.bg_var = tk.StringVar(value='Grass')
        bg_combo = ttk.Combobox(ctrl, textvariable=self.bg_var,
                                values=list(BACKGROUNDS.keys()),
                                state='readonly', width=14)
        bg_combo.pack(side='left', padx=(0, 8))
        bg_combo.bind('<<ComboboxSelected>>', lambda e: self._schedule_preview())

        # Preview scale (x1 or x2)
        ttk.Label(ctrl, text="Scale:").pack(side='left', padx=(0, 4))
        self._preview_scale_var = tk.StringVar(value='x2')
        preview_scale_combo = ttk.Combobox(ctrl, textvariable=self._preview_scale_var,
                                           values=['x1', 'x2'], state='readonly', width=3)
        preview_scale_combo.pack(side='left', padx=(0, 8))
        preview_scale_combo.bind('<<ComboboxSelected>>', lambda e: self._schedule_preview())

        # Show before toggle
        self._show_before_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Before", variable=self._show_before_var,
                        command=self._schedule_preview).pack(side='left', padx=(8, 0))

        # Building index label + zoom + arrow key hint
        nav_frame = ttk.Frame(parent)
        nav_frame.grid(row=2, column=0, columnspan=3, padx=8, sticky='ew')
        self._building_index_label = ttk.Label(nav_frame, text="",
                                                foreground='#999999',
                                                font=('Segoe UI', 8))
        self._building_index_label.pack(side='left')
        self._building_group_label = ttk.Label(nav_frame, text="",
                                                foreground='#4488cc',
                                                font=('Segoe UI', 8))
        self._building_group_label.pack(side='left', padx=(12, 0))
        ttk.Label(nav_frame, text="Arrow keys: prev/next building",
                  foreground='#999999', font=('Segoe UI', 8)).pack(side='right')

        # Zoom controls
        zoom_frame = ttk.Frame(parent)
        zoom_frame.grid(row=3, column=0, columnspan=3, padx=8, sticky='ew')
        ttk.Label(zoom_frame, text="Zoom:").pack(side='left', padx=(0, 4))
        self._zoom_var = tk.DoubleVar(value=1.0)
        self._zoom_label = ttk.Label(zoom_frame, text="Fit", width=5)
        self._zoom_label.pack(side='right', padx=(4, 0))
        zoom_slider = ttk.Scale(zoom_frame, from_=0.5, to=6.0, variable=self._zoom_var,
                                orient='horizontal', length=150,
                                command=self._on_zoom_change)
        zoom_slider.pack(side='left', fill='x', expand=True, padx=4)
        ttk.Button(zoom_frame, text="Fit", width=3,
                   command=self._zoom_fit).pack(side='left', padx=(0, 4))

        # Preview image area
        self.preview_label = ttk.Label(parent, text="Loading preview...",
                                       anchor='center', justify='center')
        self.preview_label.grid(row=4, column=0, columnspan=3, sticky='nsew',
                                padx=8, pady=8)
        parent.rowconfigure(4, weight=1)
        parent.columnconfigure(0, weight=1)

        # Track preview area size for dynamic scaling
        self._preview_w = 560
        self._preview_h = 420
        self.preview_label.bind('<Configure>', self._on_preview_resize)

        # Pan state: offset in source pixels
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start = None

        # Mouse drag to pan
        self.preview_label.bind('<ButtonPress-1>', self._on_pan_start)
        self.preview_label.bind('<B1-Motion>', self._on_pan_drag)
        self.preview_label.bind('<ButtonRelease-1>', self._on_pan_end)

        # Scroll wheel to zoom, Shift+scroll to pan vertically
        self.preview_label.bind('<MouseWheel>', self._on_scroll)

        # Keep reference to prevent GC
        self._preview_photo = None

    def _section(self, parent, row, text):
        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky='ew', pady=(12, 4))
        row += 1
        lbl = ttk.Label(parent, text=text, font=('Segoe UI', 11, 'bold'))
        lbl.grid(row=row, column=0, columnspan=3, padx=8, pady=(0, 4), sticky='w')
        row += 1
        return row

    def _slider(self, parent, row, label, tooltip, attr, lo, hi, default,
                preview=False, snap_bayer=False):
        """Add a labeled slider. Stores the IntVar as self.<attr>_var."""
        ttk.Label(parent, text=label).grid(row=row, column=0, padx=8, pady=2, sticky='w')

        var = tk.IntVar(value=default)
        setattr(self, f'{attr}_var', var)

        val_label = ttk.Label(parent, text=str(default), width=6, anchor='e')
        val_label.grid(row=row, column=2, padx=8, pady=2, sticky='e')

        def on_change(v, vl=val_label, va=var, sb=snap_bayer, pv=preview):
            val = va.get()
            if sb:
                # Snap to nearest Bayer level and show snapped %
                bayer = _pct_to_bayer(val)
                snapped = _bayer_to_pct(bayer)
                vl.configure(text=f"{snapped}%")
            else:
                vl.configure(text=str(val))
            if pv:
                self._schedule_preview()

        slider = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient='horizontal',
                           command=on_change)
        slider.grid(row=row, column=1, padx=8, pady=2, sticky='ew')

        # Store references for enable/disable
        setattr(self, f'{attr}_slider', slider)
        setattr(self, f'{attr}_label', val_label)

        if tooltip:
            tip = ttk.Label(parent, text=tooltip, foreground='#999999', font=('Segoe UI', 8))
            tip.grid(row=row + 1, column=1, columnspan=2, padx=8, sticky='w')
            return row + 2

        return row + 1

    def _get_building_index(self):
        """Get current building index, or 0."""
        sel = self.building_var.get()
        if sel in self._building_names:
            return self._building_names.index(sel)
        return 0

    def _update_building_index_label(self):
        if self._building_names:
            idx = self._get_building_index()
            self._building_index_label.configure(
                text=f"{idx + 1} / {len(self._building_names)}")
            # Show which group this building belongs to
            if idx < len(self._building_files):
                group = self._get_group_for_building(self._building_files[idx])
                self._building_group_label.configure(text=f"Group: {group['name']}")

    def _prev_building(self):
        """Select previous building in the dropdown."""
        if not self._building_names:
            return
        idx = (self._get_building_index() - 1) % len(self._building_names)
        self._building_combo.current(idx)
        self._update_building_index_label()
        self._schedule_preview()

    def _next_building(self):
        """Select next building in the dropdown."""
        if not self._building_names:
            return
        idx = (self._get_building_index() + 1) % len(self._building_names)
        self._building_combo.current(idx)
        self._update_building_index_label()
        self._schedule_preview()

    def _filter_buildings(self, event):
        """Filter building dropdown as user types."""
        # Don't filter on navigation keys
        if event.keysym in ('Left', 'Right', 'Up', 'Down', 'Return', 'Tab',
                            'Escape', 'BackSpace', 'Delete'):
            return
        typed = self.building_var.get().lower()
        if not typed:
            self._building_combo['values'] = self._building_names
            return
        filtered = [n for n in self._building_names if typed in n.lower()]
        self._building_combo['values'] = filtered

    def _on_building_typed(self):
        """Handle Enter key in building combobox — select first match."""
        typed = self.building_var.get().lower()
        for i, name in enumerate(self._building_names):
            if typed in name.lower():
                self._building_combo['values'] = self._building_names
                self._building_combo.set(name)
                self._update_building_index_label()
                self._schedule_preview()
                return

    def _on_zoom_change(self, val):
        zoom = self._zoom_var.get()
        if zoom <= 0.6:
            self._zoom_label.configure(text="Fit")
        else:
            self._zoom_label.configure(text=f"{zoom:.1f}x")
        self._schedule_viewport()

    def _zoom_fit(self):
        """Reset zoom and pan to fit mode."""
        self._zoom_var.set(0.5)
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._zoom_label.configure(text="Fit")
        self._schedule_viewport()

    def _on_pan_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_pan_drag(self, event):
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        zoom = self._zoom_var.get()
        if zoom > 0.6:
            self._pan_x -= dx / zoom
            self._pan_y -= dy / zoom
            self._schedule_viewport()

    def _on_pan_end(self, event):
        self._drag_start = None

    def _on_scroll(self, event):
        """Scroll wheel: zoom in/out."""
        zoom = self._zoom_var.get()
        delta = event.delta / 120  # Windows: 120 per notch
        new_zoom = max(0.5, min(6.0, zoom + delta * 0.3))
        self._zoom_var.set(new_zoom)
        if new_zoom <= 0.6:
            self._pan_x = 0.0
            self._pan_y = 0.0
            self._zoom_label.configure(text="Fit")
        else:
            self._zoom_label.configure(text=f"{new_zoom:.1f}x")
        self._schedule_viewport()

    def _on_preview_resize(self, event):
        """Track preview label size and re-render if it changed significantly."""
        new_w = max(200, event.width)
        new_h = max(150, event.height)
        if abs(new_w - self._preview_w) > 20 or abs(new_h - self._preview_h) > 20:
            self._preview_w = new_w
            self._preview_h = new_h
            self._schedule_preview()

    def _on_toggle_outline(self):
        state = 'disabled' if self.no_outline_var.get() else '!disabled'
        for attr in ('outline_value', 'outline_thickness'):
            getattr(self, f'{attr}_slider').state([state])
        self._schedule_preview()

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.output_var.set(d)

    def _refresh_local_mods(self):
        """Refresh the list of available mods (local and subscribed)."""
        try:
            from paths import list_available_mods
            self._local_mods = list_available_mods()
        except Exception:
            self._local_mods = []

    def _populate_combine_available(self):
        """Fill the available mods listbox, excluding already-selected ones."""
        self.combine_avail_listbox.delete(0, 'end')
        selected_names = {name for name, _ in self._selected_combine_mods}
        for name, _ in self._local_mods:
            if name not in selected_names:
                self.combine_avail_listbox.insert('end', name)

    def _populate_combine_selected(self):
        """Fill the selected mods listbox in priority order."""
        self.combine_sel_listbox.delete(0, 'end')
        for name, _ in self._selected_combine_mods:
            self.combine_sel_listbox.insert('end', name)

    def _refresh_combine_list(self):
        """Refresh both combine listboxes with current mods."""
        self._refresh_local_mods()
        # Remove selected mods that no longer exist
        available_names = {name for name, _ in self._local_mods}
        self._selected_combine_mods = [
            (n, r) for n, r in self._selected_combine_mods if n in available_names
        ]
        self._populate_combine_available()
        self._populate_combine_selected()

    def _combine_add(self):
        """Add selected available mod to the combine list."""
        sel = self.combine_avail_listbox.curselection()
        if not sel:
            return
        name = self.combine_avail_listbox.get(sel[0])
        for mname, mod_root in self._local_mods:
            if mname == name:
                self._selected_combine_mods.append((mname, mod_root))
                break
        self._populate_combine_available()
        self._populate_combine_selected()
        self._sld_cache.clear()
        self._schedule_preview()

    def _combine_remove(self):
        """Remove selected mod from the combine list."""
        sel = self.combine_sel_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < len(self._selected_combine_mods):
            self._selected_combine_mods.pop(idx)
        self._populate_combine_available()
        self._populate_combine_selected()
        self._sld_cache.clear()
        self._schedule_preview()

    def _combine_move_up(self):
        """Move selected combine mod up in priority."""
        sel = self.combine_sel_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        idx = sel[0]
        self._selected_combine_mods[idx - 1], self._selected_combine_mods[idx] = \
            self._selected_combine_mods[idx], self._selected_combine_mods[idx - 1]
        self._populate_combine_selected()
        self.combine_sel_listbox.selection_set(idx - 1)
        self._sld_cache.clear()
        self._schedule_preview()

    def _combine_move_down(self):
        """Move selected combine mod down in priority."""
        sel = self.combine_sel_listbox.curselection()
        if not sel or sel[0] >= len(self._selected_combine_mods) - 1:
            return
        idx = sel[0]
        self._selected_combine_mods[idx], self._selected_combine_mods[idx + 1] = \
            self._selected_combine_mods[idx + 1], self._selected_combine_mods[idx]
        self._populate_combine_selected()
        self.combine_sel_listbox.selection_set(idx + 1)
        self._sld_cache.clear()
        self._schedule_preview()

    def _get_combine_mod_roots(self):
        """Return list of selected combine mod root directories in priority order."""
        return [mod_root for _, mod_root in self._selected_combine_mods]

    # ------------------------------------------------------------------
    # Groups & Presets
    # ------------------------------------------------------------------

    def _get_current_group_settings(self):
        """Snapshot current UI settings into a dict."""
        return {
            'transparency': self.transparency_var.get(),
            'dither_bottom': self.dither_bottom_var.get(),
            'edge_inset': self.edge_inset_var.get(),
            'gradient_height': self.gradient_height_var.get(),
            'contour_width': self.contour_width_var.get(),
            'contour_color': self.contour_color_var.get(),
            'no_outline': self.no_outline_var.get(),
            'outline_value': self.outline_value_var.get(),
            'outline_thickness': self.outline_thickness_var.get(),
        }

    def _apply_group_settings(self, settings):
        """Set UI controls from a settings dict (without triggering save-back)."""
        self._loading_group = True
        try:
            self.transparency_var.set(settings.get('transparency', 50))
            self.dither_bottom_var.set(settings.get('dither_bottom', False))
            self.edge_inset_var.set(settings.get('edge_inset', 3))
            self.gradient_height_var.set(settings.get('gradient_height', 0))
            self.contour_width_var.set(settings.get('contour_width', 0))
            self.contour_color_var.set(settings.get('contour_color', 'Team color'))
            self.no_outline_var.set(settings.get('no_outline', False))
            self.outline_value_var.set(settings.get('outline_value', 200))
            self.outline_thickness_var.set(settings.get('outline_thickness', 4))
            # Update slider display labels
            for attr in ('transparency', 'edge_inset', 'gradient_height',
                         'contour_width', 'outline_value', 'outline_thickness'):
                label = getattr(self, f'{attr}_label', None)
                var = getattr(self, f'{attr}_var', None)
                if label and var:
                    val = var.get()
                    if attr == 'transparency':
                        bayer = _pct_to_bayer(val)
                        label.configure(text=f"{_bayer_to_pct(bayer)}%")
                    else:
                        label.configure(text=str(val))
            # Update outline enable/disable state
            self._on_toggle_outline()
        finally:
            self._loading_group = False

    def _save_current_group_settings(self):
        """Save current UI settings to the active group."""
        if self._loading_group:
            return
        if 0 <= self._current_group_idx < len(self._groups):
            self._groups[self._current_group_idx]['settings'] = self._get_current_group_settings()

    def _refresh_group_list(self):
        """Refresh the group dropdown values."""
        names = [g['name'] for g in self._groups]
        self._group_combo['values'] = names
        if 0 <= self._current_group_idx < len(self._groups):
            self._group_var.set(self._groups[self._current_group_idx]['name'])

    def _update_group_buildings_label(self):
        """Update the label showing which buildings are in the current group."""
        group = self._groups[self._current_group_idx]
        if group['name'] == 'Default':
            # Default covers all included buildings not assigned to other groups
            assigned = set()
            for g in self._groups:
                if g['name'] != 'Default':
                    assigned.update(g['buildings'])
            n_included = len(self._included_files)
            n_total = len(self._building_files)
            n_default = len(self._included_files - assigned)
            if n_included == n_total and not assigned:
                text = "Buildings: All selected"
            elif n_included == n_total:
                text = f"Buildings: {n_default} in Default, {len(assigned)} in other groups"
            else:
                text = f"Buildings: {n_default} in Default ({n_included}/{n_total} total selected)"
        else:
            n = len(group['buildings'])
            if n == 0:
                text = "Buildings: None assigned"
            elif n <= 5:
                names = [_building_display_name(f) for f in group['buildings']]
                text = f"Buildings ({n}): " + ", ".join(names)
            else:
                # Summarize by type
                types = {}
                for f in group['buildings']:
                    t = _filename_to_type(f) or 'other'
                    types[t] = types.get(t, 0) + 1
                summary = ", ".join(f"{t.replace('_',' ').title()} ({c})"
                                    for t, c in sorted(types.items()))
                text = f"Buildings ({n}): {summary}"
        self._group_buildings_label.configure(text=text)

    def _on_group_selected(self):
        """Handle group dropdown selection change."""
        name = self._group_var.get()
        # Save current group settings before switching
        self._save_current_group_settings()
        # Find and switch to selected group
        for i, g in enumerate(self._groups):
            if g['name'] == name:
                self._current_group_idx = i
                self._apply_group_settings(g['settings'])
                self._update_group_buildings_label()
                self._schedule_preview()
                return

    def _add_group(self):
        """Add a new building group."""
        name = simpledialog.askstring("New Group", "Group name:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()
        if any(g['name'] == name for g in self._groups):
            messagebox.showwarning("Duplicate", f"Group '{name}' already exists.")
            return
        # Save current group before adding
        self._save_current_group_settings()
        new_group = {
            'name': name,
            'buildings': [],
            'settings': dict(DEFAULT_GROUP_SETTINGS),
        }
        self._groups.append(new_group)
        self._current_group_idx = len(self._groups) - 1
        self._refresh_group_list()
        self._apply_group_settings(new_group['settings'])
        self._update_group_buildings_label()
        self._schedule_preview()

    def _delete_group(self):
        """Delete the current group (can't delete Default)."""
        if self._current_group_idx == 0:
            messagebox.showinfo("Info", "Cannot delete the Default group.")
            return
        name = self._groups[self._current_group_idx]['name']
        if not messagebox.askyesno("Delete Group", f"Delete group '{name}'?"):
            return
        self._groups.pop(self._current_group_idx)
        self._current_group_idx = 0
        self._refresh_group_list()
        self._apply_group_settings(self._groups[0]['settings'])
        self._update_group_buildings_label()
        self._schedule_preview()

    def _move_group_up(self):
        """Move the current group up in priority (earlier = higher priority)."""
        idx = self._current_group_idx
        if idx <= 1:  # Can't move Default (0) or first non-default (1)
            return
        self._save_current_group_settings()
        self._groups[idx], self._groups[idx - 1] = self._groups[idx - 1], self._groups[idx]
        self._current_group_idx = idx - 1
        self._refresh_group_list()

    def _move_group_down(self):
        """Move the current group down in priority (later = lower priority)."""
        idx = self._current_group_idx
        if idx == 0 or idx >= len(self._groups) - 1:  # Can't move Default or last
            return
        self._save_current_group_settings()
        self._groups[idx], self._groups[idx + 1] = self._groups[idx + 1], self._groups[idx]
        self._current_group_idx = idx + 1
        self._refresh_group_list()

    def _rename_group(self):
        """Rename the current group (can't rename Default)."""
        if self._current_group_idx == 0:
            messagebox.showinfo("Info", "Cannot rename the Default group.")
            return
        old = self._groups[self._current_group_idx]['name']
        name = simpledialog.askstring("Rename Group", "New name:", parent=self.root,
                                      initialvalue=old)
        if not name or not name.strip() or name.strip() == old:
            return
        name = name.strip()
        if any(g['name'] == name for g in self._groups):
            messagebox.showwarning("Duplicate", f"Group '{name}' already exists.")
            return
        self._groups[self._current_group_idx]['name'] = name
        self._refresh_group_list()

    def _choose_buildings_dialog(self):
        """Open a hierarchical dialog to choose which buildings to process.

        Also shows which group each building belongs to.
        Unchecked buildings are excluded from all processing.
        """
        # Build group membership lookup
        file_group = {}  # filename -> group name
        for g in self._groups:
            if g['name'] != 'Default':
                for f in g['buildings']:
                    file_group[f] = g['name']

        dlg = tk.Toplevel(self.root)
        dlg.title("Choose Buildings to Process")
        dlg.geometry("550x600")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text="Select which buildings to include in the build:",
                  font=('Segoe UI', 10, 'bold')).pack(padx=10, pady=(10, 2), anchor='w')
        ttk.Label(dlg, text="Click a type header to select/deselect all. "
                  "Group shown in blue.",
                  foreground='#999999', font=('Segoe UI', 8)).pack(padx=10, anchor='w')

        # Scrollable frame
        outer = ttk.Frame(dlg)
        outer.pack(fill='both', expand=True, padx=10, pady=5)
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        def _scroll(event):
            canvas.yview_scroll(int(-event.delta / 120), 'units')
        canvas.bind('<MouseWheel>', _scroll)
        inner.bind('<MouseWheel>', _scroll)

        # Group files by type
        type_to_files = {}
        for f in self._building_files:
            btype = _filename_to_type(f) or 'other'
            type_to_files.setdefault(btype, []).append(f)

        file_vars = {}
        type_vars = {}
        type_children = {}

        row = 0
        for btype in BUILDING_TYPES:
            files = type_to_files.get(btype, [])
            if not files:
                continue
            type_children[btype] = files

            selected = [f for f in files if f in self._included_files]
            header_var = tk.BooleanVar(value=len(selected) == len(files))
            type_vars[btype] = header_var

            header_frame = ttk.Frame(inner)
            header_frame.grid(row=row, column=0, sticky='ew', pady=(8, 2))
            header_frame.bind('<MouseWheel>', _scroll)
            row += 1

            def make_toggle(bt):
                def toggle():
                    val = type_vars[bt].get()
                    for f in type_children[bt]:
                        file_vars[f].set(val)
                return toggle

            type_label = btype.replace('_', ' ').title()
            ttk.Checkbutton(header_frame, text=f"  {type_label} ({len(files)})",
                           variable=header_var,
                           command=make_toggle(btype)).pack(side='left')

            for f in files:
                var = tk.BooleanVar(value=f in self._included_files)
                file_vars[f] = var
                display = _building_display_name(f)

                def make_update_header(bt):
                    def update():
                        all_on = all(file_vars[ff].get() for ff in type_children[bt])
                        type_vars[bt].set(all_on)
                    return update

                child_frame = ttk.Frame(inner)
                child_frame.grid(row=row, column=0, sticky='w', padx=(24, 0))
                child_frame.bind('<MouseWheel>', _scroll)
                ttk.Checkbutton(child_frame, text=display, variable=var,
                               command=make_update_header(btype)).pack(side='left')
                grp = file_group.get(f)
                if grp:
                    ttk.Label(child_frame, text=f"[{grp}]",
                             foreground='#4488cc', font=('Segoe UI', 8)).pack(side='left', padx=4)
                row += 1

        # Other
        other_files = type_to_files.get('other', [])
        if other_files:
            ttk.Label(inner, text="  Other", font=('Segoe UI', 9, 'bold')).grid(
                row=row, column=0, sticky='w', pady=(8, 2))
            row += 1
            for f in other_files:
                var = tk.BooleanVar(value=f in self._included_files)
                file_vars[f] = var
                display = _building_display_name(f)
                child_frame = ttk.Frame(inner)
                child_frame.grid(row=row, column=0, sticky='w', padx=(24, 0))
                child_frame.bind('<MouseWheel>', _scroll)
                ttk.Checkbutton(child_frame, text=display, variable=var).pack(side='left')
                row += 1

        # Select All / None buttons
        action_frame = ttk.Frame(dlg)
        action_frame.pack(padx=10, pady=(0, 5))

        def select_all():
            for v in file_vars.values():
                v.set(True)
            for tv in type_vars.values():
                tv.set(True)

        def select_none():
            for v in file_vars.values():
                v.set(False)
            for tv in type_vars.values():
                tv.set(False)

        ttk.Button(action_frame, text="Select All", command=select_all).pack(side='left', padx=4)
        ttk.Button(action_frame, text="Select None", command=select_none).pack(side='left', padx=4)

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(padx=10, pady=(0, 10))

        def apply():
            self._included_files = {f for f, v in file_vars.items() if v.get()}
            self._update_group_buildings_label()
            dlg.destroy()

        ttk.Button(btn_frame, text="OK", command=apply).pack(side='left', padx=4)
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(side='left', padx=4)

    def _assign_buildings_dialog(self):
        """Open a dialog to assign individual buildings to the current group.

        For the Default group, this controls which buildings are included in the
        build overall (unchecked = excluded entirely).

        For other groups, shows a treeview with building types as parent nodes
        and individual building files as children.
        """
        group = self._groups[self._current_group_idx]
        if group['name'] == 'Default':
            self._choose_buildings_dialog()
            return

        # Build a map of which files are taken by OTHER groups
        taken = {}  # filename -> group name
        for g in self._groups:
            if g['name'] != group['name'] and g['name'] != 'Default':
                for f in g['buildings']:
                    taken[f] = g['name']

        # Group files by building type
        type_to_files = {}
        for f in self._building_files:
            btype = _filename_to_type(f) or 'other'
            type_to_files.setdefault(btype, []).append(f)

        current_set = set(group['buildings'])

        dlg = tk.Toplevel(self.root)
        dlg.title(f"Assign Buildings — {group['name']}")
        dlg.geometry("550x600")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text=f"Select buildings for '{group['name']}':",
                  font=('Segoe UI', 10, 'bold')).pack(padx=10, pady=(10, 2), anchor='w')
        ttk.Label(dlg, text="Click a type header to select/deselect all. "
                  "Grey items are assigned to other groups.",
                  foreground='#999999', font=('Segoe UI', 8)).pack(padx=10, anchor='w')

        # Scrollable frame
        outer = ttk.Frame(dlg)
        outer.pack(fill='both', expand=True, padx=10, pady=5)
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        def _scroll(event):
            canvas.yview_scroll(int(-event.delta / 120), 'units')
        canvas.bind('<MouseWheel>', _scroll)
        inner.bind('<MouseWheel>', _scroll)

        # Track all checkbutton vars
        file_vars = {}  # filename -> BooleanVar
        type_vars = {}  # btype -> BooleanVar (header toggle)
        type_children = {}  # btype -> list of filenames

        row = 0
        for btype in BUILDING_TYPES:
            files = type_to_files.get(btype, [])
            if not files:
                continue
            type_children[btype] = files

            # Count how many are available (not taken) and selected
            available = [f for f in files if f not in taken]
            selected = [f for f in available if f in current_set]

            # Type header with checkbox
            header_var = tk.BooleanVar(value=len(selected) == len(available) and len(available) > 0)
            type_vars[btype] = header_var

            header_frame = ttk.Frame(inner)
            header_frame.grid(row=row, column=0, sticky='ew', pady=(8, 2))
            row += 1

            def make_toggle(bt):
                def toggle():
                    val = type_vars[bt].get()
                    for f in type_children[bt]:
                        if f not in taken:
                            file_vars[f].set(val)
                return toggle

            type_label = btype.replace('_', ' ').title()
            ttk.Checkbutton(header_frame, text=f"  {type_label} ({len(files)})",
                           variable=header_var,
                           command=make_toggle(btype),
                           style='Bold.TCheckbutton' if hasattr(dlg, '_bold_style') else ''
                           ).pack(side='left')
            # Make header bold
            ttk.Label(header_frame, text="", width=0).pack(side='left')

            # Individual buildings
            for f in files:
                var = tk.BooleanVar(value=f in current_set)
                file_vars[f] = var

                display = _building_display_name(f)
                child_frame = ttk.Frame(inner)
                child_frame.grid(row=row, column=0, sticky='w', padx=(24, 0))
                row += 1

                if f in taken:
                    ttk.Checkbutton(child_frame, text=display, variable=var,
                                   state='disabled').pack(side='left')
                    ttk.Label(child_frame, text=f"({taken[f]})",
                             foreground='#999999', font=('Segoe UI', 8)).pack(side='left', padx=4)
                else:
                    def make_update_header(bt):
                        def update():
                            avail = [ff for ff in type_children[bt] if ff not in taken]
                            all_on = all(file_vars[ff].get() for ff in avail)
                            type_vars[bt].set(all_on)
                        return update

                    ttk.Checkbutton(child_frame, text=display, variable=var,
                                   command=make_update_header(btype)).pack(side='left')

                child_frame.bind('<MouseWheel>', _scroll)

        # Handle 'other' type (files with no recognized type)
        other_files = type_to_files.get('other', [])
        if other_files:
            ttk.Label(inner, text="  Other", font=('Segoe UI', 9, 'bold')).grid(
                row=row, column=0, sticky='w', pady=(8, 2))
            row += 1
            for f in other_files:
                var = tk.BooleanVar(value=f in current_set)
                file_vars[f] = var
                display = _building_display_name(f)
                child_frame = ttk.Frame(inner)
                child_frame.grid(row=row, column=0, sticky='w', padx=(24, 0))
                row += 1
                if f in taken:
                    ttk.Checkbutton(child_frame, text=display, variable=var,
                                   state='disabled').pack(side='left')
                    ttk.Label(child_frame, text=f"({taken[f]})",
                             foreground='#999999', font=('Segoe UI', 8)).pack(side='left', padx=4)
                else:
                    ttk.Checkbutton(child_frame, text=display, variable=var).pack(side='left')

        # Select All / None buttons
        action_frame = ttk.Frame(dlg)
        action_frame.pack(padx=10, pady=(0, 5))

        def select_all():
            for f, v in file_vars.items():
                if f not in taken:
                    v.set(True)
            for tv in type_vars.values():
                tv.set(True)

        def select_none():
            for f, v in file_vars.items():
                if f not in taken:
                    v.set(False)
            for tv in type_vars.values():
                tv.set(False)

        ttk.Button(action_frame, text="Select All", command=select_all).pack(side='left', padx=4)
        ttk.Button(action_frame, text="Select None", command=select_none).pack(side='left', padx=4)

        # OK / Cancel
        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(padx=10, pady=(0, 10))

        def apply():
            group['buildings'] = [f for f in self._building_files
                                  if file_vars.get(f, tk.BooleanVar(value=False)).get()
                                  and f not in taken]
            self._update_group_buildings_label()
            self._schedule_preview()
            dlg.destroy()

        ttk.Button(btn_frame, text="OK", command=apply).pack(side='left', padx=4)
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(side='left', padx=4)

    def _get_group_for_building(self, filename):
        """Find which group a building file belongs to. Returns the group dict.

        Groups are checked in order (priority). First non-Default group that
        contains this file wins. Default is always the fallback.
        The x1 canonical filename is used for matching.
        """
        # Normalize to x1 filename for matching
        canonical = filename.replace('_x2.', '_x1.')
        for g in self._groups:
            if g['name'] != 'Default' and canonical in g['buildings']:
                return g
        return self._groups[0]  # Default

    def _get_settings_for_building(self, filename):
        """Get the settings dict for a specific building file."""
        group = self._get_group_for_building(filename)
        return group['settings']

    # --- Presets ---

    def _refresh_preset_list(self):
        presets = _load_presets()
        names = sorted(presets.keys())
        if 'Default' not in names:
            names.insert(0, 'Default')
        self._preset_combo['values'] = names

    def _save_preset(self):
        """Save current configuration as a named preset."""
        name = simpledialog.askstring("Save Preset", "Preset name:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()

        # Save current group settings first
        self._save_current_group_settings()

        preset = {
            'groups': [
                {
                    'name': g['name'],
                    'buildings': list(g['buildings']),
                    'settings': dict(g['settings']),
                }
                for g in self._groups
            ],
            'included_files': sorted(self._included_files),
            'build_x1': self.build_x1_var.get(),
            'build_x2': self.build_x2_var.get(),
            'workers': self.workers_var.get(),
        }

        presets = _load_presets()
        presets[name] = preset
        _save_presets(presets)
        self._refresh_preset_list()
        self._preset_var.set(name)

    def _load_preset(self):
        """Load a preset by name."""
        name = self._preset_var.get()
        if not name:
            return

        # "Default" preset = reset everything to defaults
        if name == 'Default':
            presets = _load_presets()
            preset = presets.get('Default')
            if not preset:
                # Built-in default: single Default group with default settings
                preset = {
                    'groups': [{'name': 'Default', 'buildings': [],
                                'settings': dict(DEFAULT_GROUP_SETTINGS)}],
                    'included_files': [],  # empty = all
                    'build_x1': True,
                    'build_x2': True,
                    'workers': cpu_count(),
                }
        else:
            presets = _load_presets()
            preset = presets.get(name)
            if not preset:
                return

        # Restore groups
        groups_data = preset.get('groups', [])
        if groups_data:
            self._groups = []
            for gd in groups_data:
                self._groups.append({
                    'name': gd.get('name', 'Default'),
                    'buildings': list(gd.get('buildings', [])),
                    'settings': {**DEFAULT_GROUP_SETTINGS, **gd.get('settings', {})},
                })
            # Ensure Default group exists
            if not any(g['name'] == 'Default' for g in self._groups):
                self._groups.insert(0, {
                    'name': 'Default', 'buildings': [],
                    'settings': dict(DEFAULT_GROUP_SETTINGS),
                })
        else:
            self._groups = [{'name': 'Default', 'buildings': [],
                             'settings': dict(DEFAULT_GROUP_SETTINGS)}]

        self._current_group_idx = 0
        self._refresh_group_list()
        self._apply_group_settings(self._groups[0]['settings'])

        # Restore included files (backward compat: 'exclude' key or 'included_files')
        if 'included_files' in preset:
            inc = preset['included_files']
            if inc:
                self._included_files = set(inc) & set(self._building_files)
            else:
                self._included_files = set(self._building_files)
        elif 'exclude' in preset:
            # Legacy: convert exclude types to included files
            exc_types = set(preset['exclude'])
            self._included_files = {
                f for f in self._building_files
                if _filename_to_type(f) not in exc_types
            }
        else:
            self._included_files = set(self._building_files)

        self._update_group_buildings_label()
        self.build_x1_var.set(preset.get('build_x1', True))
        self.build_x2_var.set(preset.get('build_x2', True))
        self.workers_var.set(preset.get('workers', cpu_count()))

        self._schedule_preview()

    def _delete_preset(self):
        """Delete the selected preset."""
        name = self._preset_var.get()
        if not name:
            return
        if not messagebox.askyesno("Delete Preset", f"Delete preset '{name}'?"):
            return
        presets = _load_presets()
        presets.pop(name, None)
        _save_presets(presets)
        self._preset_var.set('')
        self._refresh_preset_list()

    # ------------------------------------------------------------------
    # Preview rendering
    # ------------------------------------------------------------------

    def _schedule_preview(self):
        """Debounce full re-render — cancel pending, schedule new."""
        # Auto-save current settings to the active group
        self._save_current_group_settings()
        if self._preview_pending is not None:
            self.root.after_cancel(self._preview_pending)
        self._preview_pending = self.root.after(150, self._render_preview)

    def _schedule_viewport(self):
        """Fast viewport update (pan/zoom only, no re-render)."""
        if hasattr(self, '_cached_source') and self._cached_source is not None:
            self._update_viewport()
        else:
            self._schedule_preview()

    def _load_sld_bytes(self, filename):
        """Load SLD file bytes, with caching.

        Checks combined mod graphics directories first, then falls back
        to vanilla game graphics.
        """
        if filename in self._sld_cache:
            return self._sld_cache[filename]
        if not self.graphics_dir:
            return None
        # Check combined mod graphics dirs first
        path = None
        gfx_subpath = os.path.join("resources", "_common", "drs", "graphics")
        for mr in self._get_combine_mod_roots():
            candidate = os.path.join(mr, gfx_subpath, filename)
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            path = os.path.join(self.graphics_dir, filename)
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            data = f.read()
        self._sld_cache[filename] = data
        return data

    def _render_preview(self):
        """Render before/after preview of the selected building."""
        self._preview_pending = None

        if not self._building_files or not self.graphics_dir:
            self.preview_label.configure(text="No game graphics detected.\n"
                                              "Set AOE2_GRAPHICS_DIR to enable preview.")
            return

        # Get selected building
        sel = self.building_var.get()
        if sel not in self._building_names:
            return
        idx = self._building_names.index(sel)
        filename = self._building_files[idx]

        # Use the selected preview scale (x1 or x2)
        preview_scale = self._preview_scale_var.get()
        if '_x1.' in filename and preview_scale == 'x2':
            x2_filename = filename.replace('_x1.', '_x2.')
            x2_data = self._load_sld_bytes(x2_filename)
            if x2_data is not None:
                filename = x2_filename
                data = x2_data
            else:
                data = self._load_sld_bytes(filename)
        elif '_x2.' in filename and preview_scale == 'x1':
            x1_filename = filename.replace('_x2.', '_x1.')
            x1_data = self._load_sld_bytes(x1_filename)
            if x1_data is not None:
                filename = x1_filename
                data = x1_data
            else:
                data = self._load_sld_bytes(filename)
        else:
            data = self._load_sld_bytes(filename)

        if data is None:
            self.preview_label.configure(text=f"Could not load {filename}")
            return



        try:
            from sld import parse_sld, get_layer, get_block_positions, LAYER_MAIN, LAYER_PLAYERCOLOR
            from tools.sld_to_png import render_frame
            from build_mod import (
                process_frame, get_building_tiles, TILE_HALF_HEIGHT, TILE_WIDTH
            )

            team_rgb = TEAM_COLORS.get(self.team_color_var.get(), (0, 0, 255))
            bg_spec = BACKGROUNDS.get(self.bg_var.get(), ('solid', (0x22, 0x22, 0x22)))
            bg_tile = self._get_bg_tile(bg_spec)

            # Use the building's group settings for preview
            # (look up original x1 filename for group matching)
            orig_filename = self._building_files[self._building_names.index(sel)]
            bld_group = self._get_group_for_building(orig_filename)
            gs = bld_group['settings']

            dither_intensity = _pct_to_bayer(gs.get('transparency', 50))
            dither_bottom = gs.get('dither_bottom', False)
            edge_inset = gs.get('edge_inset', 3)
            gradient_height = gs.get('gradient_height', 0)
            outline_value = gs.get('outline_value', 200)
            outline_thickness = gs.get('outline_thickness', 4)
            no_outline = gs.get('no_outline', False)
            contour_width = gs.get('contour_width', 0)
            contour_color_str = gs.get('contour_color', 'Team color')
            contour_color = 'team' if contour_color_str == 'Team color' else 'black'

            # --- Render ORIGINAL (combined mod's unmodified sprite) ---
            sld_orig = parse_sld(data)
            orig_main = render_frame(sld_orig.frames[0], LAYER_MAIN)
            orig_pc = render_frame(sld_orig.frames[0], LAYER_PLAYERCOLOR)

            # --- Render MODIFIED ---
            sld_mod = parse_sld(data)
            scale = 'x2' if '_x2.' in filename else 'x1'
            tile_hh = TILE_HALF_HEIGHT[scale]
            tile_w = TILE_WIDTH[scale]
            main_layer = get_layer(sld_mod.frames[0], LAYER_MAIN)
            layer_w = (main_layer.offset_x2 - main_layer.offset_x1) if main_layer else 0
            tiles = get_building_tiles(filename, layer_w, tile_w)

            # Town Center front/back: fully transparent (matches process_file logic)
            name_lower = filename.lower()
            preview_outline = not no_outline
            preview_dither = dither_intensity
            preview_dither_bottom = dither_bottom
            if 'town_center' in name_lower and ('_front' in name_lower or '_back' in name_lower):
                preview_dither = 16
                preview_dither_bottom = True
                preview_outline = False

            process_frame(sld_mod.frames[0], tile_hh, tiles,
                          outline_value=outline_value,
                          edge_inset=edge_inset,
                          gradient_height=gradient_height,
                          outline_thickness=outline_thickness,
                          outline_enabled=preview_outline,
                          dither_intensity=preview_dither,
                          dither_bottom=preview_dither_bottom,
                          contour_width=contour_width,
                          contour_color=contour_color)

            mod_main = render_frame(sld_mod.frames[0], LAYER_MAIN)
            mod_pc = render_frame(sld_mod.frames[0], LAYER_PLAYERCOLOR)

            # --- Composite both ---
            orig_img = self._composite(orig_main, orig_pc, team_rgb, bg_tile)
            mod_img = self._composite(mod_main, mod_pc, team_rgb, bg_tile)

            # --- Build source image at 1x (cached for fast pan/zoom) ---
            show_before = self._show_before_var.get()
            label_h = 20
            h = max(orig_img.shape[0], mod_img.shape[0])
            w1, w2 = orig_img.shape[1], mod_img.shape[1]
            gap = 16

            def _blit_rgb(canvas, img, y, x):
                """Copy RGB img onto RGB canvas with clipping."""
                ih, iw = img.shape[:2]
                ch, cw = canvas.shape[:2]
                sh = min(ih, ch - y) if y >= 0 else min(ih + y, ch)
                sw = min(iw, cw - x) if x >= 0 else min(iw + x, cw)
                sy = 0 if y >= 0 else -y
                sx = 0 if x >= 0 else -x
                dy, dx = max(0, y), max(0, x)
                if sh > 0 and sw > 0:
                    canvas[dy:dy+sh, dx:dx+sw] = img[sy:sy+sh, sx:sx+sw, :3]

            if show_before:
                combined_w = w1 + gap + w2
            else:
                combined_w = w2
            total_h = h + label_h

            # Neutral background for outer canvas
            source = np.full((total_h, combined_w, 3), 200, dtype=np.uint8)

            if show_before:
                oy = label_h + (h - orig_img.shape[0]) // 2
                _blit_rgb(source, orig_img, oy, 0)
                my = label_h + (h - mod_img.shape[0]) // 2
                _blit_rgb(source, mod_img, my, w1 + gap)
            else:
                my = label_h + (h - mod_img.shape[0]) // 2
                _blit_rgb(source, mod_img, my, 0)

            # Draw labels onto source
            source_pil = Image.fromarray(source, 'RGB')
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(source_pil)
            avg = 200  # neutral bg
            text_color = (255, 255, 255) if avg < 128 else (0, 0, 0)
            try:
                font = ImageFont.truetype("segoeui.ttf", 13)
            except Exception:
                font = ImageFont.load_default()
            if show_before:
                draw.text((w1 // 2, 2), "Before", fill=text_color, font=font, anchor='mt')
                draw.text((w1 + gap + w2 // 2, 2), "After", fill=text_color, font=font, anchor='mt')

            # Cache and display
            self._cached_source = source_pil
            self._cached_bg_tile = bg_tile
            self._update_viewport()

        except Exception as e:
            self.preview_label.configure(image='', text=f"Preview error:\n{e}")

    def _update_viewport(self):
        """Fast viewport update: crop/scale the cached source image for current zoom/pan."""
        if self._cached_source is None:
            return

        src = self._cached_source
        src_w, src_h = src.size
        PREVIEW_W = max(200, self._preview_w)
        PREVIEW_H = max(150, self._preview_h)

        zoom = self._zoom_var.get()
        fit_scale = min(PREVIEW_W / src_w, PREVIEW_H / src_h)

        if zoom <= 0.6:
            # Fit mode: scale entire source to fit, no pan
            scale = fit_scale
            resized = src.resize((int(src_w * scale), int(src_h * scale)), Image.NEAREST)
            # Center on preview canvas
            bg_tile = getattr(self, '_cached_bg_tile', (0x22, 0x22, 0x22))
            canvas = self._make_bg_canvas(PREVIEW_H, PREVIEW_W, bg_tile)
            canvas_pil = Image.fromarray(canvas, 'RGB')
            ox = (PREVIEW_W - resized.width) // 2
            oy = (PREVIEW_H - resized.height) // 2
            canvas_pil.paste(resized, (ox, oy))
            result = canvas_pil
        else:
            # Zoomed: crop a viewport-sized region from the scaled source
            # The viewport in source coordinates
            vw = PREVIEW_W / zoom
            vh = PREVIEW_H / zoom
            # Center of view in source coords (with pan offset)
            cx = src_w / 2 + self._pan_x
            cy = src_h / 2 + self._pan_y
            # Crop box in source coords
            x1 = cx - vw / 2
            y1 = cy - vh / 2
            x2 = x1 + vw
            y2 = y1 + vh

            # Pad with background if crop extends outside source
            bg_tile = getattr(self, '_cached_bg_tile', (0x22, 0x22, 0x22))
            canvas = self._make_bg_canvas(PREVIEW_H, PREVIEW_W, bg_tile)
            canvas_pil = Image.fromarray(canvas, 'RGB')

            # Clamp crop to source bounds and compute destination offset
            sx1 = max(0, int(x1))
            sy1 = max(0, int(y1))
            sx2 = min(src_w, int(x2))
            sy2 = min(src_h, int(y2))

            if sx2 > sx1 and sy2 > sy1:
                cropped = src.crop((sx1, sy1, sx2, sy2))
                scaled = cropped.resize(
                    (int((sx2 - sx1) * zoom), int((sy2 - sy1) * zoom)),
                    Image.NEAREST)
                # Where to paste on the canvas
                dx = int((sx1 - x1) * zoom)
                dy = int((sy1 - y1) * zoom)
                canvas_pil.paste(scaled, (dx, dy))

            result = canvas_pil

        self._preview_photo = ImageTk.PhotoImage(result)
        self.preview_label.configure(image=self._preview_photo, text='')

    def _get_bg_tile(self, bg_spec):
        """Get background tile: either a terrain RGBA numpy array or a solid RGB tuple.

        Returns (tile_array,) for terrain or (r, g, b) tuple for solid color.
        """
        kind, value = bg_spec
        if kind == 'terrain':
            # Cache terrain tiles
            cache_key = f'terrain_{value}'
            if cache_key not in self._sld_cache:
                try:
                    from tools.make_poster import _load_terrain_texture
                    tex = _load_terrain_texture(value, 'x1')
                    if tex is not None:
                        self._sld_cache[cache_key] = tex
                    else:
                        # Fallback colors
                        fallback = {'grass': (87, 122, 52), 'water': (40, 70, 120)}
                        return fallback.get(value, (0x22, 0x22, 0x22))
                except Exception:
                    fallback = {'grass': (87, 122, 52), 'water': (40, 70, 120)}
                    return fallback.get(value, (0x22, 0x22, 0x22))
            return self._sld_cache[cache_key]
        else:
            return value  # solid RGB tuple

    def _make_bg_canvas(self, h, w, bg_tile):
        """Create an RGB canvas filled with background (terrain tile or solid color)."""
        if isinstance(bg_tile, np.ndarray):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            th, tw = bg_tile.shape[:2]
            for ty in range(0, h, th):
                for tx in range(0, w, tw):
                    sh = min(th, h - ty)
                    sw = min(tw, w - tx)
                    canvas[ty:ty+sh, tx:tx+sw] = bg_tile[:sh, :sw, :3]
            return canvas
        else:
            return np.full((h, w, 3), bg_tile, dtype=np.uint8)

    def _composite(self, main_canvas, pc_canvas, team_rgb, bg_tile):
        """Composite main + playercolor + team color onto background.

        bg_tile: either a terrain RGBA numpy array or a solid (r, g, b) tuple.
        Returns an RGBA numpy array (terrain only behind opaque pixels).
        """
        if main_canvas is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        h, w = main_canvas.shape[:2]

        # Main layer RGBA
        main_rgb = main_canvas[:, :, :3].astype(np.float32)
        main_a = main_canvas[:, :, 3].astype(np.float32) / 255.0

        # Player color intensity (grayscale), only where main is opaque
        if pc_canvas is not None:
            pc_intensity = pc_canvas[:, :, 0].astype(np.float32) / 255.0
            pc_intensity *= (main_a > 0).astype(np.float32)
        else:
            pc_intensity = np.zeros((h, w), dtype=np.float32)

        # Composite: team_color * pc_intensity + main * (1 - pc_intensity)
        team = np.array(team_rgb, dtype=np.float32)
        blended = (main_rgb * (1.0 - pc_intensity[:, :, None])
                   + team[None, None, :] * pc_intensity[:, :, None])

        # Build terrain background for entire canvas
        bg = self._make_bg_canvas(h, w, bg_tile)

        # Alpha composite building onto terrain bg
        bg_f = bg.astype(np.float32)
        rgb = blended * main_a[:, :, None] + bg_f * (1.0 - main_a[:, :, None])

        # Return full RGB image (terrain everywhere, building composited on top)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _log(self, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _set_status(self, msg):
        self.status_label.configure(text=msg)

    def _on_build(self):
        if self.build_thread and self.build_thread.is_alive():
            return

        # Validate
        output_dir = self.output_var.get().strip()
        if not output_dir:
            messagebox.showerror("Error", "Please set an output directory.")
            return

        # Save current group settings before building
        self._save_current_group_settings()

        # Gather global settings + groups for the build
        settings = {
            'workers': self.workers_var.get(),
            'output_dir': output_dir,
            'combine_mod_roots': self._get_combine_mod_roots(),
            'included_files': set(self._included_files),
            'build_x1': self.build_x1_var.get(),
            'build_x2': self.build_x2_var.get(),
            'groups': [
                {
                    'name': g['name'],
                    'buildings': list(g['buildings']),
                    'settings': dict(g['settings']),
                }
                for g in self._groups
            ],
        }

        # Clear log
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')

        self.build_btn.state(['disabled'])
        self.progress['value'] = 0

        self.build_thread = threading.Thread(target=self._run_build, args=(settings,), daemon=True)
        self.build_thread.start()

    def _build_settings_for_file(self, filename, groups):
        """Look up per-group settings for a specific building file."""
        canonical = filename.replace('_x2.', '_x1.')
        for g in groups:
            if g['name'] != 'Default' and canonical in g['buildings']:
                return g['settings']
        return groups[0]['settings']  # Default

    def _run_build(self, settings):
        """Run the build in a background thread."""
        try:
            from build_mod import (
                find_building_files, process_file, TILE_HALF_HEIGHT, TILE_WIDTH
            )
            from paths import get_graphics_dir, get_mod_dir

            graphics_dir = get_graphics_dir()
            output_dir = settings['output_dir']
            combine_mod_roots = settings.get('combine_mod_roots', [])
            # Derive graphics dirs from mod roots for building SLD resolution
            gfx_subpath = os.path.join("resources", "_common", "drs", "graphics")
            combine_gfx_dirs = []
            for mr in combine_mod_roots:
                gd = os.path.join(mr, gfx_subpath)
                if os.path.isdir(gd):
                    combine_gfx_dirs.append(gd)
            os.makedirs(output_dir, exist_ok=True)

            all_files = find_building_files(exclude=[])
            included = settings.get('included_files', set())

            # Filter to included files (match x1 canonical name)
            if included:
                files = [f for f in all_files
                         if f.replace('_x2.', '_x1.') in included]
            else:
                files = all_files

            # Filter by scale
            build_x1 = settings.get('build_x1', True)
            build_x2 = settings.get('build_x2', True)
            if not build_x1:
                files = [f for f in files if '_x1.' not in f]
            if not build_x2:
                files = [f for f in files if '_x2.' not in f]

            groups = settings['groups']

            self.root.after(0, self._log, f"Found {len(files)} building files to process")
            self.root.after(0, self._log, f"Groups: {', '.join(g['name'] for g in groups)}")
            self.root.after(0, self._log, f"Input:  {graphics_dir}")
            for mr in combine_mod_roots:
                self.root.after(0, self._log, f"Combine: {os.path.basename(mr)}")
            self.root.after(0, self._log, f"Output: {output_dir}")
            self.root.after(0, self._log, "")

            tile_hh = dict(TILE_HALF_HEIGHT)
            tile_w = dict(TILE_WIDTH)

            success = 0
            errors = 0
            total = len(files)

            self.root.after(0, lambda: self.progress.configure(maximum=total))
            self.root.after(0, self._set_status, f"Processing 0/{total}...")

            # Use multiprocessing for speed
            from build_mod import _process_file_worker
            from multiprocessing import Pool

            def _resolve_input(fn):
                for gd in combine_gfx_dirs:
                    combined = os.path.join(gd, fn)
                    if os.path.exists(combined):
                        return combined
                return os.path.join(graphics_dir, fn)

            work = []
            for filename in files:
                input_path = _resolve_input(filename)
                output_path = os.path.join(output_dir, filename)
                if not os.path.exists(input_path):
                    errors += 1
                    continue
                # Per-file group settings
                gs = self._build_settings_for_file(filename, groups)
                di = _pct_to_bayer(gs.get('transparency', 50))
                cc_str = gs.get('contour_color', 'Team color')
                cc = 'team' if cc_str == 'Team color' else 'black'
                work.append((input_path, output_path, tile_hh, tile_w,
                             gs.get('outline_value', 200),
                             gs.get('edge_inset', 3),
                             gs.get('gradient_height', 0),
                             gs.get('outline_thickness', 4),
                             gs.get('no_outline', False),
                             di,
                             gs.get('dither_bottom', False),
                             gs.get('contour_width', 0),
                             cc))

            num_workers = settings['workers']
            done = 0

            if len(work) == 1:
                result = _process_file_worker(work[0])
                filename, orig_size, new_size, error = result
                if error:
                    errors += 1
                    self.root.after(0, self._log, f"ERROR {filename}: {error}")
                else:
                    success += 1
                    ratio = new_size / orig_size if orig_size > 0 else 0
                    self.root.after(0, self._log, f"OK {filename} ({ratio:.0%})")
                done = 1
                self.root.after(0, lambda: self.progress.configure(value=1))
            elif work:
                with Pool(num_workers) as pool:
                    for filename, orig_size, new_size, error in pool.imap_unordered(
                            _process_file_worker, work, chunksize=8):
                        done += 1
                        if error:
                            errors += 1
                            self.root.after(0, self._log, f"ERROR {filename}: {error}")
                        else:
                            success += 1

                        # Update progress (throttle UI updates)
                        if done % 20 == 0 or done == len(work):
                            d = done
                            self.root.after(0, lambda d=d: self.progress.configure(value=d))
                            self.root.after(0, self._set_status, f"Processing {d}/{total}...")

            # Copy resources from combined mods
            if combine_mod_roots:
                import shutil
                mod_dir = get_mod_dir()
                # Files we already wrote to the graphics output — don't overwrite
                building_files = set(files)
                copied = 0
                for mr in combine_mod_roots:
                    src_res = os.path.join(mr, "resources")
                    dst_res = os.path.join(mod_dir, "resources")
                    if not os.path.isdir(src_res):
                        continue
                    for dirpath, dirnames, filenames in os.walk(src_res):
                        rel = os.path.relpath(dirpath, src_res)
                        dst_dir = os.path.join(dst_res, rel)
                        os.makedirs(dst_dir, exist_ok=True)
                        for fn in filenames:
                            dst_file = os.path.join(dst_dir, fn)
                            # Skip building files we already processed
                            if rel == os.path.join("_common", "drs", "graphics") and fn in building_files:
                                continue
                            # Don't overwrite files from earlier mods
                            if os.path.exists(dst_file):
                                continue
                            shutil.copy2(os.path.join(dirpath, fn), dst_file)
                            copied += 1
                if copied > 0:
                    self.root.after(0, self._log,
                                    f"Copied {copied} file(s) from combined mods")

            # Write info.json
            if success > 0:
                mod_dir = get_mod_dir()
                info_path = os.path.join(mod_dir, "info.json")
                info = {
                    "Title": "Transparent Buildings",
                    "Description": (
                        "See through buildings! The upper part of every building "
                        "becomes semi-transparent using a dither pattern."
                    ),
                    "Author": "Yustee",
                    "CacheStatus": 0,
                }
                with open(info_path, 'w') as f:
                    json.dump(info, f)

            self.root.after(0, self._log, "")
            self.root.after(0, self._log, f"Done! {success} succeeded, {errors} failed")
            self.root.after(0, self._set_status, f"Done! {success} files processed.")
            self.root.after(0, lambda: self.progress.configure(value=total))

        except Exception as e:
            self.root.after(0, self._log, f"Build failed: {e}")
            self.root.after(0, self._set_status, "Build failed.")

        finally:
            self.root.after(0, lambda: self.build_btn.state(['!disabled']))


def main():
    try:
        root = tk.Tk()

        # Apply a theme
        style = ttk.Style()
        available = style.theme_names()
        for theme in ('vista', 'winnative', 'clam'):
            if theme in available:
                style.theme_use(theme)
                break

        app = TransparentBuildingsGUI(root)
        root.mainloop()
    except Exception as e:
        try:
            messagebox.showerror("Startup Error", str(e))
        except Exception:
            pass
        raise


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
