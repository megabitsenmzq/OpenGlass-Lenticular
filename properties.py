import json
import math
import os

import bpy
from bpy.app.handlers import persistent


DEVICE_ITEMS = [
    ("GO", "Looking Glass Go", "10–100 views"),
    ("C1", "OpenStage AI C1", "10–100 views"),
]
DEFAULT_GRID = "5x9"
GRID_PRESETS = (
    (5, 2), (4, 3), (4, 4), (5, 4),
    (6, 4), (6, 5), (6, 6), (8, 5),
    (5, 9), (8, 6), (6, 10), (9, 8),
    (11, 8), (7, 13), (12, 8), (10, 10),
)
GRID_ITEMS = [
    (f"{cols}x{rows}", f"{cols}×{rows} ({cols * rows} views)", "")
    for cols, rows in GRID_PRESETS
]
GRID_LAYOUTS = {f"{cols}x{rows}": (cols, rows) for cols, rows in GRID_PRESETS}
VALID_GRIDS = set(GRID_LAYOUTS)

_store_loaded = False
_store = {"device": "GO", "go_grid": DEFAULT_GRID, "view_interpolation": True, "devices": {}}
_updating_calibration = False


def is_calibration_valid(x0, interval, slope):
    return (
        math.isfinite(x0)
        and math.isfinite(interval)
        and math.isfinite(slope)
        and interval > 0.001
        and abs(slope) > 1e-6
    )


def is_scene_calibrated(settings):
    if settings is None:
        return False
    return is_calibration_valid(settings.x0, settings.interval, settings.slope)


def _config_file_path(create=False):
    override_dir = os.environ.get("LENTICULAR_CONFIG_DIR")
    if override_dir:
        if create:
            os.makedirs(override_dir, exist_ok=True)
        return os.path.join(override_dir, "calibration.json")
    config_dir = bpy.utils.user_resource("CONFIG", path="lenticular", create=create)
    if not config_dir:
        return None
    return os.path.join(config_dir, "calibration.json")


def _load_store(force=False):
    global _store_loaded
    if _store_loaded and not force:
        return
    _store_loaded = True
    path = _config_file_path(create=False)
    if not path or not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    device = data.get("device")
    if device in ("GO", "C1"):
        _store["device"] = device
    go_grid = data.get("go_grid")
    if go_grid in VALID_GRIDS:
        _store["go_grid"] = go_grid
    view_interpolation = data.get("view_interpolation")
    if isinstance(view_interpolation, bool):
        _store["view_interpolation"] = view_interpolation
    devices = data.get("devices")
    if isinstance(devices, dict):
        for key in ("GO", "C1"):
            entry = devices.get(key)
            if not isinstance(entry, dict):
                continue
            try:
                x0 = float(entry["x0"])
                interval = float(entry["interval"])
                slope = float(entry["slope"])
            except (KeyError, TypeError, ValueError):
                continue
            if is_calibration_valid(x0, interval, slope):
                source = str(entry.get("source") or "Saved")
                _store["devices"][key] = {
                    "x0": x0,
                    "interval": interval,
                    "slope": slope,
                    "source": source,
                }


def _persist_store():
    path = _config_file_path(create=True)
    if not path:
        return
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(_store, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except OSError:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def get_preferences(context=None):
    prefs = getattr(context, "preferences", None) if context is not None else None
    if prefs is None:
        prefs = getattr(bpy.context, "preferences", None)
    if prefs is None:
        return None
    addon = prefs.addons.get(__package__)
    return addon.preferences if addon is not None else None


def _sync_store_to_preferences(context=None):
    _load_store()
    addon_prefs = get_preferences(context)
    if addon_prefs is None:
        return
    if _store.get("device") in ("GO", "C1") and addon_prefs.device != _store["device"]:
        addon_prefs.device = _store["device"]
    if _store.get("go_grid") in VALID_GRIDS and addon_prefs.go_grid != _store["go_grid"]:
        addon_prefs.go_grid = _store["go_grid"]
    if isinstance(_store.get("view_interpolation"), bool) and addon_prefs.view_interpolation != _store["view_interpolation"]:
        addon_prefs.view_interpolation = _store["view_interpolation"]
    for key in ("GO", "C1"):
        entry = _store["devices"].get(key)
        if not entry:
            continue
        if key == "GO" and not addon_prefs.go_calibrated:
            addon_prefs.go_x0 = entry["x0"]
            addon_prefs.go_interval = entry["interval"]
            addon_prefs.go_slope = entry["slope"]
            addon_prefs.go_calibration_source = entry["source"]
            addon_prefs.go_calibrated = True
        elif key == "C1" and not addon_prefs.c1_calibrated:
            addon_prefs.c1_x0 = entry["x0"]
            addon_prefs.c1_interval = entry["interval"]
            addon_prefs.c1_slope = entry["slope"]
            addon_prefs.c1_calibration_source = entry["source"]
            addon_prefs.c1_calibrated = True


def get_saved_calibration(device, context=None):
    _load_store()
    addon_prefs = get_preferences(context)
    if addon_prefs is not None:
        if device == "GO" and addon_prefs.go_calibrated and is_calibration_valid(addon_prefs.go_x0, addon_prefs.go_interval, addon_prefs.go_slope):
            return True, addon_prefs.go_x0, addon_prefs.go_interval, addon_prefs.go_slope, addon_prefs.go_calibration_source
        if device == "C1" and addon_prefs.c1_calibrated and is_calibration_valid(addon_prefs.c1_x0, addon_prefs.c1_interval, addon_prefs.c1_slope):
            return True, addon_prefs.c1_x0, addon_prefs.c1_interval, addon_prefs.c1_slope, addon_prefs.c1_calibration_source
    entry = _store["devices"].get(device)
    if entry and is_calibration_valid(entry["x0"], entry["interval"], entry["slope"]):
        return True, entry["x0"], entry["interval"], entry["slope"], entry["source"]
    return False, 0.0, 0.0, 0.0, ""


def save_device_calibration(device, x0, interval, slope, source="Manual", context=None, switch_scene_device=False):
    _load_store()
    x0 = float(x0)
    interval = float(interval)
    slope = float(slope)
    valid = is_calibration_valid(x0, interval, slope)
    if valid:
        _store["devices"][device] = {
            "x0": x0,
            "interval": interval,
            "slope": slope,
            "source": source,
        }
    else:
        _store["devices"].pop(device, None)
    if switch_scene_device and device in ("GO", "C1"):
        _store["device"] = device
    _persist_store()

    addon_prefs = get_preferences(context)
    if addon_prefs is not None:
        if device == "GO":
            addon_prefs.go_x0 = x0
            addon_prefs.go_interval = interval
            addon_prefs.go_slope = slope
            addon_prefs.go_calibration_source = source
            addon_prefs.go_calibrated = valid
        else:
            addon_prefs.c1_x0 = x0
            addon_prefs.c1_interval = interval
            addon_prefs.c1_slope = slope
            addon_prefs.c1_calibration_source = source
            addon_prefs.c1_calibrated = valid
        if switch_scene_device:
            addon_prefs.device = device

    scene = getattr(context, "scene", None) if context is not None else getattr(bpy.context, "scene", None)
    settings = getattr(scene, "lenticular", None)
    if settings is not None:
        if switch_scene_device and settings.device != device:
            settings.device = device
        if settings.device == device:
            _set_scene_calibration_values(settings, x0, interval, slope)


def _set_scene_calibration_values(settings, x0, interval, slope):
    global _updating_calibration
    _updating_calibration = True
    try:
        settings.x0 = float(x0)
        settings.interval = float(interval)
        settings.slope = float(slope)
    finally:
        _updating_calibration = False


def calibration_value_changed(self, context):
    if _updating_calibration:
        return
    if not is_scene_calibrated(self):
        return
    save_device_calibration(
        self.device,
        self.x0,
        self.interval,
        self.slope,
        source="Manual",
        context=context,
    )


def try_auto_import_go_calibration(context=None):
    calibrated, _, _, _, _ = get_saved_calibration("GO", context)
    if calibrated:
        return True
    from . import calibration
    return calibration.auto_import_go_calibration(context=context) is not None


def update_device_calibration(settings, device, context=None):
    calibrated, x0, interval, slope, _ = get_saved_calibration(device, context)
    if not calibrated and device == "GO":
        if try_auto_import_go_calibration(context):
            calibrated, x0, interval, slope, _ = get_saved_calibration(device, context)
    if calibrated:
        _set_scene_calibration_values(settings, x0, interval, slope)
    else:
        _set_scene_calibration_values(settings, 0.0, 0.0, 0.0)


def _set_shared_pref(key, value, context=None, sync_prefs=True):
    """Persist a device-independent setting and mirror it into AddonPreferences."""
    _load_store()
    if _store.get(key) != value:
        _store[key] = value
        _persist_store()
    if sync_prefs:
        addon_prefs = get_preferences(context)
        if addon_prefs is not None and getattr(addon_prefs, key) != value:
            setattr(addon_prefs, key, value)


def device_changed(self, context):
    _load_store()
    update_device_calibration(self, self.device, context)
    _set_shared_pref("device", self.device, context=context)


def grid_changed(self, context):
    if self.go_grid in VALID_GRIDS:
        _set_shared_pref("go_grid", self.go_grid, context=context)


def view_interpolation_changed(self, context):
    _set_shared_pref("view_interpolation", bool(self.view_interpolation), context=context)


def show_screen_plane_changed(self, context):
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def preferred_device():
    _load_store()
    addon_prefs = get_preferences()
    if addon_prefs is not None:
        return addon_prefs.device
    return _store.get("device", "GO")


def preferred_grid():
    _load_store()
    grid = _store.get("go_grid")
    if grid in VALID_GRIDS:
        return grid
    addon_prefs = get_preferences()
    if addon_prefs is not None and addon_prefs.go_grid in VALID_GRIDS:
        return addon_prefs.go_grid
    return "5x9"


def preferred_view_interpolation():
    _load_store()
    val = _store.get("view_interpolation")
    if isinstance(val, bool):
        return val
    addon_prefs = get_preferences()
    if addon_prefs is not None:
        return bool(addon_prefs.view_interpolation)
    return True


def apply_saved_device(scene=None, preserve_existing=False):
    _sync_store_to_preferences()
    if not isinstance(scene, bpy.types.Scene):
        scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return
    settings = getattr(scene, "lenticular", None)
    if settings is None:
        return
    if not preserve_existing or not settings.is_property_set("go_grid"):
        target_grid = preferred_grid()
        if settings.go_grid != target_grid:
            settings.go_grid = target_grid
    if not preserve_existing or not settings.is_property_set("view_interpolation"):
        target_interp = preferred_view_interpolation()
        if settings.view_interpolation != target_interp:
            settings.view_interpolation = target_interp
    if preserve_existing:
        if not is_scene_calibrated(settings):
            update_device_calibration(settings, settings.device)
        return
    target = preferred_device()
    if settings.device != target:
        settings.device = target
    elif not is_scene_calibrated(settings):
        update_device_calibration(settings, target)


@persistent
def load_saved_device(filepath=""):
    apply_saved_device(preserve_existing=bool(filepath))


def restore_saved_device():
    apply_saved_device(preserve_existing=bool(bpy.data.filepath))
    return None


def _pref_grid_edited(self, context):
    if self.go_grid in VALID_GRIDS:
        _set_shared_pref("go_grid", self.go_grid, context=context, sync_prefs=False)


def _pref_view_interpolation_edited(self, context):
    _set_shared_pref("view_interpolation", bool(self.view_interpolation), context=context, sync_prefs=False)


def _pref_device_calibration_edited(prefs, device, prefix):
    x0 = getattr(prefs, f"{prefix}_x0")
    interval = getattr(prefs, f"{prefix}_interval")
    slope = getattr(prefs, f"{prefix}_slope")
    valid = is_calibration_valid(x0, interval, slope)
    calibrated_attr = f"{prefix}_calibrated"
    source_attr = f"{prefix}_calibration_source"
    if getattr(prefs, calibrated_attr) != valid:
        setattr(prefs, calibrated_attr, valid)
    if valid and not getattr(prefs, source_attr):
        setattr(prefs, source_attr, "Manual")
    _load_store()
    if valid:
        _store["devices"][device] = {
            "x0": float(x0),
            "interval": float(interval),
            "slope": float(slope),
            "source": getattr(prefs, source_attr) or "Manual",
        }
    else:
        _store["devices"].pop(device, None)
    _persist_store()


def _pref_go_calibration_edited(self, context):
    _pref_device_calibration_edited(self, "GO", "go")


def _pref_c1_calibration_edited(self, context):
    _pref_device_calibration_edited(self, "C1", "c1")


class LenticularPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    device: bpy.props.EnumProperty(
        name="Default Output Device",
        items=DEVICE_ITEMS,
        default="GO",
    )
    go_grid: bpy.props.EnumProperty(
        name="Default View Grid",
        items=GRID_ITEMS,
        default="5x9",
        update=_pref_grid_edited,
    )
    view_interpolation: bpy.props.BoolProperty(
        name="Smooth View Transitions",
        default=True,
        update=_pref_view_interpolation_edited,
    )
    go_calibrated: bpy.props.BoolProperty(name="Looking Glass Go Calibrated", default=False)
    go_x0: bpy.props.FloatProperty(name="Center", default=0.0, precision=6, update=_pref_go_calibration_edited)
    go_interval: bpy.props.FloatProperty(name="Pitch", default=0.0, min=0.0, precision=6, update=_pref_go_calibration_edited)
    go_slope: bpy.props.FloatProperty(name="Slope", default=0.0, precision=6, update=_pref_go_calibration_edited)
    go_calibration_source: bpy.props.StringProperty(name="Calibration Source", default="")

    c1_calibrated: bpy.props.BoolProperty(name="OpenStage AI C1 Calibrated", default=False)
    c1_x0: bpy.props.FloatProperty(name="X0", default=0.0, precision=6, update=_pref_c1_calibration_edited)
    c1_interval: bpy.props.FloatProperty(name="Interval", default=0.0, min=0.0, precision=6, update=_pref_c1_calibration_edited)
    c1_slope: bpy.props.FloatProperty(name="Slope", default=0.0, precision=6, update=_pref_c1_calibration_edited)
    c1_calibration_source: bpy.props.StringProperty(name="Calibration Source", default="")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "device")
        layout.prop(self, "go_grid")
        layout.prop(self, "view_interpolation")

        go_box = layout.box()
        go_header = go_box.row()
        go_status = f"Calibrated ({self.go_calibration_source})" if self.go_calibrated else "Not calibrated"
        go_header.label(text=f"Looking Glass Go: {go_status}")
        go_actions = go_box.row(align=True)
        op = go_actions.operator("lenticular.import_calibration", text="Import visual.json / JSON…", icon="FILE_FOLDER")
        op.target_device = "GO"
        go_actions.operator("lenticular.autodetect_calibration", text="Auto-Detect visual.json", icon="VIEWZOOM")
        go_row = go_box.row(align=True)
        go_row.prop(self, "go_x0")
        go_row.prop(self, "go_interval")
        go_row.prop(self, "go_slope")

        c1_box = layout.box()
        c1_header = c1_box.row()
        c1_status = f"Calibrated ({self.c1_calibration_source})" if self.c1_calibrated else "Not calibrated"
        c1_header.label(text=f"OpenStage AI C1: {c1_status}")
        c1_actions = c1_box.row(align=True)
        op = c1_actions.operator("lenticular.import_calibration", text="Import Calibration JSON…", icon="FILE_FOLDER")
        op.target_device = "C1"
        c1_row = c1_box.row(align=True)
        c1_row.prop(self, "c1_x0")
        c1_row.prop(self, "c1_interval")
        c1_row.prop(self, "c1_slope")


class LenticularSettings(bpy.types.PropertyGroup):
    device: bpy.props.EnumProperty(
        name="Output Device",
        items=DEVICE_ITEMS,
        default="GO",
        update=device_changed,
    )
    go_grid: bpy.props.EnumProperty(
        name="View Grid",
        items=GRID_ITEMS,
        default="5x9",
        update=grid_changed,
    )
    x0: bpy.props.FloatProperty(name="X0", default=0.0, precision=6, update=calibration_value_changed)
    interval: bpy.props.FloatProperty(name="Interval", default=0.0, min=0.0, precision=6, update=calibration_value_changed)
    slope: bpy.props.FloatProperty(name="Slope", default=0.0, precision=6, update=calibration_value_changed)
    material_debounce: bpy.props.FloatProperty(
        name="Live Render Delay",
        description="Refresh the stereo pair after the view or scene has been unchanged for this many seconds. Set to 0 to disable the delay; full-frame rendering is unaffected",
        default=0.3,
        min=0.0,
        max=60.0,
        soft_max=30.0,
        precision=1,
    )
    rendered_debounce: bpy.props.FloatProperty(
        name="Live Render Delay",
        description="Refresh the stereo pair after the view or scene has been unchanged for this many seconds. Set to 0 to disable the delay; full-frame rendering is unaffected",
        default=10.0,
        min=0.0,
        max=60.0,
        soft_max=30.0,
        precision=1,
    )
    resolution_mode: bpy.props.EnumProperty(
        name="Render Resolution",
        items=[
            ("FAST", "Reduced Resolution", "OpenStage AI C1 / Looking Glass Go: 372×682 per view"),
            ("FULL", "Full Resolution", "OpenStage AI C1 / Looking Glass Go: 720×2560 per view"),
        ],
        default="FAST",
    )
    compensation_x: bpy.props.IntProperty(
        name="X Compensation (px)",
        description="Pixels added to each tile's width; negative values reduce it. The final width is at least 1 pixel",
        default=0,
        min=-719,
        max=512,
    )
    compensation_y: bpy.props.IntProperty(
        name="Y Compensation (px)",
        description="Pixels added to each tile's height; negative values reduce it. The final height is at least 1 pixel",
        default=0,
        min=-681,
        max=512,
    )
    view_interpolation: bpy.props.BoolProperty(
        name="Smooth View Transitions",
        description="Blend adjacent views for smoother motion when enabled. When disabled, show the nearest view directly for sharper out-of-focus details",
        default=True,
        update=view_interpolation_changed,
    )
    show_screen_plane: bpy.props.BoolProperty(
        name="Show Screen Plane",
        description="Draw a translucent light-blue glass plane at the zero-parallax screen plane in the editing 3D viewport while Live Output is running so everything behind the screen plane (distance > screen plane) is tinted light blue",
        default=True,
        update=show_screen_plane_changed,
    )
    status: bpy.props.StringProperty(default="Not started")
    cycles_progress: bpy.props.IntProperty(default=0, min=0, max=2147483647)


classes = (LenticularSettings, LenticularPreferences)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.lenticular = bpy.props.PointerProperty(type=LenticularSettings)
    bpy.app.timers.register(restore_saved_device, first_interval=0.0)
    if load_saved_device not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(load_saved_device)


def unregister():
    if bpy.app.timers.is_registered(restore_saved_device):
        bpy.app.timers.unregister(restore_saved_device)
    if load_saved_device in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(load_saved_device)
    del bpy.types.Scene.lenticular
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
