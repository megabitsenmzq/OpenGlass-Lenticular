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


def get_saved_calibration(device, context=None):
    _load_store()
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


def _set_shared_pref(key, value):
    """Persist a device-independent setting into the JSON config store."""
    _load_store()
    if _store.get(key) != value:
        _store[key] = value
        _persist_store()


def device_changed(self, context):
    _load_store()
    update_device_calibration(self, self.device, context)
    _set_shared_pref("device", self.device)


def grid_changed(self, context):
    if self.go_grid in VALID_GRIDS:
        _set_shared_pref("go_grid", self.go_grid)


def view_interpolation_changed(self, context):
    _set_shared_pref("view_interpolation", bool(self.view_interpolation))


def tag_redraw_view3d(self=None, context=None):
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def show_screen_plane_changed(self, context):
    tag_redraw_view3d(self, context)


def preferred_device():
    _load_store()
    device = _store.get("device", "GO")
    return device if device in ("GO", "C1") else "GO"


def preferred_grid():
    _load_store()
    grid = _store.get("go_grid")
    if grid in VALID_GRIDS:
        return grid
    return DEFAULT_GRID


def preferred_view_interpolation():
    _load_store()
    val = _store.get("view_interpolation")
    if isinstance(val, bool):
        return val
    return True


def apply_saved_device(scene=None, preserve_existing=False):
    _load_store()
    if isinstance(scene, bpy.types.Scene):
        scenes = (scene,)
    else:
        scenes = tuple(getattr(bpy.data, "scenes", ()) or ())
        if not scenes:
            active_scene = getattr(bpy.context, "scene", None)
            if active_scene is not None:
                scenes = (active_scene,)
    target_grid = preferred_grid()
    target_interp = preferred_view_interpolation()
    target_device = preferred_device()
    for target_scene in scenes:
        settings = getattr(target_scene, "lenticular", None)
        if settings is None:
            continue
        if settings.go_grid != target_grid:
            settings.go_grid = target_grid
        if settings.view_interpolation != target_interp:
            settings.view_interpolation = target_interp
        if preserve_existing and is_scene_calibrated(settings):
            continue
        if preserve_existing:
            update_device_calibration(settings, settings.device)
            continue
        if settings.device != target_device:
            settings.device = target_device
        elif not is_scene_calibrated(settings):
            update_device_calibration(settings, target_device)


@persistent
def load_saved_device(filepath=""):
    apply_saved_device(preserve_existing=bool(filepath))


def restore_saved_device():
    apply_saved_device(preserve_existing=bool(bpy.data.filepath))
    return None


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
        description="Seconds to wait after moving the view or editing the scene before refreshing (0 to disable)",
        default=0.3,
        min=0.0,
        max=60.0,
        soft_max=30.0,
        precision=1,
    )
    rendered_debounce: bpy.props.FloatProperty(
        name="Live Render Delay",
        description="Seconds to wait after moving the view or editing the scene before refreshing (0 to disable)",
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
        description="Pixel offset added to each view's width",
        default=0,
        min=-719,
        max=512,
    )
    compensation_y: bpy.props.IntProperty(
        name="Y Compensation (px)",
        description="Pixel offset added to each view's height",
        default=0,
        min=-681,
        max=512,
    )
    view_interpolation: bpy.props.BoolProperty(
        name="Smooth View Transitions",
        description="Blend adjacent views for smoother motion; disable for sharper nearest-view details",
        default=True,
        update=view_interpolation_changed,
    )
    show_screen_plane: bpy.props.BoolProperty(
        name="Show Screen Plane",
        description="Show a translucent blue screen plane in the 3D viewport; everything behind it appears inside the screen",
        default=True,
        update=show_screen_plane_changed,
    )
    status: bpy.props.StringProperty(default="Not started", update=tag_redraw_view3d)
    cycles_progress: bpy.props.IntProperty(default=0, min=0, max=2147483647, update=tag_redraw_view3d)


classes = (LenticularSettings,)


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
