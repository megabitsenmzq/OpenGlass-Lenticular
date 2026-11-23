import glob
import json
import math
import os
import sys

import bpy
from bpy_extras.io_utils import ImportHelper

from . import properties


def _extract_number(data, key):
    if key not in data:
        return None
    raw = data[key]
    if isinstance(raw, dict):
        raw = raw.get("value")
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_calibration_file(filepath, fallback_device="GO"):
    """Parse Looking Glass visual.json or flat {x0, interval, slope} calibration JSON."""
    with open(filepath, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Calibration JSON must be an object")

    filename = os.path.basename(filepath)

    # Format 1: Looking Glass Go visual.json (raw optical parameters with DPI / screenW / screenH)
    pitch_raw = _extract_number(data, "pitch")
    slope_raw = _extract_number(data, "slope")
    center_raw = _extract_number(data, "center")
    has_lkg_fields = (
        pitch_raw is not None
        and slope_raw is not None
        and center_raw is not None
        and (
            isinstance(data.get("pitch"), dict)
            or "DPI" in data
            or "screenW" in data
            or "screenH" in data
            or "configVersion" in data
        )
    )
    if has_lkg_fields:
        if pitch_raw <= 0.0:
            raise ValueError(f"{filename}: pitch must be positive")
        if abs(slope_raw) <= 1e-6:
            raise ValueError(f"{filename}: slope must be non-zero")
        dpi = _extract_number(data, "DPI")
        screen_w = _extract_number(data, "screenW")
        screen_h = _extract_number(data, "screenH")
        dpi = dpi if (dpi is not None and dpi > 0.0) else 491.0
        screen_w = screen_w if (screen_w is not None and screen_w > 0.0) else 1440.0
        screen_h = screen_h if (screen_h is not None and screen_h > 0.0) else 2560.0

        x0 = center_raw
        interval = pitch_raw * screen_w / dpi * math.sin(math.atan(abs(slope_raw)))
        slope = screen_h / (screen_w * slope_raw)
        if not properties.is_calibration_valid(x0, interval, slope):
            raise ValueError(f"{filename}: converted calibration values are out of range")
        serial = data.get("serial")
        source = f"{serial} ({filename})" if isinstance(serial, str) and serial.strip() else filename
        return {
            "device": "GO",
            "x0": x0,
            "interval": interval,
            "slope": slope,
            "source": source,
        }

    # Format 2: Flat calibration JSON {"x0": ..., "interval": ..., "slope": ...}
    x0 = _extract_number(data, "x0")
    interval = _extract_number(data, "interval")
    slope = _extract_number(data, "slope")
    if x0 is not None and interval is not None and slope is not None:
        if not properties.is_calibration_valid(x0, interval, slope):
            raise ValueError(f"{filename}: interval must be > 0 and slope must be non-zero")
        device = data.get("device")
        if device not in ("GO", "C1"):
            device = fallback_device if fallback_device in ("GO", "C1") else "GO"
        return {
            "device": device,
            "x0": x0,
            "interval": interval,
            "slope": slope,
            "source": filename,
        }

    raise ValueError(
        f"{filename}: unsupported calibration format (expected Looking Glass visual.json or JSON with x0, interval, slope)"
    )


def find_auto_calibration_files():
    """Search standard Looking Glass Go mount locations and LKG_CALIBRATION for visual.json."""
    candidates = []

    env_path = os.environ.get("LKG_CALIBRATION")
    if env_path:
        if os.path.isfile(env_path):
            candidates.append(env_path)
        elif os.path.isdir(env_path):
            candidates.append(os.path.join(env_path, "visual.json"))
            candidates.append(os.path.join(env_path, "LKG_calibration", "visual.json"))

    if sys.platform == "darwin":
        volumes_root = "/Volumes"
        if os.path.isdir(volumes_root):
            try:
                entries = sorted(os.listdir(volumes_root))
            except OSError:
                entries = []
            for entry in entries:
                vol = os.path.join(volumes_root, entry)
                if os.path.islink(vol) or not os.path.isdir(vol):
                    continue
                candidates.append(os.path.join(vol, "LKG_calibration", "visual.json"))
                candidates.append(os.path.join(vol, "visual.json"))
    elif sys.platform.startswith("linux"):
        candidates.append("/run/lkg/visual.json")
        for pattern in (
            "/media/*/*/LKG_calibration/visual.json",
            "/run/media/*/*/LKG_calibration/visual.json",
            "/mnt/*/LKG_calibration/visual.json",
            "/media/*/*/visual.json",
            "/run/media/*/*/visual.json",
        ):
            candidates.extend(sorted(glob.glob(pattern)))
    elif os.name == "nt":
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            drive = f"{letter}:\\"
            if os.path.isdir(drive):
                candidates.append(os.path.join(drive, "LKG_calibration", "visual.json"))
                candidates.append(os.path.join(drive, "visual.json"))

    seen = set()
    found = []
    for path in candidates:
        norm = os.path.normpath(path)
        if norm not in seen and os.path.isfile(norm):
            seen.add(norm)
            found.append(norm)
    return found


def auto_import_go_calibration(context=None):
    """Try to locate and import visual.json automatically. Returns parsed dict or None."""
    for path in find_auto_calibration_files():
        try:
            result = parse_calibration_file(path, fallback_device="GO")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if result["device"] != "GO":
            continue
        properties.save_device_calibration(
            "GO",
            result["x0"],
            result["interval"],
            result["slope"],
            source=result["source"],
            context=context,
        )
        return result
    return None


class Lenticular_OT_import_calibration(bpy.types.Operator, ImportHelper):
    bl_idname = "lenticular.import_calibration"
    bl_label = "Import Calibration JSON"
    bl_description = "Import Looking Glass Go visual.json or a calibration JSON file (x0, interval, slope)"

    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})
    target_device: bpy.props.EnumProperty(
        name="Target Device",
        items=properties.DEVICE_ITEMS,
        default="GO",
        options={"HIDDEN"},
    )

    def invoke(self, context, event):
        scene = getattr(context, "scene", None)
        settings = getattr(scene, "lenticular", None)
        if settings is not None and not self.properties.is_property_set("target_device"):
            self.target_device = settings.device
        return super().invoke(context, event)

    def execute(self, context):
        try:
            result = parse_calibration_file(self.filepath, fallback_device=self.target_device)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, f"Failed to import calibration: {exc}")
            return {"CANCELLED"}

        device = result["device"]
        properties.save_device_calibration(
            device,
            result["x0"],
            result["interval"],
            result["slope"],
            source=result["source"],
            context=context,
            switch_scene_device=True,
        )
        device_name = "Looking Glass Go" if device == "GO" else "OpenStage AI C1"
        self.report({"INFO"}, f"Imported {device_name} calibration from {result['source']}")
        return {"FINISHED"}


class Lenticular_OT_autodetect_calibration(bpy.types.Operator):
    bl_idname = "lenticular.autodetect_calibration"
    bl_label = "Auto-Detect visual.json"
    bl_description = "Scan mounted drives and LKG_CALIBRATION for Looking Glass Go visual.json"

    def execute(self, context):
        result = auto_import_go_calibration(context=context)
        if result is None:
            self.report(
                {"WARNING"},
                "No visual.json found on mounted volumes (LKG_calibration/visual.json) or LKG_CALIBRATION. Use Import Calibration JSON instead",
            )
            return {"CANCELLED"}
        scene = getattr(context, "scene", None)
        settings = getattr(scene, "lenticular", None)
        if settings is not None and settings.device != "GO":
            settings.device = "GO"
        self.report({"INFO"}, f"Auto-imported Looking Glass Go calibration from {result['source']}")
        return {"FINISHED"}


class Lenticular_OT_configure_calibration(bpy.types.Operator):
    bl_idname = "lenticular.configure_calibration"
    bl_label = "Configure Optical Calibration"
    bl_description = "Set or import optical calibration parameters for the active device"

    device: bpy.props.EnumProperty(
        name="Device",
        items=properties.DEVICE_ITEMS,
        default="GO",
        options={"HIDDEN"},
    )
    x0: bpy.props.FloatProperty(name="X0", default=0.0, precision=6)
    interval: bpy.props.FloatProperty(name="Interval", default=0.0, min=0.0, precision=6)
    slope: bpy.props.FloatProperty(name="Slope", default=0.0, precision=6)

    def invoke(self, context, event):
        settings = context.scene.lenticular
        self.device = settings.device
        if properties.is_scene_calibrated(settings):
            self.x0 = settings.x0
            self.interval = settings.interval
            self.slope = settings.slope
        else:
            calibrated, x0, interval, slope, _ = properties.get_saved_calibration(self.device)
            if calibrated:
                self.x0 = x0
                self.interval = interval
                self.slope = slope
            else:
                self.x0 = 0.0
                self.interval = 0.0
                self.slope = 0.0
        return context.window_manager.invoke_props_dialog(self, width=360)

    def draw(self, context):
        layout = self.layout
        device_name = "Looking Glass Go" if self.device == "GO" else "OpenStage AI C1"
        layout.label(text=f"Device: {device_name}")
        row = layout.row(align=True)
        op = row.operator("lenticular.import_calibration", text="Import JSON…", icon="FILE_FOLDER")
        op.target_device = self.device
        if self.device == "GO":
            row.operator("lenticular.autodetect_calibration", text="Auto-Detect", icon="VIEWZOOM")
        layout.separator()
        layout.label(text="Or enter parameters manually:")
        layout.prop(self, "x0", text="Center" if self.device == "GO" else "X0")
        layout.prop(self, "interval", text="Pitch" if self.device == "GO" else "Interval")
        layout.prop(self, "slope", text="Slope")

    def execute(self, context):
        if not properties.is_calibration_valid(self.x0, self.interval, self.slope):
            self.report({"ERROR"}, "Pitch/Interval must be greater than 0 and Slope must be non-zero")
            return {"CANCELLED"}
        properties.save_device_calibration(
            self.device,
            self.x0,
            self.interval,
            self.slope,
            source="Manual",
            context=context,
        )
        device_name = "Looking Glass Go" if self.device == "GO" else "OpenStage AI C1"
        self.report({"INFO"}, f"Saved {device_name} calibration")
        return {"FINISHED"}


classes = (
    Lenticular_OT_import_calibration,
    Lenticular_OT_autodetect_calibration,
    Lenticular_OT_configure_calibration,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
