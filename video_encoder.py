import importlib
import json
import os
import sys
from types import SimpleNamespace

import bpy
import numpy as np

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_parent = os.path.dirname(_pkg_dir)
_pkg_name = __package__ or os.path.basename(_pkg_dir)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)
if __name__ == "__main__":
    for module_name in tuple(sys.modules):
        if module_name == _pkg_name or module_name.startswith(f"{_pkg_name}."):
            del sys.modules[module_name]

_realtime = importlib.import_module(f"{_pkg_name}.realtime")
_cycles_render = importlib.import_module(f"{_pkg_name}.cycles_render")

DEVICE_PROFILES = _realtime.DEVICE_PROFILES
VIEW_HEIGHT = _realtime.VIEW_HEIGHT
save_composite = _cycles_render.save_composite
interlaced_path = _cycles_render.interlaced_path
save_quilt_metadata = _cycles_render.save_quilt_metadata


def encode_sequence(frame_paths, output, fps, fps_base, width, height, scene_name):
    if not frame_paths:
        raise ValueError("No frames provided for video encoding")
    encoding_scene = bpy.data.scenes.new(scene_name)
    try:
        render = encoding_scene.render
        render.resolution_x = width
        render.resolution_y = height
        render.resolution_percentage = 100
        render.fps = fps
        render.fps_base = fps_base
        encoding_scene.frame_start = 1
        encoding_scene.frame_end = len(frame_paths)
        render.image_settings.media_type = "VIDEO"
        render.image_settings.file_format = "FFMPEG"
        render.image_settings.color_mode = "RGB"
        render.ffmpeg.format = "MPEG4"
        render.ffmpeg.codec = "H264"
        render.ffmpeg.constant_rate_factor = "HIGH"
        render.ffmpeg.ffmpeg_preset = "GOOD"
        render.ffmpeg.audio_codec = "NONE"
        render.filepath = output
        encoding_scene.view_settings.view_transform = "Standard"
        encoding_scene.view_settings.look = "None"
        encoding_scene.view_settings.exposure = 0.0
        encoding_scene.view_settings.gamma = 1.0
        encoding_scene.sequencer_colorspace_settings.name = "sRGB"

        editor = encoding_scene.sequence_editor_create()
        strip = editor.strips.new_image("frames", frame_paths[0], channel=1, frame_start=1)
        for path in frame_paths[1:]:
            strip.elements.append(os.path.basename(path))
        strip.frame_final_duration = len(frame_paths)
        print(f"Lenticular encoding {len(frame_paths)} frames {width}x{height} -> {output}", flush=True)
        bpy.ops.render.render(animation=True, scene=encoding_scene.name)
    finally:
        bpy.data.scenes.remove(encoding_scene)


def encode_video(atlas_paths, output, fps, fps_base, view_height=VIEW_HEIGHT, calibration=None, device="GO", view_width=None, profile=None, interlaced_paths=None):
    profile = profile or DEVICE_PROFILES[device]
    view_width = view_width or profile["width"]
    encode_sequence(
        atlas_paths, output, fps, fps_base,
        profile["columns"] * view_width, profile["rows"] * view_height, "Lenticular_Video_Encoder",
    )
    effective_fps = (fps / fps_base) if (fps and fps_base) else 0.0
    duration = round(len(atlas_paths) / effective_fps, 6) if effective_fps > 0.0 else 10.0
    save_quilt_metadata(output, profile, movie=True, duration=duration)
    if not calibration:
        return None
    target = interlaced_path(output)
    if interlaced_paths and len(interlaced_paths) == len(atlas_paths) and all(os.path.isfile(p) for p in interlaced_paths):
        paths = interlaced_paths
    else:
        settings = SimpleNamespace(
            device=device, x0=calibration["x0"], interval=calibration["interval"], slope=calibration["slope"],
            view_interpolation=calibration.get("view_interpolation", True), status="",
        )
        frames_directory = os.path.splitext(output)[0] + "_interlaced_frames"
        os.makedirs(frames_directory, exist_ok=True)
        paths = []
        for index, atlas_frame in enumerate(atlas_paths):
            image = bpy.data.images.load(atlas_frame, check_existing=False)
            try:
                pixels = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
                image.pixels.foreach_get(pixels)
                pixels = pixels.reshape(image.size[1], image.size[0], 4)
            finally:
                bpy.data.images.remove(image)
            path = os.path.join(frames_directory, f"frame_{index:06d}.png")
            save_composite(pixels, settings, path, f"Lenticular_Interlaced_{index:06d}", profile=profile)
            paths.append(path)
    encode_sequence(paths, target, fps, fps_base, 1440, 2560, "Lenticular_Interlaced_Encoder")
    return target


def main():
    with open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8") as handle:
        config = json.load(handle)
    calibration = None
    if "x0" in config:
        calibration = {"x0": config["x0"], "interval": config["interval"], "slope": config["slope"], "view_interpolation": config.get("view_interpolation", True)}
    encode_video(
        config["atlas_paths"], config["output"], config["fps"], config["fps_base"],
        int(config.get("view_height", VIEW_HEIGHT)), calibration,
        device=config.get("device", "GO"), view_width=config.get("view_width"), profile=config.get("profile"),
        interlaced_paths=config.get("interlaced_paths"),
    )


if __name__ == "__main__":
    main()
