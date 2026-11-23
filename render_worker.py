import importlib
import json
import os
import sys
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Matrix, Vector

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_parent = os.path.dirname(_pkg_dir)
_pkg_name = __package__ or os.path.basename(_pkg_dir)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)
if __name__ == "__main__":
    for module_name in tuple(sys.modules):
        if module_name == _pkg_name or module_name.startswith(f"{_pkg_name}."):
            del sys.modules[module_name]

_cycles_render = importlib.import_module(f"{_pkg_name}.cycles_render")
_realtime = importlib.import_module(f"{_pkg_name}.realtime")
_video_encoder = importlib.import_module(f"{_pkg_name}.video_encoder")

atlas_path = _cycles_render.atlas_path
interlaced_frames_dir = _cycles_render.interlaced_frames_dir
save_atlas = _cycles_render.save_atlas
save_composite = _cycles_render.save_composite
save_quilt_metadata = _cycles_render.save_quilt_metadata

DEVICE_PROFILES = _realtime.DEVICE_PROFILES
VIEW_WIDTH = _realtime.VIEW_WIDTH
VIEW_HEIGHT = _realtime.VIEW_HEIGHT
VIEW_COLUMNS = _realtime.VIEW_COLUMNS
VIEW_ROWS = _realtime.VIEW_ROWS
VIEW_COUNT = _realtime.VIEW_COUNT
OUTPUT_WIDTH = _realtime.OUTPUT_WIDTH
OUTPUT_HEIGHT = _realtime.OUTPUT_HEIGHT
apply_camera_pose_shift = _realtime.apply_camera_pose_shift
camera_view_pose = _realtime.camera_view_pose
configure_anamorphic_camera = _realtime.configure_anamorphic_camera

encode_video = _video_encoder.encode_video


def _write_progress(directory, progress):
    progress_path = os.path.join(directory, "progress.tmp")
    with open(progress_path, "w", encoding="ascii") as handle:
        json.dump(progress, handle)
    os.replace(progress_path, os.path.join(directory, "progress"))


def main():
    global VIEW_WIDTH, VIEW_COLUMNS, VIEW_ROWS, VIEW_COUNT
    with open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8") as handle:
        config = json.load(handle)
    device = config.get("device", "GO")
    profile = config.get("profile", DEVICE_PROFILES[device])
    VIEW_WIDTH = int(config.get("view_width", profile["width"]))
    VIEW_COLUMNS = profile["columns"]
    VIEW_ROWS = profile["rows"]
    VIEW_COUNT = profile["count"]
    scene = bpy.data.scenes[config["scene"]]
    view_layer_name = config.get("view_layer")
    view_layer = scene.view_layers.get(view_layer_name) if view_layer_name else None
    if view_layer is None:
        view_layer = scene.view_layers[0]
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
        bpy.context.window.view_layer = view_layer
    view_height = int(config.get("view_height", VIEW_HEIGHT))
    render = scene.render
    camera_data = bpy.data.cameras.new("Lenticular_Worker_Camera")
    configure_anamorphic_camera(
        render, camera_data, VIEW_WIDTH, view_height, config["clip_start"], config["clip_end"]
    )
    camera_object = bpy.data.objects.new("Lenticular_Worker_Camera", camera_data)
    scene.collection.objects.link(camera_object)
    scene.camera = camera_object

    directory = config["directory"]
    animation = "frame_start" in config
    frames = list(range(config["frame_start"], config["frame_end"] + 1, config["frame_step"])) if animation else [scene.frame_current]
    if not frames:
        raise ValueError("The animation range contains no renderable frames")
    frames_directory = config.get("frames_directory")
    if frames_directory:
        os.makedirs(frames_directory, exist_ok=True)
    interlaced_directory = config.get("interlaced_frames_directory") or interlaced_frames_dir(config["output"])
    if animation:
        os.makedirs(interlaced_directory, exist_ok=True)
    atlas_paths = []
    interlaced_paths = []
    completed_views = 0

    for frame_index, frame in enumerate(frames):
        scene.frame_set(frame)
        poses = config["poses"]
        if animation and config.get("camera_mode"):
            source_camera = bpy.data.objects.get(config["source_camera"])
            if source_camera is None:
                raise RuntimeError("Could not find the active camera for the animation scene")
            scene.camera = source_camera
            rotation = source_camera.matrix_world.to_quaternion()
            focus = source_camera.matrix_world.translation - rotation @ Vector((0.0, 0.0, poses[0]["distance"]))
            camera_region = SimpleNamespace(view_perspective="CAMERA", view_location=focus)
            # camera_view_pose calls calc_matrix_camera using the scene pixel aspect ratio.
            # Recompute poses with the original pixel aspect ratio serialized by the GUI (normally 1:1).
            # The worker has already changed it to the anamorphic value; failing to restore it would
            # scale the effective aspect ratio twice and corrupt every view's camera FOV and stereo offset.
            render.pixel_aspect_x = float(config.get("pixel_aspect_x", 1.0))
            render.pixel_aspect_y = float(config.get("pixel_aspect_y", 1.0))
            poses = [
                {
                    "matrix": [list(row) for row in matrix],
                    "projection": [list(row) for row in projection],
                    "offset": offset,
                    "distance": distance,
                }
                for matrix, projection, offset, distance in (
                    camera_view_pose(camera_region, index, scene, device, VIEW_COUNT) for index in range(VIEW_COUNT)
                )
            ]
            render.pixel_aspect_x = OUTPUT_WIDTH / VIEW_WIDTH
            render.pixel_aspect_y = OUTPUT_HEIGHT / view_height
            scene.camera = camera_object

        atlas = np.zeros((VIEW_ROWS * view_height, VIEW_COLUMNS * VIEW_WIDTH, 4), dtype=np.float32)
        for index, pose in enumerate(poses):
            apply_camera_pose_shift(
                camera_object,
                camera_data,
                Matrix(pose["matrix"]),
                Matrix(pose["projection"]),
                pose["offset"],
                pose["distance"],
            )
            render.filepath = os.path.join(directory, f"view_{index:02d}.png")
            view_layer.update()
            bpy.ops.render.render(write_still=True, scene=scene.name)

            image = bpy.data.images.load(render.filepath, check_existing=False)
            try:
                pixels = np.empty(VIEW_WIDTH * view_height * 4, dtype=np.float32)
                image.pixels.foreach_get(pixels)
                pixels = pixels.reshape(view_height, VIEW_WIDTH, 4)
                column = index % VIEW_COLUMNS
                row = index // VIEW_COLUMNS if device == "GO" else VIEW_ROWS - 1 - index // VIEW_COLUMNS
                x = column * VIEW_WIDTH
                y = row * view_height
                atlas[y:y + view_height, x:x + VIEW_WIDTH] = pixels
            finally:
                bpy.data.images.remove(image)

            completed_views += 1
            _write_progress(
                directory,
                {
                    "views": completed_views,
                    "frame_index": frame_index + 1,
                    "frame_count": len(frames),
                    "frame": frame,
                },
            )
            print(f"Lenticular progress frame {frame_index + 1}/{len(frames)} view {index + 1}/{VIEW_COUNT}", flush=True)

        settings = SimpleNamespace(
            device=device, x0=config["x0"], interval=config["interval"], slope=config["slope"],
            view_interpolation=config.get("view_interpolation", True), status="",
        )
        if animation:
            path = os.path.join(frames_directory, f"frame_{frame_index:06d}.png")
            save_atlas(atlas, path)
            atlas_paths.append(path)
            comp_path = os.path.join(interlaced_directory, f"frame_{frame_index:06d}.png")
            save_composite(atlas, settings, comp_path, f"Lenticular_Interlaced_{frame_index:06d}", profile=profile)
            interlaced_paths.append(comp_path)
        else:
            _write_progress(
                directory,
                {
                    "views": completed_views,
                    "frame_index": frame_index + 1,
                    "frame_count": len(frames),
                    "frame": frame,
                    "stage": "compositing",
                },
            )
            tiled_path = atlas_path(config["output"])
            save_atlas(atlas, tiled_path)
            save_quilt_metadata(tiled_path, profile)
            save_composite(atlas, settings, config["output"], "Lenticular_Final_Interlaced", profile=profile)

    if animation:
        _write_progress(
            directory,
            {
                "views": completed_views,
                "frame_index": len(frames),
                "frame_count": len(frames),
                "frame": frames[-1],
                "stage": "encoding",
            },
        )
        encode_video(
            atlas_paths, config["output"], scene.render.fps, scene.render.fps_base, view_height,
            {"x0": config["x0"], "interval": config["interval"], "slope": config["slope"], "view_interpolation": config.get("view_interpolation", True)},
            device=device, view_width=VIEW_WIDTH, profile=profile, interlaced_paths=interlaced_paths,
        )
    print("Lenticular complete", flush=True)


if __name__ == "__main__":
    main()
