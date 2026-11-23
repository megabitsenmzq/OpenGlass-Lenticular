import json
import os
import subprocess
import tempfile
import time
import traceback

import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent
from bpy_extras.io_utils import ExportHelper

from . import properties, realtime
from .realtime import (
    view_pose, camera_view_pose, device_for, view_dimensions_for,
)


_active = None


def is_cycles_active():
    return _active is not None


def active_total_views():
    return _active.total_views if _active is not None else 0


def active_progress_info():
    if _active is None:
        return 0, ""
    total_steps = getattr(_active, "total_steps", _active.total_views)
    label = getattr(_active, "progress_label", "")
    if label:
        return total_steps, f"{total_steps}/{total_steps} {label}"
    return total_steps, f"{_active.settings.cycles_progress}/{total_steps} views"


def cancel_active():
    if _active is not None:
        _active.cancelled = True
        _active.cancel_requested = True
        _active.unloading = True
        _active.finish()


@persistent
def cancel_active_on_load(_filepath=""):
    cancel_active()


def valid_viewport(context):
    return (
        context.area is not None
        and context.area.type == "VIEW_3D"
        and context.space_data is not None
        and _active is None
    )


def suspend_rendered_viewports():
    saved = []
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D" and space.shading.type == "RENDERED":
                    saved.append(space)
                    space.shading.type = "SOLID"
    return saved


def restore_rendered_viewports(saved):
    for space in saved:
        try:
            space.shading.type = "RENDERED"
        except ReferenceError:
            pass


def atlas_path(filepath):
    stem, extension = os.path.splitext(filepath)
    return f"{stem}_atlas{extension or '.png'}"


def interlaced_path(filepath):
    stem, extension = os.path.splitext(filepath)
    return f"{stem}_interlaced{extension or '.mp4'}"


def interlaced_frames_dir(filepath):
    return os.path.splitext(filepath)[0] + "_interlaced_frames"


def project_name(context):
    """Return the .blend filename without its extension, or the scene name if unsaved."""
    if bpy.data.filepath:
        return os.path.splitext(os.path.basename(bpy.data.filepath))[0]
    return context.scene.name


def default_filepath(default_filename, context):
    """Build the default export path with a project-name prefix to avoid overwriting other projects."""
    stem, extension = os.path.splitext(default_filename)
    prefix = project_name(context).replace("/", "_").replace("\\", "_")
    return bpy.path.abspath(f"//{prefix}_{stem}{extension or '.png'}")


def save_png(pixels, width, height, filepath, image_name):
    if pixels.dtype == np.uint8:
        pixels = pixels.astype(np.float32) / 255.0
    image = bpy.data.images.new(image_name, width=width, height=height, alpha=True)
    try:
        image.pixels.foreach_set(pixels.ravel())
        image.filepath_raw = filepath
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)


def save_atlas(atlas, filepath):
    rows, columns = atlas.shape[:2]
    save_png(atlas, columns, rows, filepath, "Lenticular_View_Atlas")


def save_quilt_metadata(filepath, profile, movie=False, duration=10.0):
    metadata = {
        "movie": movie,
        "mediaType": "quilt",
        "quilt_settings": {
            "viewX": profile["columns"],
            "viewY": profile["rows"],
            "viewTotal": profile["count"],
            "invertViews": False,
            "aspect": realtime.FOV_ASPECT,
        },
        "depthiness": 1.0,
        "depthInversion": False,
        "chromaDepth": False,
        "depthPosition": "right",
        "focus": 0.0,
        "viewOrderReversed": False,
        "zoom": 1.0,
        "position_x": 0.0,
        "position_y": 0.0,
        "duration": float(duration),
    }
    with open(filepath + ".json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=4)


def save_composite(atlas, settings, filepath, image_name, profile=None):
    settings.status = "Compositing interlaced image"
    device = device_for(settings)
    profile = profile or realtime.profile_for(settings)
    view_count = profile["count"]
    columns = profile["columns"]
    rows = profile["rows"]
    view_width = atlas.shape[1] // columns
    view_height = atlas.shape[0] // rows
    use_interpolation = getattr(settings, "view_interpolation", True)
    x = np.arange(1440, dtype=np.float32)
    y = np.arange(2560, dtype=np.float32)
    local_x = np.clip((x + 0.5) / 1440.0 * view_width - 0.5, 0.0, view_width - 1.0)
    local_y = np.clip((y + 0.5) / 2560.0 * view_height - 0.5, 0.0, view_height - 1.0)
    x0 = np.floor(local_x).astype(np.int32)
    y0 = np.floor(local_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, view_width - 1)
    y1 = np.minimum(y0 + 1, view_height - 1)
    wx = (local_x - x0)[None, :]
    wy = (local_y - y0)[:, None]
    output = np.ones((2560, 1440, 4), dtype=np.float32)

    def sample_view(view, channel):
        tile_col = view % columns
        tile_row = view // columns if device == "GO" else (rows - 1 - view // columns)
        source_x0 = tile_col * view_width + x0[None, :]
        source_x1 = tile_col * view_width + x1[None, :]
        source_y0 = tile_row * view_height + y0[:, None]
        source_y1 = tile_row * view_height + y1[:, None]
        p00 = atlas[source_y0, source_x0, channel]
        p01 = atlas[source_y0, source_x1, channel]
        p10 = atlas[source_y1, source_x0, channel]
        p11 = atlas[source_y1, source_x1, channel]
        if atlas.dtype == np.uint8:
            p00 = p00.astype(np.float32) / 255.0
            p01 = p01.astype(np.float32) / 255.0
            p10 = p10.astype(np.float32) / 255.0
            p11 = p11.astype(np.float32) / 255.0
        top = p00 * (1.0 - wx) + p01 * wx
        bottom = p10 * (1.0 - wx) + p11 * wx
        return top * (1.0 - wy) + bottom * wy

    if device == "GO":
        phase_base = ((x[None, :] + 0.5) / 1440.0 + (y[:, None] + 0.5) / 2560.0 * settings.slope) * settings.interval - settings.x0 + 0.5 / view_count
        for channel in range(3):
            phase = 1.0 - np.mod(phase_base + channel / (3.0 * 1440.0) * settings.interval, 1.0)
            view = np.clip(phase, 0.00001, 0.99999) * view_count
            lower = np.minimum(np.floor(view).astype(np.int32), view_count - 1)
            if not use_interpolation:
                output[:, :, channel] = sample_view(lower, channel)
            else:
                upper = np.minimum(lower + 1, view_count - 1)
                weight = view - lower
                output[:, :, channel] = sample_view(lower, channel) * (1.0 - weight) + sample_view(upper, channel) * weight
    else:
        interval = max(float(settings.interval), 0.001)
        top_y = 2560.0 - y
        phase_base = ((x[None, :] + 1 + top_y[:, None] * settings.slope) * 3.0 + settings.x0)
        for channel in range(3):
            phase = phase_base + channel
            wrapped = phase - interval * np.floor(phase / interval)
            continuous = np.clip(wrapped / interval, 0.0, 0.99999) * float(view_count)
            choice = np.minimum(np.floor(continuous).astype(np.int32), view_count - 1)
            view_a = (view_count - 1) - choice
            if not use_interpolation:
                output[:, :, channel] = sample_view(view_a, channel)
            else:
                view_b = np.maximum(view_a - 1, 0)
                weight = continuous - choice
                output[:, :, channel] = sample_view(view_a, channel) * (1.0 - weight) + sample_view(view_b, channel) * weight

    save_png(output, 1440, 2560, filepath, image_name)


def open_preview(filepath, source_window=None):
    image = bpy.data.images.load(filepath, check_existing=True)
    image.reload()
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "IMAGE_EDITOR" and area.spaces.active.image == image:
                area.tag_redraw()
                return
    before = {window.as_pointer() for window in bpy.context.window_manager.windows}
    if source_window is not None:
        with bpy.context.temp_override(window=source_window):
            bpy.ops.wm.window_new()
    else:
        bpy.ops.wm.window_new()
    window = next((w for w in bpy.context.window_manager.windows if w.as_pointer() not in before), None)
    if window is None:
        raise RuntimeError("Could not create the preview window")
    area = next((a for a in window.screen.areas if a.type == "VIEW_3D"), window.screen.areas[0])
    area.type = "IMAGE_EDITOR"
    space = area.spaces.active
    space.image = image
    space.display_channels = "COLOR"
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    if region is not None:
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.image.view_zoom_ratio(ratio=1.0)
    area.tag_redraw()


def open_video_preview(filepath, scene, source_window=None):
    render = scene.render
    prev_filepath = render.filepath
    prev_media_type = render.image_settings.media_type
    prev_file_format = render.image_settings.file_format
    prev_ffmpeg_format = render.ffmpeg.format
    try:
        render.filepath = filepath
        render.image_settings.media_type = "VIDEO"
        render.image_settings.file_format = "FFMPEG"
        render.ffmpeg.format = "MPEG4"
        override = {"scene": scene}
        if source_window is not None:
            override["window"] = source_window
        with bpy.context.temp_override(**override):
            bpy.ops.render.play_rendered_anim()
    except Exception as exc:
        print("Lenticular video preview error:", repr(exc))
    finally:
        render.filepath = prev_filepath
        render.image_settings.media_type = prev_media_type
        render.image_settings.file_format = prev_file_format
        render.ffmpeg.format = prev_ffmpeg_format


class LenticularBaseExportMixin(ExportHelper):
    """Shared calibration gating, progress bookkeeping, and cleanup for still/animation export."""

    animation = False
    extension = ".png"
    default_filename = "lenticular_export.png"

    @classmethod
    def poll(cls, context):
        return valid_viewport(context)

    def invoke(self, context, event):
        settings = context.scene.lenticular
        if not properties.is_scene_calibrated(settings):
            if settings.device == "GO" and properties.try_auto_import_go_calibration(context):
                properties.update_device_calibration(settings, "GO", context)
        if not properties.is_scene_calibrated(settings):
            device_name = "Looking Glass Go" if settings.device == "GO" else "OpenStage AI C1"
            self.report({"WARNING"}, f"Configure optical calibration for {device_name} first")
            bpy.ops.lenticular.configure_calibration("INVOKE_DEFAULT")
            return {"CANCELLED"}
        view_perspective = context.space_data.region_3d.view_perspective
        if view_perspective not in {"PERSP", "CAMERA"} or (view_perspective == "CAMERA" and context.scene.camera is None):
            self.report({"ERROR"}, "Switch to Perspective or Camera View with an active camera first")
            return {"CANCELLED"}
        if not self.filepath:
            filename = self.default_filename.replace("lenticular_", "go_", 1) if settings.device == "GO" else self.default_filename
            self.filepath = default_filepath(filename, context)
        return super().invoke(context, event)

    def _prepare_export(self, context):
        if not self.filepath.lower().endswith(self.extension):
            self.filepath += self.extension
        frames = []
        if self.animation:
            frames = list(range(context.scene.frame_start, context.scene.frame_end + 1, context.scene.frame_step))
            if not frames:
                self.report({"ERROR"}, "The animation range contains no renderable frames")
                return None
        if realtime._session is not None:
            if not realtime._session.paused:
                bpy.ops.lenticular.pause_realtime()
            self.report({"INFO"}, "Live output was automatically paused during rendering")
        self.scene = context.scene
        self.settings = context.scene.lenticular
        self.space = context.space_data
        self.device = device_for(self.settings)
        self.profile = realtime.profile_for(self.settings)
        self.view_count = self.profile["count"]
        self.view_width, self.view_height = view_dimensions_for(self.settings)
        self.index = 0
        self.total_views = len(frames) * self.view_count if self.animation else self.view_count
        self.total_steps = self.total_views + 1
        self.progress_label = ""
        self.progress_stage = None
        self.frames = frames
        self.frames_directory = os.path.splitext(self.filepath)[0] + "_frames" if self.animation else None
        self.interlaced_frames_directory = interlaced_frames_dir(self.filepath) if self.animation else None
        self.cancelled = False
        self.cancel_requested = False
        self.unloading = False
        self.timer = None
        self.process = None
        self.temp_dir = None
        self.log_path = None
        self.offscreen = None
        self.saved_shading = None
        return frames

    def worker_log(self):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as handle:
                return "".join(handle.readlines()[-12:])
        except OSError as exc:
            return str(exc)

    def _terminate_process(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            except OSError:
                pass
        self.process = None

    def finish(self):
        global _active
        try:
            self._terminate_process()
            if self.temp_dir is not None:
                try:
                    self.temp_dir.cleanup()
                except OSError:
                    pass
                self.temp_dir = None
            if getattr(self, "offscreen", None) is not None:
                try:
                    self.offscreen.free()
                except Exception:
                    pass
                self.offscreen = None
            if self.timer is not None:
                try:
                    bpy.context.window_manager.event_timer_remove(self.timer)
                except Exception:
                    pass
                self.timer = None
        finally:
            if _active is self:
                _active = None
            restore_rendered_viewports(self.saved_shading or [])
            self.saved_shading = None
            # The modal handler writes the final status later; only the cancelled state needs to be set here.
            if self.cancelled:
                try:
                    self.settings.status = "Cancelled"
                except (AttributeError, ReferenceError):
                    pass
            properties.tag_redraw_view3d()


class LenticularOffscreenExportMixin(LenticularBaseExportMixin):
    default_filename = "lenticular_offscreen.png"

    def execute(self, context):
        global _active
        region = next((r for r in context.area.regions if r.type == "WINDOW"), None)
        if region is None:
            self.report({"ERROR"}, "Could not find the 3D viewport window region")
            return {"CANCELLED"}
        frames = self._prepare_export(context)
        if frames is None:
            return {"CANCELLED"}
        self.area = context.area
        self.window = context.window
        self.region = region
        self.region_3d = context.space_data.region_3d
        self.completed_views = 0
        self.frame_position = 0
        self.atlas_paths = []
        self.interlaced_paths = []
        self.background_mode = None
        self._next_render_at = 0.0
        prefix = "lenticular_video_" if self.animation else "lenticular_offscreen_"
        self.temp_dir = tempfile.TemporaryDirectory(prefix=prefix)
        self.log_path = os.path.join(self.temp_dir.name, "worker.log")
        if self.animation:
            os.makedirs(self.frames_directory, exist_ok=True)
            os.makedirs(self.interlaced_frames_directory, exist_ok=True)
        self.offscreen = gpu.types.GPUOffScreen(self.view_width, self.view_height, format="RGBA8")
        self.atlas_np = np.zeros((self.profile["rows"] * self.view_height, self.profile["columns"] * self.view_width, 4), dtype=np.uint8)
        _active = self
        self.settings.cycles_progress = 0
        self.settings.status = f"Viewport render: {self.view_count} views, 0/{self.total_views}"
        try:
            self.saved_shading = suspend_rendered_viewports()
            self.timer = context.window_manager.event_timer_add(0.05, window=context.window)
            context.window_manager.modal_handler_add(self)
        except Exception:
            self.finish()
            raise
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if _active is not self:
            return {"FINISHED"}
        if event.type == "ESC":
            self.cancelled = True
            self.finish()
            self.report({"INFO"}, "Rendering cancelled")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"RUNNING_MODAL"}
        try:
            if self.background_mode is not None:
                if self.process.poll() is None:
                    return {"RUNNING_MODAL"}
                if self.process.returncode != 0:
                    raise RuntimeError("Background export worker failed:\n" + self.worker_log())
                if self.background_mode == "VIDEO":
                    output_paths = [self.filepath, interlaced_path(self.filepath)]
                    missing = [path for path in output_paths if not os.path.isfile(path)]
                    if missing:
                        raise RuntimeError("Background video encoding did not generate output files: " + ", ".join(missing) + "\n" + self.worker_log())
                    self.finish()
                    open_video_preview(interlaced_path(self.filepath), self.scene, source_window=self.window)
                    self.settings.status = (
                        f"Exported video: {self.filepath}; interlaced: {interlaced_path(self.filepath)}; "
                        f"frame atlases: {self.frames_directory}"
                    )
                    self.report({"INFO"}, f"Exported viewport animation video: {self.filepath}")
                    return {"FINISHED"}
                tiled_path = atlas_path(self.filepath)
                output_paths = [self.filepath, tiled_path]
                missing = [path for path in output_paths if not os.path.isfile(path)]
                if missing:
                    raise RuntimeError("Background image saving did not generate output files: " + ", ".join(missing) + "\n" + self.worker_log())
                self.finish()
                open_preview(self.filepath, source_window=self.window)
                self.settings.status = f"Exported: {self.filepath}; atlas: {tiled_path}"
                self.report({"INFO"}, f"Exported interlaced image and {self.view_count}-view atlas: {tiled_path}")
                return {"FINISHED"}
            if self.index < self.view_count:
                if time.monotonic() < self._next_render_at:
                    return {"RUNNING_MODAL"}
                if self.animation and self.index == 0:
                    self.scene.frame_set(self.frames[self.frame_position])
                self.render_current_view()
                self._next_render_at = time.monotonic() + 0.01
                return {"RUNNING_MODAL"}
            if self.animation:
                path = os.path.join(self.frames_directory, f"frame_{self.frame_position:06d}.png")
                save_atlas(self.atlas_np, path)
                self.atlas_paths.append(path)
                comp_path = os.path.join(self.interlaced_frames_directory, f"frame_{self.frame_position:06d}.png")
                save_composite(
                    self.atlas_np, self.settings, comp_path,
                    f"Lenticular_Interlaced_{self.frame_position:06d}", profile=self.profile,
                )
                self.interlaced_paths.append(comp_path)
                self.frame_position += 1
                if self.frame_position < len(self.frames):
                    self.index = 0
                    self.atlas_np.fill(0)
                    self._next_render_at = time.monotonic() + 0.01
                    return {"RUNNING_MODAL"}
                self.start_encoding()
                return {"RUNNING_MODAL"}
            self.start_still_save()
            return {"RUNNING_MODAL"}
        except Exception as exc:
            self.finish()
            self.settings.status = f"Viewport export failed: {exc}"
            self.report({"ERROR"}, f"Export failed: {exc}")
            print("Lenticular offscreen error:", repr(exc))
            traceback.print_exc()
            return {"CANCELLED"}

    def start_still_save(self):
        self.background_mode = "STILL"
        self.progress_label = "Compositing interlaced image…"
        self.settings.status = f"Viewport rendering of {self.view_count} views complete; saving interlaced image and atlas"
        atlas_npy_path = os.path.join(self.temp_dir.name, "atlas.npy")
        np.save(atlas_npy_path, self.atlas_np, allow_pickle=False)
        self.atlas_np = None
        config_path = os.path.join(self.temp_dir.name, "encoder.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump({
                "still_atlas_npy": atlas_npy_path,
                "output": self.filepath,
                "device": self.device,
                "profile": self.profile,
                "x0": self.settings.x0,
                "interval": self.settings.interval,
                "slope": self.settings.slope,
                "view_interpolation": self.settings.view_interpolation,
            }, handle)
        worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_encoder.py")
        with open(self.log_path, "w", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                [bpy.app.binary_path, "--background", "--factory-startup", "--python-exit-code", "1", "--python", worker_path, "--", config_path],
                stdout=log,
                stderr=subprocess.STDOUT,
            )

    def start_encoding(self):
        self.background_mode = "VIDEO"
        self.progress_label = "Encoding MP4 video…"
        self.settings.status = f"Viewport rendering of {self.view_count} views complete; encoding {len(self.atlas_paths)} MP4 frames"
        config_path = os.path.join(self.temp_dir.name, "encoder.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump({
                "atlas_paths": self.atlas_paths,
                "interlaced_paths": self.interlaced_paths,
                "output": self.filepath,
                "fps": self.scene.render.fps,
                "fps_base": self.scene.render.fps_base,
                "device": self.device,
                "profile": self.profile,
                "view_width": self.view_width,
                "view_height": self.view_height,
                "x0": self.settings.x0,
                "interval": self.settings.interval,
                "slope": self.settings.slope,
                "view_interpolation": self.settings.view_interpolation,
            }, handle)
        worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_encoder.py")
        with open(self.log_path, "w", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                [bpy.app.binary_path, "--background", "--factory-startup", "--python-exit-code", "1", "--python", worker_path, "--", config_path],
                stdout=log,
                stderr=subprocess.STDOUT,
            )

    def render_current_view(self):
        self.region_3d.update()
        view_matrix, projection = view_pose(self.region_3d, self.index, self.scene, self.device, self.view_count)
        realtime._offscreen_draw_depth += 1
        try:
            with bpy.context.temp_override(window=self.window, area=self.area, region=self.region):
                with self.offscreen.bind():
                    gpu.state.scissor_test_set(False)
                    gpu.state.viewport_set(0, 0, self.view_width, self.view_height)
                    self.offscreen.draw_view3d(
                        self.scene,
                        bpy.context.view_layer,
                        self.space,
                        self.region,
                        view_matrix,
                        projection,
                        do_color_management=True,
                    )
        finally:
            realtime._offscreen_draw_depth -= 1
            self.region_3d.update()
        buf = self.offscreen.texture_color.read()
        # draw_view3d(do_color_management=True) already outputs display-referred sRGB values.
        # save_png writes them unchanged using the default sRGB colorspace; do not apply gamma conversion here,
        # otherwise this path would disagree with live preview, final export, and video encoding (sRGB sequencer).
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(self.view_height, self.view_width, 4)
        column = self.index % self.profile["columns"]
        row = self.index // self.profile["columns"] if self.device == "GO" else self.profile["rows"] - 1 - self.index // self.profile["columns"]
        self.atlas_np[row * self.view_height:(row + 1) * self.view_height, column * self.view_width:(column + 1) * self.view_width] = arr
        self.index += 1
        self.completed_views += 1
        self.settings.cycles_progress = self.completed_views
        if self.animation:
            self.settings.status = (
                f"Viewport animation frame {self.frame_position + 1}/{len(self.frames)} "
                f"(scene frame {self.frames[self.frame_position]}) | View {self.completed_views}/{self.total_views}"
            )
        else:
            self.settings.status = f"Viewport render: {self.view_count} views, {self.index}/{self.view_count}"


class LenticularFinalExportMixin(LenticularBaseExportMixin):
    default_filename = "lenticular_final.png"

    def execute(self, context):
        global _active
        frames = self._prepare_export(context)
        if frames is None:
            return {"CANCELLED"}
        self.engine = self.scene.render.engine
        self.window = context.window
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lenticular_final_")
        self.log_path = os.path.join(self.temp_dir.name, "worker.log")
        _active = self
        try:
            self.start_worker()
            self.saved_shading = suspend_rendered_viewports()
            self.settings.cycles_progress = 0
            job_name = "Background animation render" if self.animation else "Background final render"
            self.settings.status = f"{job_name} {self.engine} 0/{self.total_views}"
            self.timer = context.window_manager.event_timer_add(0.2, window=context.window)
            context.window_manager.modal_handler_add(self)
        except Exception:
            self.finish()
            raise
        return {"RUNNING_MODAL"}

    def start_worker(self):
        source_camera = self.scene.camera if self.space.region_3d.view_perspective == "CAMERA" else None
        poses = [camera_view_pose(self.space.region_3d, index, self.scene, self.device, self.view_count) for index in range(self.view_count)]
        config = {
            "scene": self.scene.name,
            "view_layer": bpy.context.view_layer.name,
            "directory": self.temp_dir.name,
            "output": self.filepath,
            "clip_start": source_camera.data.clip_start if source_camera else self.space.clip_start,
            "clip_end": source_camera.data.clip_end if source_camera else self.space.clip_end,
            "device": self.device,
            "profile": self.profile,
            "x0": self.settings.x0,
            "interval": self.settings.interval,
            "slope": self.settings.slope,
            "view_interpolation": self.settings.view_interpolation,
            "view_width": self.view_width,
            "view_height": self.view_height,
            "pixel_aspect_x": self.scene.render.pixel_aspect_x,
            "pixel_aspect_y": self.scene.render.pixel_aspect_y,
            "poses": [
                {
                    "matrix": [list(row) for row in matrix],
                    "projection": [list(row) for row in projection],
                    "offset": offset,
                    "distance": distance,
                }
                for matrix, projection, offset, distance in poses
            ],
        }
        if self.animation:
            config.update({
                "frame_start": self.scene.frame_start,
                "frame_end": self.scene.frame_end,
                "frame_step": self.scene.frame_step,
                "frames_directory": self.frames_directory,
                "interlaced_frames_directory": self.interlaced_frames_directory,
                "camera_mode": source_camera is not None,
                "source_camera": source_camera.name if source_camera else None,
            })
        config_path = os.path.join(self.temp_dir.name, "config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        scene_path = os.path.join(self.temp_dir.name, "scene.blend")
        self.settings.status = "Saving a scene copy for background rendering"
        bpy.ops.wm.save_as_mainfile(filepath=scene_path, copy=True, relative_remap=True, compress=True, check_existing=False)
        worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_worker.py")
        with open(self.log_path, "w", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                [bpy.app.binary_path, "--background", scene_path, "--python-exit-code", "1", "--python", worker_path, "--", config_path],
                stdout=log,
                stderr=subprocess.STDOUT,
            )

    def modal(self, context, event):
        if _active is not self:
            return {"FINISHED"}
        if event.type == "ESC":
            self.cancelled = True
            self.cancel_requested = True
            self.finish()
            self.report({"INFO"}, "Background rendering cancelled")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"RUNNING_MODAL"}
        try:
            self.read_progress()
            if context.area is not None:
                context.area.tag_redraw()
            if self.process.poll() is None:
                return {"RUNNING_MODAL"}
            if self.process.returncode != 0:
                raise RuntimeError("Background render process failed:\n" + self.worker_log())
            tiled_path = self.frames_directory if self.animation else atlas_path(self.filepath)
            if self.animation:
                output_paths = [self.filepath, interlaced_path(self.filepath)]
            else:
                output_paths = [self.filepath, tiled_path]
            missing = [path for path in output_paths if not os.path.isfile(path)]
            if missing:
                raise RuntimeError("Background rendering did not generate output files: " + ", ".join(missing) + "\n" + self.worker_log())
            self.finish()
            if self.animation:
                open_video_preview(interlaced_path(self.filepath), self.scene, source_window=self.window)
                self.settings.status = (
                    f"Exported video: {self.filepath}; interlaced: {interlaced_path(self.filepath)}; "
                    f"frame atlases: {tiled_path}"
                )
                self.report({"INFO"}, f"Exported {self.view_count}-view animation video: {self.filepath}")
            else:
                open_preview(self.filepath, source_window=self.window)
                self.settings.status = f"Exported: {self.filepath}; atlas: {tiled_path}"
                self.report({"INFO"}, f"Exported interlaced image and {self.view_count}-view atlas: {tiled_path}")
            return {"FINISHED"}
        except Exception as exc:
            self.finish()
            self.settings.status = f"Final render failed: {exc}"
            self.report({"ERROR"}, f"Final render failed: {exc}")
            print("Lenticular final render error:", repr(exc))
            traceback.print_exc()
            return {"CANCELLED"}

    def read_progress(self):
        try:
            with open(os.path.join(self.temp_dir.name, "progress"), encoding="ascii") as handle:
                progress = json.load(handle)
            index = int(progress["views"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return
        index = min(index, self.total_views)
        stage = progress.get("stage")
        if index != self.index or stage != self.progress_stage:
            self.index = index
            self.progress_stage = stage
            self.settings.cycles_progress = index
            if stage == "compositing":
                self.progress_label = "Compositing interlaced image…"
                self.settings.status = "Compositing interlaced image…"
            elif stage == "encoding":
                self.progress_label = "Encoding MP4 video…"
                self.settings.status = "Encoding MP4 video…"
            elif self.animation:
                self.progress_label = ""
                self.settings.status = (
                    f"Background animation render {self.engine} frame {progress['frame_index']}/{progress['frame_count']} "
                    f"(scene frame {progress['frame']}) | View {index}/{self.total_views}"
                )
            else:
                self.progress_label = ""
                self.settings.status = f"Background final render {self.engine} {index}/{self.total_views}"


class Lenticular_OT_offscreen_export(LenticularOffscreenExportMixin, bpy.types.Operator):
    bl_idname = "lenticular.offscreen_export"
    bl_label = "Render Viewport (All Views)"
    bl_description = "Render all views from the viewport and export an interlaced PNG and atlas"
    filename_ext = ".png"
    filter_glob: bpy.props.StringProperty(default="*.png", options={"HIDDEN"})


class Lenticular_OT_offscreen_animation_export(LenticularOffscreenExportMixin, bpy.types.Operator):
    bl_idname = "lenticular.offscreen_animation_export"
    bl_label = "Render Viewport Animation (All Views per Frame)"
    bl_description = "Render all views per frame from the viewport and export an MP4 animation"
    filename_ext = ".mp4"
    filter_glob: bpy.props.StringProperty(default="*.mp4", options={"HIDDEN"})
    animation = True
    extension = ".mp4"
    default_filename = "lenticular_viewport_animation.mp4"


class Lenticular_OT_final_export(LenticularFinalExportMixin, bpy.types.Operator):
    bl_idname = "lenticular.final_export"
    bl_label = "Render Final Image (All Views)"
    bl_description = "Render all views with the active render engine and export an interlaced PNG and atlas"
    filename_ext = ".png"
    filter_glob: bpy.props.StringProperty(default="*.png", options={"HIDDEN"})


class Lenticular_OT_animation_export(LenticularFinalExportMixin, bpy.types.Operator):
    bl_idname = "lenticular.animation_export"
    bl_label = "Render Final Animation (All Views per Frame)"
    bl_description = "Render all views per frame with the active render engine and export an MP4 animation"
    filename_ext = ".mp4"
    filter_glob: bpy.props.StringProperty(default="*.mp4", options={"HIDDEN"})
    animation = True
    extension = ".mp4"
    default_filename = "lenticular_animation.mp4"


classes = (
    Lenticular_OT_offscreen_export,
    Lenticular_OT_offscreen_animation_export,
    Lenticular_OT_final_export,
    Lenticular_OT_animation_export,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    if cancel_active_on_load not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(cancel_active_on_load)


def unregister():
    if cancel_active_on_load in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(cancel_active_on_load)
    cancel_active()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
