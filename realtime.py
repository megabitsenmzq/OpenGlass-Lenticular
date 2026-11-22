import math
import os
import tempfile
import time
import traceback

import bpy
import blf
import gpu
import numpy as np
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import properties
from .shaders import make_downsample_shader, make_shaders


VIEW_COLUMNS = 5
VIEW_ROWS = 9
VIEW_WIDTH = 372
VIEW_HEIGHT = 2560
VIEW_COUNT = 45
OUTPUT_WIDTH = 1440
OUTPUT_HEIGHT = 2560
FOV_ASPECT = OUTPUT_WIDTH / OUTPUT_HEIGHT
DEVICE_PROFILES = {
    "C1": {
        "columns": 5, "rows": 9, "width": 372, "height": 682,
        "count": 45, "cone": 40.0,
    },
    "GO": {
        "columns": 5, "rows": 9, "width": 372, "height": 682,
        "count": 45, "cone": 54.0,
    },
}
GO_GRID_LAYOUTS = properties.GRID_LAYOUTS
FAST_REFINE_DELAY = 0.3
FAST_REFINE_VIEW_COUNT = 10
_session = None
_screen_plane_draw_handle = None
_offscreen_draw_depth = 0


def device_for(settings):
    return getattr(settings, "device", "GO")


def profile_for(settings):
    profile = DEVICE_PROFILES[device_for(settings)]
    columns, rows = GO_GRID_LAYOUTS.get(getattr(settings, "go_grid", "5x9"), GO_GRID_LAYOUTS["5x9"])
    return {**profile, "columns": columns, "rows": rows, "count": columns * rows}


def stereo_separation_for(settings):
    count = profile_for(settings)["count"]
    return max(1, min(count - 1, round(count * 0.2)))


def refinement_view_indices(view_count, count=FAST_REFINE_VIEW_COUNT):
    count = min(count, view_count)
    if count == 1:
        return (0,)
    return tuple(round(index * (view_count - 1) / (count - 1)) for index in range(count))


def refinement_targets_for(view_count, views):
    targets = [[] for _ in views]
    for target in range(view_count):
        position = min(range(len(views)), key=lambda index: abs(views[index] - target))
        targets[position].append(target)
    return tuple(tuple(group) for group in targets)


def view_dimensions_for(settings):
    profile = profile_for(settings)
    width = 720 if settings.resolution_mode == "FULL" else profile["width"]
    height = OUTPUT_HEIGHT if settings.resolution_mode == "FULL" else profile["height"]
    width = max(1, width + getattr(settings, "compensation_x", 0))
    height = max(1, height + getattr(settings, "compensation_y", 0))
    return width, height


def view_height_for(settings):
    return view_dimensions_for(settings)[1]


def make_offscreen(width, height):
    return gpu.types.GPUOffScreen(width, height)


def reset_fullscreen_gpu_state():
    gpu.state.scissor_test_set(False)
    gpu.state.face_culling_set("NONE")
    gpu.state.color_mask_set(True, True, True, True)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def max_gpu_texture_size():
    try:
        size = int(gpu.capabilities.max_texture_size_get())
        if size > 0:
            return size
    except Exception:
        pass
    return 16384


def has_valid_view(region_3d, scene):
    if region_3d is None:
        return False
    if region_3d.view_perspective == "PERSP":
        return True
    return region_3d.view_perspective == "CAMERA" and scene is not None and scene.camera is not None


def draw_view3d_at_output_scale(
    offscreen,
    scene,
    view_layer,
    space,
    region,
    view_matrix,
    projection,
    shader,
    batch,
    output_scale=False,
):
    def draw(target, width, height):
        global _offscreen_draw_depth
        _offscreen_draw_depth += 1
        try:
            with target.bind():
                gpu.state.scissor_test_set(False)
                gpu.state.viewport_set(0, 0, width, height)
                target.draw_view3d(
                    scene,
                    view_layer,
                    space,
                    region,
                    view_matrix,
                    projection,
                    do_color_management=True,
                )
        finally:
            _offscreen_draw_depth -= 1

    if (
        not output_scale
        or (not space.overlay.show_overlays and not space.show_gizmo)
        or (offscreen.width == OUTPUT_WIDTH and offscreen.height == OUTPUT_HEIGHT)
    ):
        draw(offscreen, offscreen.width, offscreen.height)
        return

    full = make_offscreen(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    try:
        draw(full, OUTPUT_WIDTH, OUTPUT_HEIGHT)
        scale_x = OUTPUT_WIDTH / offscreen.width
        scale_y = OUTPUT_HEIGHT / offscreen.height
        with offscreen.bind():
            gpu.state.viewport_set(0, 0, offscreen.width, offscreen.height)
            reset_fullscreen_gpu_state()
            shader.bind()
            shader.uniform_sampler("image", full.texture_color)
            shader.uniform_float("scale_x", scale_x)
            shader.uniform_float("scale_y", scale_y)
            shader.uniform_int("sample_x", min(8, max(1, math.ceil(scale_x))))
            shader.uniform_int("sample_y", min(4, max(1, math.ceil(scale_y))))
            batch.draw(shader)
    finally:
        full.free()


def debounce_delay_for(settings, shading_type):
    if shading_type == "MATERIAL":
        return max(0.0, settings.material_debounce)
    if shading_type == "RENDERED":
        return max(0.0, settings.rendered_debounce)
    return 0.0


def screen_plane_distance(region_3d, scene=None):
    """Return the camera-to-screen-plane (zero-parallax) distance for the given 3D viewport."""
    if region_3d.view_perspective == "CAMERA" and scene is not None and scene.camera is not None:
        camera = scene.camera
        camera_world = camera.matrix_world
        focus_object = camera.data.dof.focus_object
        if focus_object is not None:
            forward = (camera_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
            delta = focus_object.matrix_world.translation - camera_world.translation
            projected = delta.dot(forward)
            distance = projected if projected > 0.001 else delta.length
        elif camera.data.dof.use_dof and camera.data.dof.focus_distance > 0.0:
            distance = camera.data.dof.focus_distance
        else:
            distance = (camera_world.translation - region_3d.view_location).length
        return max(float(distance), 0.001)
    return max(float(region_3d.view_distance), 0.001)


def camera_view_pose(region_3d, index, scene=None, device="GO", view_count=None):
    profile = DEVICE_PROFILES[device]
    if view_count is not None:
        profile = {**profile, "count": view_count}
    camera_mode = region_3d.view_perspective == "CAMERA"
    if camera_mode:
        if scene is None or scene.camera is None:
            raise ValueError("Camera View requires an active scene camera")
        camera = scene.camera
        bpy.context.view_layer.update()
        camera_world = camera.matrix_world.copy()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        projection = camera.calc_matrix_camera(
            depsgraph,
            x=OUTPUT_WIDTH,
            y=OUTPUT_HEIGHT,
            scale_x=scene.render.pixel_aspect_x,
            scale_y=scene.render.pixel_aspect_y,
        )
        distance = screen_plane_distance(region_3d, scene)
        right = (camera_world.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
    else:
        distance = screen_plane_distance(region_3d, scene)
        camera_world = region_3d.view_matrix.inverted()
        right = (camera_world.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
        projection = region_3d.window_matrix.copy()
        projection[0][0] = projection[1][1] / FOV_ASPECT

    position = index / (profile["count"] - 1) - 0.5
    if device == "GO":
        offset = distance * math.tan(math.radians(position * profile["cone"]))
    else:
        span = 2.0 * distance / projection[1][1]
        offset = position * span
    camera_world.translation += right * offset
    return camera_world, projection, offset, distance


def view_pose(region_3d, index, scene=None, device="GO", view_count=None):
    camera_world, projection, offset, distance = camera_view_pose(region_3d, index, scene, device, view_count)
    projection[0][2] -= offset * projection[0][0] / distance
    return camera_world.inverted(), projection


def configure_anamorphic_camera(render, camera_data, view_width, view_height, clip_start, clip_end):
    """Configure anamorphic portrait camera and PNG output settings for multi-view rendering."""
    camera_data.sensor_fit = "VERTICAL"
    camera_data.sensor_width = 24.0 * FOV_ASPECT
    camera_data.sensor_height = 24.0
    camera_data.clip_start = clip_start
    camera_data.clip_end = clip_end
    render.resolution_x = view_width
    render.resolution_y = view_height
    render.resolution_percentage = 100
    render.use_border = False
    render.use_crop_to_border = False
    render.pixel_aspect_x = OUTPUT_WIDTH / view_width
    render.pixel_aspect_y = OUTPUT_HEIGHT / view_height
    render.image_settings.media_type = "IMAGE"
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.color_depth = "8"


def apply_camera_pose_shift(camera_object, camera_data, camera_world, projection, offset, distance):
    """Apply off-axis camera world transform, focal length, and stereo shift."""
    camera_object.matrix_world = camera_world
    camera_data.lens = projection[1][1] * camera_data.sensor_height * 0.5
    camera_data.shift_x = (projection[0][2] - offset * projection[0][0] / distance) * FOV_ASPECT / 2.0
    camera_data.shift_y = projection[1][2] / 2.0


def lenticular_depsgraph_update_post(scene, depsgraph):
    if _session is None or scene != bpy.context.scene or _session.rendering:
        return
    _, _, source_space, _ = _session.resolve_source()
    if source_space is None:
        return
    if source_space.shading.type != "RENDERED":
        _session.mark_scene_dirty()
        return
    for update in depsgraph.updates:
        update_id = update.id
        if isinstance(update_id, (bpy.types.Scene, bpy.types.Collection)) or update_id.name.startswith("Lenticular_Live_Temporary_"):
            continue
        if update.is_updated_geometry or update.is_updated_transform:
            _session.mark_scene_dirty()
            return
        if not isinstance(update_id, (bpy.types.Object, bpy.types.Camera, bpy.types.Light)):
            _session.mark_scene_dirty()
            return


class RealtimeSession:
    def __init__(self, source_window, source_area, output_window, output_area):
        self.source_window = source_window
        self.source_area = source_area
        self.source_space = source_area.spaces.active
        self.output_window = output_window
        self.output_area = output_area
        self.output_space = output_area.spaces.active
        self.offscreens = {}
        self.copy_shader = None
        self.interlace_shader = None
        self.downsample_shader = None
        self.copy_batch = None
        self.interlace_batch = None
        self.downsample_batch = None
        self.atlas = None
        self.atlas_upper = None
        self.atlas_third = None
        self.atlas_meta = None
        self.full_atlas = None
        self.full_atlas_upper = None
        self.full_atlas_third = None
        self.full_render_job = None
        self.stereo_render_job = None
        self.full_render_requested = False
        self.stereo_render_requested = False
        self.render_textures = {}
        self.draw_handle = None
        self.timer_callback = None
        self.rendering = False
        self.stopping = False
        self.paused = False
        self.rendered_key = None
        self.refined_key = None
        self.observed_key = None
        self.debounce_started = None
        self.force_render = False
        self.scene_dirty = True
        self.last_view_change = time.monotonic()
        self.idle_ui_key = None
        self.render_ui_at_output_scale = False
        self.device = device_for(bpy.context.scene.lenticular)
        self.profile = profile_for(bpy.context.scene.lenticular)
        self.view_count = self.profile["count"]
        self.view_width, self.view_height = view_dimensions_for(bpy.context.scene.lenticular)
        self.engine_scene = None
        self.engine_camera = None
        self.engine_camera_data = None
        self.engine_temp_dir = None
        self.engine_pending = None
        self.engine_scene_dirty = True
        self.engine_generation = 0
        self.engine_cleanup_callback = None

    def mark_scene_dirty(self):
        self.scene_dirty = True
        self.engine_scene_dirty = True

    def resolve_source(self):
        """Resolve the source 3D viewport safely in case its workspace, window, or area changed."""
        try:
            source_space = self.source_space
            if source_space is None:
                return None, None, None, None
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "VIEW_3D" and area.spaces.active == source_space:
                        region = next((r for r in area.regions if r.type == "WINDOW"), None)
                        if region is not None:
                            self.source_window = window
                            self.source_area = area
                            return window, area, source_space, region
        except ReferenceError:
            pass
        return None, None, None, None

    def resolve_output(self):
        """Resolve the output each Space tick because it may be fullscreened or moved."""
        try:
            output_space = self.output_space
            if output_space is None:
                return None, None, None
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "VIEW_3D" and area.spaces.active == output_space:
                        region = next((r for r in area.regions if r.type == "WINDOW"), None)
                        if region is not None:
                            return window, area, region
        except ReferenceError:
            pass
        return None, None, None

    def start(self):
        self.output_space.show_region_ui = False
        self.output_space.show_region_toolbar = False
        self.output_space.overlay.show_overlays = False
        self.output_space.show_gizmo = False
        self.output_space.shading.type = "SOLID"
        self.draw_handle = self.output_space.draw_handler_add(self.draw, (), "WINDOW", "POST_VIEW")
        self.timer_callback = self.tick
        if lenticular_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(lenticular_depsgraph_update_post)
        bpy.app.timers.register(self.timer_callback, first_interval=0.0)
        properties.show_screen_plane_changed(None, bpy.context)
        _, area, _ = self.resolve_output()
        if area:
            area.tag_redraw()

    def source_window_alive(self):
        try:
            pointer = self.source_window.as_pointer()
        except ReferenceError:
            return False
        return any(window.as_pointer() == pointer for window in bpy.context.window_manager.windows)

    def tick(self):
        if self.stopping:
            return None
        if not self.source_window_alive():
            stop_realtime()
            return None
        _, output_area, _ = self.resolve_output()
        if output_area is None:
            stop_realtime()
            return None
        self.update_views()
        output_area.tag_redraw()
        return 1.0 / 30.0

    def atlas_primary_rows(self):
        rows = self.profile["rows"]
        max_rows = max(1, max_gpu_texture_size() // self.view_height)
        chunk_count = math.ceil(rows / max_rows)
        return math.ceil(rows / chunk_count)

    def current_atlas_meta(self):
        return {
            "device": self.device,
            "columns": self.profile["columns"],
            "rows": self.profile["rows"],
            "count": self.view_count,
            "width": self.view_width,
            "height": self.view_height,
            "chunk_rows": self.atlas_primary_rows(),
        }

    def make_atlases(self):
        max_size = max_gpu_texture_size()
        chunk_rows = self.atlas_primary_rows()
        width = self.profile["columns"] * self.view_width
        if width > max_size:
            raise RuntimeError(
                f"Atlas width {width} exceeds the GPU texture limit ({max_size}); reduce resolution, grid columns, or X compensation"
            )
        if self.view_height > max_size:
            raise RuntimeError(
                f"View height {self.view_height} exceeds the GPU texture limit ({max_size}); reduce resolution or Y compensation"
            )
        atlases = []
        for start in range(0, self.profile["rows"], chunk_rows):
            rows = min(chunk_rows, self.profile["rows"] - start)
            atlases.append(make_offscreen(width, rows * self.view_height))
        if len(atlases) > 3:
            for atlas in atlases:
                atlas.free()
            raise RuntimeError("The current grid and compensation require too many GPU atlas textures; reduce Y compensation")
        atlases.extend([None] * (3 - len(atlases)))
        return tuple(atlases)

    def swap_atlases(self, new_atlases, meta):
        old_atlases = (self.atlas, self.atlas_upper, self.atlas_third)
        self.atlas, self.atlas_upper, self.atlas_third = new_atlases
        self.atlas_meta = meta
        for back in old_atlases:
            if back is not None and back not in new_atlases:
                try:
                    back.free()
                except Exception:
                    pass

    def initialize_gpu(self):
        if self.interlace_shader is not None:
            return
        self.copy_shader, self.interlace_shader = make_shaders()
        self.downsample_shader = make_downsample_shader()
        positions = ((-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0))
        self.copy_batch = batch_for_shader(self.copy_shader, "TRIS", {"pos": positions})
        self.interlace_batch = batch_for_shader(self.interlace_shader, "TRIS", {"pos": positions})
        self.downsample_batch = batch_for_shader(self.downsample_shader, "TRIS", {"pos": positions})

    def view_key(self, region_3d):
        q = region_3d.view_rotation
        loc = region_3d.view_location
        key = [region_3d.view_perspective,
               q.w, q.x, q.y, q.z,
               loc[0], loc[1], loc[2], region_3d.view_distance]
        camera = bpy.context.scene.camera
        if region_3d.view_perspective == "CAMERA" and camera is not None:
            key.extend(value for row in camera.matrix_world for value in row)
            data = camera.data
            key.extend((camera.as_pointer(), data.lens, data.shift_x, data.shift_y,
                        data.sensor_fit, data.sensor_width, data.sensor_height,
                        data.clip_start, data.clip_end, data.dof.focus_distance,
                        data.dof.focus_object.as_pointer() if data.dof.focus_object else 0))
        else:
            key.extend(region_3d.view_matrix.translation)
        return tuple(key)

    def draw(self):
        if self.stopping:
            return
        area = bpy.context.area
        if area is None or area.type != "VIEW_3D" or area.spaces.active != self.output_space:
            return
        _, _, source_space, _ = self.resolve_source()
        if source_space is not None and not has_valid_view(source_space.region_3d, bpy.context.scene):
            self.draw_message("Switch the source viewport to Perspective or Camera View with an active camera")
            return
        if not self.atlas or self.atlas_meta is None:
            return
        self.draw_output(bpy.context.region, bpy.context.scene.lenticular)

    def current_key(self, source_space):
        region_3d = source_space.region_3d
        settings = bpy.context.scene.lenticular
        return (self.view_key(region_3d), stereo_separation_for(settings), self.view_count, source_space.shading.type)

    def cancel_render_jobs(self):
        self.full_render_job = None
        self.stereo_render_job = None
        self.full_render_requested = False
        self.stereo_render_requested = False
        if self.engine_pending is not None:
            self.engine_pending["stale"] = True
        for name in ("full_atlas", "full_atlas_upper", "full_atlas_third"):
            back = getattr(self, name)
            if back is not None:
                try:
                    back.free()
                except Exception:
                    pass
                setattr(self, name, None)
        for index in tuple(self.offscreens):
            self.offscreens.pop(index).free()
        self.render_textures.clear()

    def discard_stale_engine_render(self):
        pending = self.engine_pending
        if pending is None or not pending["stale"] or bpy.app.is_job_running("RENDER"):
            return
        self.engine_pending = None
        if self.engine_scene_dirty:
            self.release_engine_resources()

    def render_job_view(self, source_space, source_region, job):
        position = job["index"]
        index = job["views"][position]
        region_3d = source_space.region_3d
        if job["engine"]:
            if self.engine_pending is None:
                if bpy.app.is_job_running("RENDER"):
                    return False
                self.start_engine_view(index, region_3d, source_space)
                return False
            if bpy.app.is_job_running("RENDER"):
                return False
            pixels = self.finish_engine_view()
            data = gpu.types.Buffer("FLOAT", len(pixels), pixels)
            self.render_textures[index] = gpu.types.GPUTexture(
                (self.view_width, self.view_height), format="RGBA32F", data=data
            )
            texture = self.render_textures[index]
        else:
            self.offscreens[index] = make_offscreen(self.view_width, self.view_height)
            self.render_view(index, region_3d, source_space, source_region)
            texture = self.offscreens[index].texture_color
        if job["full"]:
            targets = (index,)
        elif job.get("refine"):
            targets = job["targets"][position]
        elif position == 0:
            targets = range(self.view_count // 2)
        else:
            targets = range(self.view_count // 2, self.view_count)
        targets = tuple(targets)
        chunk_rows = self.atlas_primary_rows()
        columns = self.profile["columns"]
        backs = (self.full_atlas, self.full_atlas_upper, self.full_atlas_third)
        for chunk_index, back in enumerate(backs):
            if back is None:
                continue
            start = chunk_index * chunk_rows * columns
            end = min(start + chunk_rows * columns, self.view_count)
            chunk_targets = tuple(target for target in targets if start <= target < end)
            if not chunk_targets:
                continue
            with back.bind():
                reset_fullscreen_gpu_state()
                self.copy_shader.bind()
                for target in chunk_targets:
                    column = target % columns
                    row = target // columns - chunk_index * chunk_rows
                    gpu.state.viewport_set(column * self.view_width, row * self.view_height, self.view_width, self.view_height)
                    self.copy_shader.uniform_sampler("image", texture)
                    self.copy_batch.draw(self.copy_shader)
        if index in self.offscreens:
            self.offscreens.pop(index).free()
        self.render_textures.clear()
        job["index"] += 1
        self.scene_dirty = False
        return job["index"] == len(job["views"])

    def update_views(self):
        if self.rendering or self.stopping:
            return
        self.discard_stale_engine_render()
        if (
            self.paused
            and self.full_render_job is None
            and self.stereo_render_job is None
            and not self.full_render_requested
            and not self.stereo_render_requested
        ):
            return
        settings = bpy.context.scene.lenticular
        desired_device = device_for(settings)
        desired_profile = profile_for(settings)
        desired_width, desired_height = view_dimensions_for(settings)
        layout_changed = (
            desired_device != self.device
            or desired_profile["columns"] != self.profile["columns"]
            or desired_profile["rows"] != self.profile["rows"]
            or desired_profile["count"] != self.view_count
        )
        if layout_changed or desired_width != self.view_width or desired_height != self.view_height:
            self.cancel_render_jobs()
            self.engine_scene_dirty = True
            self.device = desired_device
            self.profile = desired_profile
            self.view_count = desired_profile["count"]
            self.view_width = desired_width
            self.view_height = desired_height
            # Keep the previous self.atlas on screen until the new layout finishes rendering to avoid black flashes.
            self.rendered_key = None
            self.refined_key = None
            self.idle_ui_key = None
            self.scene_dirty = True
        source_window, source_area, source_space, resolved_source_region = self.resolve_source()
        if source_space is None:
            return
        if self.engine_pending is None and self.engine_scene is not None:
            if self.engine_scene_dirty or source_space.shading.type != "RENDERED":
                self.release_engine_resources()
        now = time.monotonic()
        key = self.current_key(source_space)
        active_job = self.full_render_job or self.stereo_render_job
        if active_job is not None and (active_job["key"] != key or self.scene_dirty):
            self.cancel_render_jobs()
            self.force_render = True
        if key != self.observed_key or self.scene_dirty:
            self.observed_key = key
            self.refined_key = None
            self.debounce_started = now
            self.last_view_change = now
            self.idle_ui_key = None
            self.force_render = True
            self.scene_dirty = False
        immediate_stereo = self.stereo_render_requested
        if immediate_stereo:
            self.stereo_render_requested = False
            self.full_render_requested = False
            if self.full_render_job is not None or self.stereo_render_job is not None:
                self.cancel_render_jobs()
            self.force_render = True
            self.debounce_started = None
            self.scene_dirty = False
        if self.full_render_requested:
            self.full_render_requested = False
            if self.full_render_job is None:
                if self.stereo_render_job is not None:
                    self.cancel_render_jobs()
                self.full_render_job = {"key": key, "index": 0, "views": tuple(range(self.view_count)), "full": True}
                self.debounce_started = None
                self.scene_dirty = False

        active_job = self.full_render_job or self.stereo_render_job
        if active_job is None:
            # Fast modes refresh immediately, then refine to more views after a short stable interval.
            fast = source_space.shading.type in ("WIREFRAME", "SOLID")
            idle_ui_refresh = (
                fast
                and key == self.rendered_key
                and not self.force_render
                and not self.scene_dirty
                and self.idle_ui_key != key
                and now - self.last_view_change >= 0.2
            )
            refine_due = (
                fast
                and not immediate_stereo
                and key == self.rendered_key
                and self.refined_key != key
                and not self.force_render
                and not self.scene_dirty
                and now - self.last_view_change >= FAST_REFINE_DELAY
            )
            if key == self.rendered_key and not self.force_render and not self.scene_dirty and not idle_ui_refresh and not refine_due:
                return
            if not fast and not immediate_stereo and self.atlas is not None:
                delay = debounce_delay_for(settings, source_space.shading.type)
                if self.debounce_started is None:
                    self.debounce_started = now
                remaining = delay - (now - self.debounce_started)
                if remaining > 0.0:
                    mode_name = "Material Preview" if source_space.shading.type == "MATERIAL" else "Rendered"
                    settings.status = f"Stereo {mode_name} | Refresh in {remaining:.1f} s"
                    return
            self.rendering = True
            try:
                self.initialize_gpu()
                region_3d = source_space.region_3d
                if not has_valid_view(region_3d, bpy.context.scene):
                    settings.status = "Switch to Perspective or Camera View with an active camera"
                    return
                _, output_area, region = self.resolve_output()
                if region is None:
                    return
                source_region = resolved_source_region or region
                separation = stereo_separation_for(settings)
                left = (self.view_count - 1 - separation) // 2
                right = self.view_count - 1 - left
                if refine_due:
                    views = refinement_view_indices(self.view_count)
                    self.stereo_render_job = {
                        "key": key, "index": 0, "views": views, "full": False, "refine": True,
                        "targets": refinement_targets_for(self.view_count, views),
                        "width": region.width, "height": region.height,
                    }
                    self.debounce_started = None
                    self.scene_dirty = False
                elif not fast:
                    self.stereo_render_job = {
                        "key": key, "index": 0, "views": (left, right), "full": False,
                        "width": region.width, "height": region.height,
                    }
                    self.debounce_started = None
                    self.scene_dirty = False
                else:
                    self.render_ui_at_output_scale = idle_ui_refresh
                    desired_meta = self.current_atlas_meta()
                    reuse_current = self.atlas is not None and self.atlas_meta == desired_meta
                    target_atlases = (self.atlas, self.atlas_upper, self.atlas_third) if reuse_current else self.make_atlases()
                    try:
                        with bpy.context.temp_override(window=source_window, area=source_area, region=source_region):
                            for index in tuple(self.offscreens):
                                self.offscreens.pop(index).free()
                            self.render_textures.clear()
                            for index in (left, right):
                                self.offscreens[index] = make_offscreen(self.view_width, self.view_height)
                                self.render_view(index, region_3d, source_space, source_region)
                            self.fill_atlas(left, right, target_atlases, desired_meta["chunk_rows"])
                    except Exception:
                        if not reuse_current:
                            for back in target_atlases:
                                if back is not None:
                                    try:
                                        back.free()
                                    except Exception:
                                        pass
                        raise
                    if not reuse_current:
                        self.swap_atlases(target_atlases, desired_meta)
                    self.rendered_key = key
                    self.idle_ui_key = key if idle_ui_refresh else None
                    self.debounce_started = None
                    self.force_render = False
                    self.scene_dirty = False
                    mode_name = {"WIREFRAME": "Wireframe", "SOLID": "Solid"}.get(source_space.shading.type, source_space.shading.type)
                    settings.status = f"Live Output {mode_name} | {region.width}×{region.height} | 2 views"
                    return
            except Exception as exc:
                bpy.context.scene.lenticular.status = f"Live output error: {exc}"
                print("Lenticular realtime error:", repr(exc), flush=True)
                traceback.print_exc()
                stop_realtime()
            finally:
                self.rendering = False
                self.render_ui_at_output_scale = False

        job = self.full_render_job or self.stereo_render_job
        if job is None:
            return
        self.rendering = True
        try:
            self.initialize_gpu()
            region_3d = source_space.region_3d
            if not has_valid_view(region_3d, bpy.context.scene):
                settings.status = "Switch to Perspective or Camera View with an active camera"
                return
            _, output_area, region = self.resolve_output()
            if region is None:
                return
            source_region = resolved_source_region or region
            job.setdefault("width", region.width)
            job.setdefault("height", region.height)
            job["engine"] = source_space.shading.type == "RENDERED"
            job["mode"] = {"WIREFRAME": "Wireframe", "SOLID": "Solid", "MATERIAL": "Material Preview", "RENDERED": "Rendered"}.get(source_space.shading.type, source_space.shading.type)
            self.render_ui_at_output_scale = not job["full"]
            with bpy.context.temp_override(window=source_window, area=source_area, region=source_region):
                for index in tuple(self.offscreens):
                    self.offscreens.pop(index).free()
                self.render_textures.clear()
                if self.full_atlas is None:
                    self.full_atlas, self.full_atlas_upper, self.full_atlas_third = self.make_atlases()
                completed = self.render_job_view(source_space, source_region, job)
            if completed:
                new_atlases = (self.full_atlas, self.full_atlas_upper, self.full_atlas_third)
                self.full_atlas = None
                self.full_atlas_upper = None
                self.full_atlas_third = None
                self.swap_atlases(new_atlases, self.current_atlas_meta())
                self.rendered_key = job["key"]
                self.idle_ui_key = job["key"]
                if job["full"] or job.get("refine"):
                    self.refined_key = job["key"]
                if job["full"]:
                    self.full_render_job = None
                else:
                    self.stereo_render_job = None
                self.force_render = False
                self.scene_dirty = False
                if job["full"]:
                    bpy.context.scene.lenticular.status = f"Full Frame {job['mode']} | {job['width']}×{job['height']} | {self.view_count} views"
                else:
                    bpy.context.scene.lenticular.status = f"Live Output {job['mode']} | {job['width']}×{job['height']} | {len(job['views'])} views"
            else:
                prefix = "Full Frame" if job["full"] else "Refine" if job.get("refine") else "Stereo"
                bpy.context.scene.lenticular.status = f"{prefix} {job['mode']} | Rendered {job['index']}/{len(job['views'])} views"
        except Exception as exc:
            bpy.context.scene.lenticular.status = f"Live output error: {exc}"
            print("Lenticular realtime error:", repr(exc), flush=True)
            traceback.print_exc()
            stop_realtime()
        finally:
            self.rendering = False
            self.render_ui_at_output_scale = False

    def render_view(self, index, region_3d, view3d_space, region):
        region_3d.update()
        view_matrix, projection = view_pose(region_3d, index, bpy.context.scene, self.device, self.view_count)
        offscreen = self.offscreens[index]
        bpy.context.view_layer.update()
        try:
            draw_view3d_at_output_scale(
                offscreen,
                bpy.context.scene,
                bpy.context.view_layer,
                view3d_space,
                region,
                view_matrix,
                projection,
                self.downsample_shader,
                self.downsample_batch,
                self.render_ui_at_output_scale,
            )
        finally:
            region_3d.update()

    def start_engine_view(self, index, region_3d, source_space):
        if self.engine_scene_dirty:
            self.release_engine_resources()
        source_scene = bpy.context.scene
        if self.engine_scene is None:
            self.engine_scene = source_scene.copy()
            self.engine_camera_data = bpy.data.cameras.new("Lenticular_Live_Temporary_Camera")
            self.engine_camera = bpy.data.objects.new("Lenticular_Live_Temporary_Camera", self.engine_camera_data)
            self.engine_scene.collection.objects.link(self.engine_camera)
            self.engine_scene.camera = self.engine_camera
            self.engine_temp_dir = tempfile.TemporaryDirectory(prefix="lenticular_live_")
        scene = self.engine_scene
        render = scene.render
        camera_data = self.engine_camera_data
        camera_world, projection, offset, distance = camera_view_pose(region_3d, index, source_scene, self.device, self.view_count)
        # Match render_worker.py: VERTICAL plus pixel aspect ratio exactly reconstructs the 9:16 viewport projection.
        source_camera = source_scene.camera if region_3d.view_perspective == "CAMERA" else None
        configure_anamorphic_camera(
            render,
            camera_data,
            self.view_width,
            self.view_height,
            source_camera.data.clip_start if source_camera else source_space.clip_start,
            source_camera.data.clip_end if source_camera else source_space.clip_end,
        )
        apply_camera_pose_shift(self.engine_camera, camera_data, camera_world, projection, offset, distance)
        render.use_persistent_data = True
        self.engine_generation += 1
        render.filepath = os.path.join(self.engine_temp_dir.name, f"view_{self.engine_generation}_{index}.png")
        active_layer = getattr(bpy.context, "view_layer", None)
        target_layer = scene.view_layers.get(active_layer.name) if active_layer else None
        (target_layer or scene.view_layers[0]).update()
        self.engine_pending = {"index": index, "path": render.filepath, "stale": False}
        display_type = bpy.context.preferences.view.render_display_type
        try:
            bpy.context.preferences.view.render_display_type = "NONE"
            result = bpy.ops.render.render("INVOKE_DEFAULT", write_still=True, scene=scene.name)
        except Exception:
            self.engine_pending = None
            raise
        finally:
            bpy.context.preferences.view.render_display_type = display_type
        if result not in ({"RUNNING_MODAL"}, {"FINISHED"}):
            self.engine_pending = None
            raise RuntimeError(f"Failed to start asynchronous render: {result}")
        self.engine_scene_dirty = False

    def finish_engine_view(self):
        pending = self.engine_pending
        self.engine_pending = None
        if pending is None:
            raise RuntimeError("No asynchronous render is pending")
        if not os.path.isfile(pending["path"]):
            raise RuntimeError("The asynchronous render was cancelled or did not generate an image")
        image = bpy.data.images.load(pending["path"], check_existing=False)
        try:
            pixels = np.empty(self.view_width * self.view_height * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            return np.ascontiguousarray(np.clip(pixels, 0.0, 1.0), dtype=np.float32)
        finally:
            bpy.data.images.remove(image)

    def release_engine_resources(self):
        if self.engine_pending is not None and bpy.app.is_job_running("RENDER"):
            return False
        self.engine_pending = None
        if self.engine_camera is not None:
            try:
                bpy.data.objects.remove(self.engine_camera, do_unlink=True)
            except ReferenceError:
                pass
        if self.engine_camera_data is not None:
            try:
                if self.engine_camera_data.users == 0:
                    bpy.data.cameras.remove(self.engine_camera_data)
            except ReferenceError:
                pass
        if self.engine_scene is not None:
            try:
                bpy.data.scenes.remove(self.engine_scene)
            except ReferenceError:
                pass
        if self.engine_temp_dir is not None:
            self.engine_temp_dir.cleanup()
        self.engine_scene = None
        self.engine_camera = None
        self.engine_camera_data = None
        self.engine_temp_dir = None
        self.engine_scene_dirty = True
        return True

    def cleanup_stopped_engine(self):
        if bpy.app.is_job_running("RENDER"):
            return 0.1
        self.release_engine_resources()
        self.engine_cleanup_callback = None
        return None

    def fill_atlas(self, left, right, atlases=None, chunk_rows=None):
        chunk_rows = chunk_rows if chunk_rows is not None else self.atlas_primary_rows()
        atlases = atlases if atlases is not None else (self.atlas, self.atlas_upper, self.atlas_third)
        columns = self.profile["columns"]
        chunks = []
        for chunk_index, back in enumerate(atlases):
            if back is None:
                continue
            start = chunk_index * chunk_rows * columns
            end = min(start + chunk_rows * columns, self.view_count)
            chunks.append((back, start, end, chunk_index * chunk_rows))
        for back, start, end, row_base in chunks:
            with back.bind():
                reset_fullscreen_gpu_state()
                self.copy_shader.bind()
                for view in range(start, end):
                    index = left if view < self.view_count // 2 else right
                    texture = self.render_textures.get(index)
                    if texture is None:
                        texture = self.offscreens[index].texture_color
                    column = view % columns
                    row = view // columns - row_base
                    gpu.state.viewport_set(
                        column * self.view_width,
                        row * self.view_height,
                        self.view_width,
                        self.view_height,
                    )
                    self.copy_shader.uniform_sampler("image", texture)
                    self.copy_batch.draw(self.copy_shader)

    def draw_output(self, region, settings):
        meta = self.atlas_meta
        prev_viewport = gpu.state.viewport_get()
        prev_depth = gpu.state.depth_test_get()
        prev_blend = gpu.state.blend_get()
        try:
            gpu.state.viewport_set(0, 0, region.width, region.height)
            reset_fullscreen_gpu_state()
            self.interlace_shader.bind()
            self.interlace_shader.uniform_float("x0", settings.x0)
            self.interlace_shader.uniform_float("interval", settings.interval)
            self.interlace_shader.uniform_float("slope", settings.slope)
            self.interlace_shader.uniform_int("device_kind", 1 if meta["device"] == "GO" else 0)
            self.interlace_shader.uniform_int("view_interpolation", 1 if settings.view_interpolation else 0)
            self.interlace_shader.uniform_int("view_count", meta["count"])
            self.interlace_shader.uniform_int("view_columns", meta["columns"])
            self.interlace_shader.uniform_int("view_rows", meta["rows"])
            self.interlace_shader.uniform_int("atlas_chunk_rows", meta["chunk_rows"])
            self.interlace_shader.uniform_sampler("image", self.atlas.texture_color)
            upper = self.atlas_upper if self.atlas_upper is not None else self.atlas
            third = self.atlas_third if self.atlas_third is not None else self.atlas
            self.interlace_shader.uniform_sampler("image_upper", upper.texture_color)
            self.interlace_shader.uniform_sampler("image_third", third.texture_color)
            self.interlace_batch.draw(self.interlace_shader)
        finally:
            gpu.state.viewport_set(*prev_viewport)
            gpu.state.depth_test_set(prev_depth)
            gpu.state.blend_set(prev_blend)

    def draw_message(self, message):
        font_id = 0
        blf.position(font_id, 32, 48, 0)
        blf.size(font_id, 20)
        blf.color(font_id, 1.0, 0.65, 0.2, 1.0)
        blf.draw(font_id, message)

    def stop(self):
        if self.stopping:
            return
        self.stopping = True
        if self.timer_callback and bpy.app.timers.is_registered(self.timer_callback):
            bpy.app.timers.unregister(self.timer_callback)
        if lenticular_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(lenticular_depsgraph_update_post)
        if self.draw_handle:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self.draw_handle, "WINDOW")
            except Exception:
                pass
            self.draw_handle = None
        self.cancel_render_jobs()
        if self.engine_pending is not None and bpy.app.is_job_running("RENDER"):
            self.engine_cleanup_callback = self.cleanup_stopped_engine
            bpy.app.timers.register(self.engine_cleanup_callback, first_interval=0.1)
        else:
            self.release_engine_resources()
        for offscreen in self.offscreens.values():
            try:
                offscreen.free()
            except Exception:
                pass
        self.offscreens.clear()
        self.render_textures.clear()
        for name in ("atlas", "atlas_upper", "atlas_third"):
            back = getattr(self, name)
            if back is not None:
                try:
                    back.free()
                except Exception:
                    pass
            setattr(self, name, None)
        self.atlas_meta = None
        try:
            properties.show_screen_plane_changed(None, bpy.context)
            windows = bpy.context.window_manager.windows
            output_window, area, _ = self.resolve_output()
            if output_window is None and self.output_window is not None:
                try:
                    target_ptr = self.output_window.as_pointer()
                    output_window = next((w for w in windows if w.as_pointer() == target_ptr), None)
                except ReferenceError:
                    output_window = None
            source_ptr = None
            if self.source_window is not None:
                try:
                    source_ptr = self.source_window.as_pointer()
                except ReferenceError:
                    source_ptr = None
            closed = False
            if (
                output_window is not None
                and len(windows) > 1
                and output_window.as_pointer() != source_ptr
            ):
                try:
                    with bpy.context.temp_override(window=output_window):
                        bpy.ops.wm.window_close()
                    closed = True
                except Exception:
                    closed = False
            if not closed and area:
                area.tag_redraw()
        except Exception:
            pass
        finally:
            self.output_window = None
            self.output_area = None
            self.output_space = None


def stop_realtime():
    global _session
    if _session:
        session = _session
        _session = None
        session.stop()
    for scene in bpy.data.scenes:
        if hasattr(scene, "lenticular"):
            scene.lenticular.status = "Not started"


@persistent
def stop_realtime_on_load(_filepath=""):
    stop_realtime()


class Lenticular_OT_start_realtime(bpy.types.Operator):
    bl_idname = "lenticular.start_realtime"
    bl_label = "Start Live Output"
    bl_description = "Create a separate render window. Move it to the target device and enter full screen"

    def execute(self, context):
        global _session
        if _session:
            self.report({"INFO"}, "Live output is already running")
            return {"FINISHED"}
        settings = context.scene.lenticular
        if not properties.is_scene_calibrated(settings):
            if settings.device == "GO" and properties.try_auto_import_go_calibration(context):
                properties.update_device_calibration(settings, "GO", context)
        if not properties.is_scene_calibrated(settings):
            device_name = "Looking Glass Go" if settings.device == "GO" else "OpenStage AI C1"
            self.report({"WARNING"}, f"Configure optical calibration for {device_name} first")
            bpy.ops.lenticular.configure_calibration("INVOKE_DEFAULT")
            return {"CANCELLED"}
        if (
            context.area is None
            or context.area.type != "VIEW_3D"
            or context.space_data is None
            or context.space_data.region_3d.view_perspective not in {"PERSP", "CAMERA"}
            or (context.space_data.region_3d.view_perspective == "CAMERA" and context.scene.camera is None)
        ):
            self.report({"ERROR"}, "Start in Perspective or Camera View with an active camera")
            return {"CANCELLED"}

        source_window = context.window
        source_area = context.area
        before = {window.as_pointer() for window in context.window_manager.windows}
        bpy.ops.wm.window_new()
        output_window = next(
            (window for window in context.window_manager.windows if window.as_pointer() not in before),
            None,
        )
        if output_window is None:
            self.report({"ERROR"}, "Could not create the output window")
            return {"CANCELLED"}
        output_area = next((area for area in output_window.screen.areas if area.type == "VIEW_3D"), None)
        if output_area is None:
            self.report({"ERROR"}, "The new window has no 3D viewport")
            return {"CANCELLED"}

        _session = RealtimeSession(source_window, source_area, output_window, output_area)
        _session.start()
        context.scene.lenticular.status = "Started"
        self.report({"INFO"}, "Move the render window to the target device and enter full screen")
        return {"FINISHED"}


class Lenticular_OT_render_stereo_once(bpy.types.Operator):
    bl_idname = "lenticular.render_stereo_once"
    bl_label = "Render Current Stereo Frame"
    bl_description = "Ignore the delay and immediately render the current two-view stereo pair, cancelling any unfinished task"

    def execute(self, context):
        if _session is None:
            self.report({"WARNING"}, "Start live output first")
            return {"CANCELLED"}
        _session.stereo_render_requested = True
        context.scene.lenticular.status = "Rendering current stereo frame…"
        return {"FINISHED"}


class Lenticular_OT_render_once(bpy.types.Operator):
    bl_idname = "lenticular.render_once"
    bl_label = "Render Full Current Frame"
    bl_description = "Render all device views in stages. Scene or view changes automatically cancel unfinished tasks"

    def execute(self, context):
        global _session
        if _session:
            _session.full_render_requested = True
            context.scene.lenticular.status = "Rendering full current frame…"
        else:
            self.report({"WARNING"}, "Start live output first")
        return {"FINISHED"}


class Lenticular_OT_pause_realtime(bpy.types.Operator):
    bl_idname = "lenticular.pause_realtime"
    bl_label = "Pause/Resume Live Output"
    bl_description = "Pausing keeps the output window and current image but stops automatic rendering. Manual stereo and full-frame rendering remain available"

    def execute(self, context):
        if _session is None:
            self.report({"WARNING"}, "Start live output first")
            return {"CANCELLED"}
        _session.paused = not _session.paused
        if _session.paused:
            _session.cancel_render_jobs()
            context.scene.lenticular.status = "Live rendering paused | Current image retained"
        else:
            _session.mark_scene_dirty()
            _session.force_render = True
            context.scene.lenticular.status = "Live rendering resumed; waiting to refresh"
        return {"FINISHED"}


class Lenticular_OT_stop_realtime(bpy.types.Operator):
    bl_idname = "lenticular.stop_realtime"
    bl_label = "Stop Live Output"

    def execute(self, context):
        stop_realtime()
        return {"FINISHED"}


def draw_screen_plane_overlay():
    """Draw a translucent light-blue glass plane at the zero-parallax distance in the editing viewport."""
    if _offscreen_draw_depth > 0:
        return
    context = bpy.context
    scene = getattr(context, "scene", None)
    space = getattr(context, "space_data", None)
    region_3d = getattr(context, "region_data", None) or getattr(space, "region_3d", None)
    if scene is None or space is None or getattr(space, "type", None) != "VIEW_3D" or region_3d is None:
        return
    if _session is None or _session.stopping or space == _session.output_space or _session.rendering:
        return
    settings = getattr(scene, "lenticular", None)
    if settings is None or not getattr(settings, "show_screen_plane", False):
        return
    if not has_valid_view(region_3d, scene):
        return

    distance = screen_plane_distance(region_3d, scene)
    if region_3d.view_perspective == "CAMERA" and scene.camera is not None:
        camera_world = scene.camera.matrix_world
    else:
        camera_world = region_3d.view_matrix.inverted()

    origin = camera_world.translation
    rot = camera_world.to_3x3()
    right = (rot @ Vector((1.0, 0.0, 0.0))).normalized()
    up = (rot @ Vector((0.0, 1.0, 0.0))).normalized()
    forward = (rot @ Vector((0.0, 0.0, -1.0))).normalized()
    center = origin + forward * distance

    proj = region_3d.window_matrix
    px = max(abs(float(proj[0][0])), 1e-4)
    py = max(abs(float(proj[1][1])), 1e-4)
    sx = abs(float(proj[0][2]))
    sy = abs(float(proj[1][2]))
    half_w = distance * (1.5 + sx) / px
    half_h = distance * (1.5 + sy) / py

    p0 = tuple(center - right * half_w - up * half_h)
    p1 = tuple(center + right * half_w - up * half_h)
    p2 = tuple(center + right * half_w + up * half_h)
    p3 = tuple(center - right * half_w + up * half_h)

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "TRIS", {"pos": (p0, p1, p2, p0, p2, p3)})

    prev_depth = gpu.state.depth_test_get()
    prev_depth_mask = gpu.state.depth_mask_get()
    prev_blend = gpu.state.blend_get()
    try:
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set("ALPHA")
        gpu.state.face_culling_set("NONE")
        shader.bind()
        shader.uniform_float("color", (0.25, 0.65, 1.0, 0.22))
        batch.draw(shader)
    finally:
        gpu.state.depth_test_set(prev_depth)
        gpu.state.depth_mask_set(prev_depth_mask)
        gpu.state.blend_set(prev_blend)
        gpu.state.face_culling_set("NONE")


classes = (
    Lenticular_OT_start_realtime,
    Lenticular_OT_stop_realtime,
    Lenticular_OT_render_stereo_once,
    Lenticular_OT_render_once,
    Lenticular_OT_pause_realtime,
)


def register():
    global _screen_plane_draw_handle
    for cls in classes:
        bpy.utils.register_class(cls)
    if stop_realtime_on_load not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(stop_realtime_on_load)
    if _screen_plane_draw_handle is None:
        _screen_plane_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_screen_plane_overlay, (), "WINDOW", "POST_VIEW"
        )


def unregister():
    global _screen_plane_draw_handle
    if _screen_plane_draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_screen_plane_draw_handle, "WINDOW")
        except Exception:
            pass
        _screen_plane_draw_handle = None
    if stop_realtime_on_load in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(stop_realtime_on_load)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
