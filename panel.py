import bpy

from . import properties, realtime, cycles_render


class LENTICULAR_PT_viewport(bpy.types.Panel):
    bl_label = "Lenticular"
    bl_idname = "LENTICULAR_PT_viewport"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Lenticular"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.lenticular
        export_active = cycles_render.is_cycles_active()
        calibrated = properties.is_scene_calibrated(settings)
        device_name = "Looking Glass Go" if settings.device == "GO" else "OpenStage AI C1"
        device_row = layout.row()
        device_row.enabled = realtime._session is None and not export_active
        device_row.prop(settings, "device", text="")
        if not calibrated:
            cal_box = layout.box()
            cal_box.alert = True
            cal_box.label(text=f"Calibration required for {device_name}", icon="ERROR")
            op = cal_box.operator("lenticular.import_calibration", text="Import Calibration JSON…", icon="FILE_FOLDER")
            op.target_device = settings.device
            if settings.device == "GO":
                cal_box.operator("lenticular.autodetect_calibration", text="Auto-Detect visual.json", icon="VIEWZOOM")
            cal_box.operator("lenticular.configure_calibration", text="Enter Manually…", icon="PREFERENCES")
        if realtime._session:
            layout.operator("lenticular.stop_realtime", text="Stop Live Output", icon="EVENT_MEDIASTOP")
        else:
            layout.operator("lenticular.start_realtime", text="Start Live Output", icon="PLAY")
        if realtime._session:
            layout.operator("lenticular.render_stereo_once", icon="RENDER_STILL")
            layout.operator("lenticular.render_once", text=f"Render Full Frame ({realtime._session.view_count} Views)", icon="RENDER_STILL")
            pause_text = "Resume Live Output" if realtime._session.paused else "Pause Live Output"
            pause_icon = "PLAY" if realtime._session.paused else "PAUSE"
            layout.operator("lenticular.pause_realtime", text=pause_text, icon=pause_icon)
            layout.prop(settings, "show_screen_plane")
            layout.label(text=f"Move the render window to {device_name} and enter full screen")
        shading_type = context.space_data.shading.type if context.space_data else None
        if shading_type == "MATERIAL":
            layout.prop(settings, "material_debounce", slider=True)
        elif shading_type == "RENDERED":
            layout.prop(settings, "rendered_debounce", slider=True)
        params_col = layout.column()
        params_col.enabled = not export_active
        params_col.label(text="Lenticular Parameters:")
        cal_row = params_col.row(align=True)
        op = cal_row.operator("lenticular.import_calibration", text="Import JSON", icon="FILE_FOLDER")
        op.target_device = settings.device
        if settings.device == "GO":
            cal_row.operator("lenticular.autodetect_calibration", text="Auto-Detect", icon="VIEWZOOM")
        params_col.prop(settings, "x0", text="Center" if settings.device == "GO" else "X0")
        params_col.prop(settings, "interval", text="Pitch" if settings.device == "GO" else "Interval")
        params_col.prop(settings, "slope")
        params_col.prop(settings, "go_grid")
        params_col.prop(settings, "view_interpolation")
        view_width, view_height = realtime.view_dimensions_for(settings)
        params_col.label(text=f"Resolution: {view_width} × {view_height} (Native {realtime.OUTPUT_WIDTH} × {realtime.OUTPUT_HEIGHT})")
        row = params_col.row()
        row.prop(settings, "resolution_mode", expand=True)
        row = params_col.row()
        row.prop(settings, "compensation_x", text="X Compensation")
        row.prop(settings, "compensation_y", text="Y Compensation")
        layout.label(text=settings.status, icon="INFO")


class LENTICULAR_PT_render_output(bpy.types.Panel):
    bl_label = "Render Output"
    bl_idname = "LENTICULAR_PT_render_output"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Lenticular"
    bl_parent_id = "LENTICULAR_PT_viewport"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.lenticular
        view_count = realtime.profile_for(settings)["count"]
        layout.label(text="Still Images:")
        column = layout.column()
        column.enabled = not cycles_render.is_cycles_active()
        column.operator("lenticular.offscreen_export", text=f"Render {view_count} Viewport Views", icon="VIEW3D")
        column.operator("lenticular.final_export", text=f"Render {view_count} Final Views", icon="RENDER_STILL")

        layout.separator()
        layout.label(text="Animation:")
        column = layout.column()
        column.enabled = not cycles_render.is_cycles_active()
        column.operator("lenticular.offscreen_animation_export", text=f"Render Viewport Animation ({view_count} views/frame)", icon="VIEW3D")
        column.operator("lenticular.animation_export", text=f"Render Final Animation ({view_count} views/frame)", icon="RENDER_ANIMATION")

        if cycles_render.is_cycles_active():
            total_views = cycles_render.active_total_views()
            if total_views > 0 and hasattr(layout, "progress"):
                progress = min(max(settings.cycles_progress / total_views, 0.0), 1.0)
                layout.progress(
                    factor=progress,
                    type="BAR",
                    text=f"{settings.cycles_progress}/{total_views} views",
                )
            layout.label(text="Press Esc to cancel rendering", icon="INFO")


classes = (LENTICULAR_PT_viewport, LENTICULAR_PT_render_output)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
