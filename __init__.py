bl_info = {
    "name": "Lenticular",
    "author": "Megabits",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "3D View > Sidebar > Lenticular",
    "description": "Live lenticular 3D viewport and multiview export",
    "doc_url": "https://github.com/megabitsenmzq/OpenGlass-Lenticular",
    "tracker_url": "https://github.com/megabitsenmzq/OpenGlass-Lenticular/issues",
    "category": "3D View",
}

from . import properties, calibration, realtime, cycles_render, panel


def register():
    properties.register()
    calibration.register()
    realtime.register()
    cycles_render.register()
    panel.register()


def unregister():
    realtime.stop_realtime()
    panel.unregister()
    cycles_render.unregister()
    realtime.unregister()
    calibration.unregister()
    properties.unregister()


if __name__ == "__main__":
    register()
