from __future__ import annotations

import numpy as np


def range_center(value_range: tuple[float, float]) -> float:
    return (value_range[0] + value_range[1]) * 0.5


def range_span(value_range: tuple[float, float]) -> float:
    return max(value_range[1] - value_range[0], 1e-6)


def axis_reference(value_range: tuple[float, float], preferred: float = 0.0) -> float:
    return float(np.clip(preferred, value_range[0], value_range[1]))


def tick_values(value_range: tuple[float, float]) -> np.ndarray:
    span = range_span(value_range)
    if span <= 2.0:
        step = 0.5
    elif span <= 5.0:
        step = 1.0
    else:
        step = 2.0

    start = np.ceil(value_range[0] / step) * step
    stop = np.floor(value_range[1] / step) * step
    values = np.arange(start, stop + step * 0.5, step)
    return values[(values >= value_range[0] - 1e-9) & (values <= value_range[1] + 1e-9)]


def add_gl_line(view, gl, points, color, width: float = 1.0, mode: str = 'lines'):
    item = gl.GLLinePlotItem(
        pos=np.asarray(points, dtype=np.float32),
        color=color,
        width=width,
        mode=mode,
        antialias=True,
    )
    view.addItem(item)
    return item


def add_gl_text(view, gl, qt_gui, pos, text: str, color, size: int = 10):
    font = qt_gui.QFont('Helvetica', size)
    item = gl.GLTextItem(
        pos=np.asarray(pos, dtype=np.float32),
        text=text,
        color=color,
        font=font,
    )
    view.addItem(item)
    return item


def add_floor_plane(view, gl, display_ranges) -> None:
    (x_min, x_max), (y_min, y_max), (z_min, _z_max) = display_ranges
    x_span = range_span((x_min, x_max))
    y_span = range_span((y_min, y_max))
    x_center = range_center((x_min, x_max))
    y_center = range_center((y_min, y_max))

    grid = gl.GLGridItem()
    grid.setSize(x=x_span, y=y_span)
    grid.setSpacing(x=max(x_span / 8.0, 0.25), y=max(y_span / 8.0, 0.25))
    grid.translate(x_center, y_center, z_min)
    view.addItem(grid)

    border_color = (0.8, 0.8, 0.8, 0.28)
    border = [
        (x_min, y_min, z_min), (x_max, y_min, z_min),
        (x_max, y_min, z_min), (x_max, y_max, z_min),
        (x_max, y_max, z_min), (x_min, y_max, z_min),
        (x_min, y_max, z_min), (x_min, y_min, z_min),
    ]
    add_gl_line(view, gl, border, border_color, width=1.2)


def init_plot(display_ranges, title: str = 'Realtime points 3D (pyqtgraph OpenGL)'):
    try:
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        from pyqtgraph.Qt import QtGui
        from pyqtgraph.Qt import QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            '需要先安裝 pyqtgraph 3D 依賴：python -m pip install pyqtgraph PyQt5 PyOpenGL'
        ) from exc

    app = QtWidgets.QApplication.instance() or pg.mkQApp('MARS UART 3D Point Cloud')
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = display_ranges
    x_center = range_center((x_min, x_max))
    y_center = range_center((y_min, y_max))
    z_center = range_center((z_min, z_max))
    x_span = range_span((x_min, x_max))
    y_span = range_span((y_min, y_max))
    z_span = range_span((z_min, z_max))

    view = gl.GLViewWidget()
    view.setWindowTitle(title)
    view.setBackgroundColor((18, 20, 24))
    view.setCameraPosition(distance=max(x_span, y_span, z_span) * 1.8, elevation=20, azimuth=-62)
    view.opts['center'].setX(x_center)
    view.opts['center'].setY(y_center)
    view.opts['center'].setZ(z_center)

    add_floor_plane(view, gl, display_ranges)

    scatter = gl.GLScatterPlotItem(
        pos=np.empty((0, 3), dtype=np.float32),
        color=np.empty((0, 4), dtype=np.float32),
        size=8,
        pxMode=True,
    )
    view.addItem(scatter)
    view.show()
    app.processEvents()
    return {'app': app, 'view': view, 'scatter': scatter}


def clip_display_points(points: np.ndarray, display_ranges) -> np.ndarray:
    if points is None or points.size == 0:
        return np.zeros((0, 5), dtype=np.float64)

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = display_ranges
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return points[mask]


def y_to_rgba(y_values: np.ndarray, y_range: tuple[float, float]) -> np.ndarray:
    if y_values.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    y_min, _y_max = y_range
    t = np.clip((y_values - y_min) / range_span(y_range), 0.0, 1.0).astype(np.float32)
    return np.column_stack((
        0.2 + 0.7 * t,
        0.9 - 0.5 * t,
        1.0 - 0.8 * t,
        np.ones_like(t),
    )).astype(np.float32)


def update_plot_points(points: np.ndarray, plot, display_ranges) -> None:
    points = clip_display_points(points, display_ranges)
    scatter = plot['scatter']
    if points.size == 0:
        scatter.setData(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
        )
    else:
        scatter.setData(
            pos=points[:, 0:3].astype(np.float32, copy=False),
            color=y_to_rgba(points[:, 1], display_ranges[1]),
            size=8,
            pxMode=True,
        )
    plot['app'].processEvents()


def process_plot_events(plot) -> None:
    plot['app'].processEvents()
