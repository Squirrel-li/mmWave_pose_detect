"""Live point-cloud plus experimental point-driven skeleton view.

This experiment keeps all new code under ansen_test. It reuses the existing
UART parser but does not use the MARS model; instead, it starts from a fixed
19-joint MARS-style skeleton and lets nearby radar points pull each joint.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable
from typing import Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from radar_uart import RadarUARTCapture
from radar_uart import RadarUARTSettings
from util.radar_config import cfg_get
from util.radar_config import cfg_range
from util.radar_config import load_radar_config
from util.radar_config import resolve_cfg_path


EXPERIMENT_VERSION = 'v2-front-depth-heatmap'
RADAR_CONFIG = load_radar_config()

DEFAULT_CONFIG_PORT = str(cfg_get(RADAR_CONFIG, 'radar', 'config_port', default='COM19'))
DEFAULT_DATA_PORT = str(cfg_get(RADAR_CONFIG, 'radar', 'data_port', default='COM20'))
DEFAULT_CFG_FILE = str(cfg_get(RADAR_CONFIG, 'radar', 'cfg_file', default='IWRL6844_4T4R_record_high_accuracy.cfg'))
DEFAULT_CFG_PATH = resolve_cfg_path(DEFAULT_CFG_FILE)
DEFAULT_BAUDRATE_CFG = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_cfg', default=115200))
DEFAULT_BAUDRATE_DATA = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_data', default=1250000))

PC_X, PC_Y, PC_Z = cfg_range(RADAR_CONFIG, 'point_cloud')
LBL_X, LBL_Y, LBL_Z = cfg_range(RADAR_CONFIG, 'label')

INTENSITY_MODE = str(cfg_get(RADAR_CONFIG, 'point_output', 'intensity_mode', default='snr_db'))
SNR_NORM_MEAN = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_mean', default=20.0))
SNR_NORM_STD = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_std', default=10.0))
SIDE_INFO_DB_LIMIT = float(cfg_get(RADAR_CONFIG, 'point_output', 'side_info_db_limit', default=100.0))

# Same 19-joint order and edges used by scripts/MARS_UART_realtime_predict.py.
JOINT_NAMES = [
    'SpineBase', 'SpineMid', 'Neck', 'Head',
    'ShoulderLeft', 'ElbowLeft', 'WristLeft',
    'ShoulderRight', 'ElbowRight', 'WristRight',
    'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft',
    'HipRight', 'KneeRight', 'AnkleRight', 'FootRight',
    'SpineShoulder',
]

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (1, 4), (4, 5), (5, 6),
    (1, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16), (16, 17),
    (18, 4), (18, 7), (1, 18),
]


@dataclass(frozen=True)
class ExperimentSettings:
    cfg_path: str
    port_cfg: str
    port_data: str
    baudrate_cfg: int
    baudrate_data: int
    frames: int
    send_config: bool
    demo: bool
    update_hz: float
    bins_x: int
    bins_z: int
    heatmap_decay: float
    influence_radius: float
    pull_gain: float
    smoothing: float
    min_intensity: float
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    intensity_mode: str
    snr_norm_mean: float
    snr_norm_std: float
    side_info_db_limit: float


def range_center(value_range: Tuple[float, float]) -> float:
    return (float(value_range[0]) + float(value_range[1])) * 0.5


def range_span(value_range: Tuple[float, float]) -> float:
    return max(float(value_range[1]) - float(value_range[0]), 1e-6)


def clip_points(points: np.ndarray, settings: ExperimentSettings) -> np.ndarray:
    if points is None or points.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float64)

    mask = np.any(points != 0, axis=1)
    mask &= (points[:, 0] >= settings.x_range[0]) & (points[:, 0] <= settings.x_range[1])
    mask &= (points[:, 1] >= settings.y_range[0]) & (points[:, 1] <= settings.y_range[1])
    mask &= (points[:, 2] >= settings.z_range[0]) & (points[:, 2] <= settings.z_range[1])
    if settings.min_intensity > 0:
        mask &= points[:, 4] >= settings.min_intensity
    return points[mask]


def make_base_skeleton(settings: ExperimentSettings) -> np.ndarray:
    center_x = range_center(LBL_X)
    center_y = min(max(range_center(settings.y_range), settings.y_range[0] + 0.6), settings.y_range[1] - 0.6)
    floor_z = max(settings.z_range[0] + 0.05, -0.85)
    spine_base_z = floor_z + 0.75

    joints = np.array([
        [0.00, 0.00, 0.00],   # SpineBase
        [0.00, 0.00, 0.36],   # SpineMid
        [0.00, 0.00, 0.78],   # Neck
        [0.00, 0.00, 1.02],   # Head
        [-0.24, 0.02, 0.70],  # ShoulderLeft
        [-0.48, 0.00, 0.45],  # ElbowLeft
        [-0.62, 0.02, 0.22],  # WristLeft
        [0.24, 0.02, 0.70],   # ShoulderRight
        [0.48, 0.00, 0.45],   # ElbowRight
        [0.62, 0.02, 0.22],   # WristRight
        [-0.16, 0.00, -0.08], # HipLeft
        [-0.22, 0.00, -0.50], # KneeLeft
        [-0.22, 0.03, -0.92], # AnkleLeft
        [-0.22, 0.18, -1.02], # FootLeft
        [0.16, 0.00, -0.08],  # HipRight
        [0.22, 0.00, -0.50],  # KneeRight
        [0.22, 0.03, -0.92],  # AnkleRight
        [0.22, 0.18, -1.02],  # FootRight
        [0.00, 0.02, 0.66],   # SpineShoulder
    ], dtype=np.float64)
    joints[:, 0] += center_x
    joints[:, 1] += center_y
    joints[:, 2] += spine_base_z
    return joints


def dynamic_skeleton(base_joints: np.ndarray, points: np.ndarray, settings: ExperimentSettings) -> np.ndarray:
    if points.shape[0] == 0:
        return base_joints.copy()

    coords = points[:, 0:3]
    intensity = np.clip(points[:, 4], 0.0, None)
    if np.max(intensity) > 0:
        intensity = intensity / np.max(intensity)
    else:
        intensity = np.ones((points.shape[0],), dtype=np.float64)

    moved = base_joints.copy()
    for joint_index, joint in enumerate(base_joints):
        deltas = coords - joint
        distances = np.linalg.norm(deltas, axis=1)
        mask = distances < settings.influence_radius
        if not np.any(mask):
            continue

        local_dist = distances[mask]
        distance_weight = 1.0 - np.clip(local_dist / settings.influence_radius, 0.0, 1.0)
        weights = distance_weight * (0.35 + 0.65 * intensity[mask])
        total = float(np.sum(weights))
        if total <= 1e-6:
            continue

        target = np.sum(coords[mask] * weights[:, None], axis=0) / total
        offset = (target - joint) * settings.pull_gain
        moved[joint_index] = joint + offset

    return moved


def blend_joints(previous: np.ndarray, current: np.ndarray, smoothing: float) -> np.ndarray:
    return previous * smoothing + current * (1.0 - smoothing)


def y_to_rgba(y_values: np.ndarray, y_range: Tuple[float, float]) -> np.ndarray:
    if y_values.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    y_min, y_max = y_range
    t = np.clip((y_values - y_min) / range_span(y_range), 0.0, 1.0).astype(np.float32)
    return np.column_stack((
        0.2 + 0.7 * t,
        0.9 - 0.5 * t,
        1.0 - 0.8 * t,
        np.ones_like(t),
    )).astype(np.float32)


def add_points_to_front_heatmap(heatmap: np.ndarray, points: np.ndarray, settings: ExperimentSettings) -> None:
    if points.shape[0] == 0:
        return

    x_min, x_max = settings.x_range
    z_min, z_max = settings.z_range
    x_idx = ((points[:, 0] - x_min) / range_span(settings.x_range) * settings.bins_x).astype(np.int32)
    z_idx = ((points[:, 2] - z_min) / range_span(settings.z_range) * settings.bins_z).astype(np.int32)
    valid = (
        (x_idx >= 0) & (x_idx < settings.bins_x) &
        (z_idx >= 0) & (z_idx < settings.bins_z)
    )
    if not np.any(valid):
        return

    # Color/intensity is radar Y depth in meters, matching topdown_depth_heatmap.py v3.
    np.maximum.at(heatmap, (z_idx[valid], x_idx[valid]), points[:, 1][valid].astype(np.float32, copy=False))


def line_points_for_joints(joints: np.ndarray) -> np.ndarray:
    return np.asarray([joints[index] for edge in SKELETON_EDGES for index in edge], dtype=np.float32)


class DualSceneViewer:
    def __init__(self, settings: ExperimentSettings):
        try:
            import pyqtgraph as pg
            import pyqtgraph.opengl as gl
            from pyqtgraph.Qt import QtCore
            from pyqtgraph.Qt import QtGui
            from pyqtgraph.Qt import QtWidgets
        except ImportError as exc:
            raise RuntimeError('Install pyqtgraph, PyQt5 and PyOpenGL to show the experiment viewer.') from exc

        self.gl = gl
        self.qt_gui = QtGui
        self.qt_widgets = QtWidgets
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.app = QtWidgets.QApplication.instance() or pg.mkQApp('IWRL6844 Skeleton Experiment')
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle(f'IWRL6844 Dynamic Skeleton Experiment {EXPERIMENT_VERSION}')
        self.window.resize(1360, 760)
        self.window.setStyleSheet(
            'QWidget { background-color: #111318; color: #f2f4f8; }'
            'QLabel { font-family: Consolas, Microsoft JhengHei, sans-serif; }'
        )

        self.point_title = QtWidgets.QLabel('Front depth heatmap')
        self.skeleton_title = QtWidgets.QLabel('Point-driven skeleton')
        for label in (self.point_title, self.skeleton_title):
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet('font-size: 17px; font-weight: 600; padding: 8px;')

        self.heatmap = np.zeros((settings.bins_z, settings.bins_x), dtype=np.float32)
        self.point_view, self.heatmap_image = self._make_heatmap_plot(pg, QtCore, settings)
        self.skeleton_view = self._make_view(LBL_X, settings.y_range, LBL_Z)

        self.joint_scatter = gl.GLScatterPlotItem(pos=np.empty((0, 3), dtype=np.float32), color=(1.0, 0.22, 0.12, 1.0), size=10, pxMode=True)
        self.skeleton_view.addItem(self.joint_scatter)
        self.skeleton_lines = gl.GLLinePlotItem(pos=np.empty((0, 3), dtype=np.float32), color=(1.0, 0.12, 0.06, 1.0), width=2.6, mode='lines', antialias=True)
        self.skeleton_view.addItem(self.skeleton_lines)

        self.status = QtWidgets.QLabel('')
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        self.status.setMinimumHeight(48)
        self.status.setStyleSheet('font-size: 14px; padding: 7px; border-top: 1px solid rgba(255,255,255,45);')

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.point_title)
        left.addWidget(self.point_view, stretch=1)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.skeleton_title)
        right.addWidget(self.skeleton_view, stretch=1)
        top = QtWidgets.QHBoxLayout()
        top.addLayout(left, stretch=1)
        top.addLayout(right, stretch=1)
        layout = QtWidgets.QVBoxLayout(self.window)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(top, stretch=1)
        layout.addWidget(self.status)

        self.window.show()
        self.app.processEvents()

    def _make_heatmap_plot(self, pg, qt_core, settings: ExperimentSettings):
        plot = pg.PlotWidget()
        plot.setBackground((17, 19, 24))
        plot.setLabel('bottom', 'X left/right', units='m')
        plot.setLabel('left', 'Z height', units='m')
        plot.setXRange(settings.x_range[0], settings.x_range[1], padding=0)
        plot.setYRange(settings.z_range[0], settings.z_range[1], padding=0)
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.setAspectLocked(False)

        image = pg.ImageItem()
        try:
            cmap = pg.ColorMap(
                pos=np.array([0.0, 0.25, 0.50, 0.75, 1.0]),
                color=np.array([
                    [0, 0, 0, 255],
                    [38, 14, 84, 255],
                    [178, 51, 84, 255],
                    [249, 142, 8, 255],
                    [252, 255, 164, 255],
                ], dtype=np.ubyte),
            )
            image.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        image.setImage(self.heatmap, autoLevels=False, levels=settings.y_range)
        image.setRect(qt_core.QRectF(
            settings.x_range[0],
            settings.z_range[0],
            range_span(settings.x_range),
            range_span(settings.z_range),
        ))
        plot.addItem(image)
        return plot, image

    def _make_view(self, x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float]):
        x_center = range_center(x_range)
        y_center = range_center(y_range)
        z_center = range_center(z_range)
        view = self.gl.GLViewWidget()
        view.setBackgroundColor((17, 19, 24))
        view.setCameraPosition(distance=max(range_span(x_range), range_span(y_range), range_span(z_range)) * 1.9, elevation=18, azimuth=-62)
        view.opts['center'].setX(x_center)
        view.opts['center'].setY(y_center)
        view.opts['center'].setZ(z_center)

        grid = self.gl.GLGridItem()
        grid.setSize(x=range_span(x_range), y=range_span(y_range))
        grid.setSpacing(x=max(range_span(x_range) / 8.0, 0.25), y=max(range_span(y_range) / 8.0, 0.25))
        grid.translate(x_center, y_center, z_range[0])
        view.addItem(grid)
        return view

    def update(self, frame_number: int, points: np.ndarray, joints: np.ndarray, settings: ExperimentSettings) -> None:
        self.heatmap *= settings.heatmap_decay
        add_points_to_front_heatmap(self.heatmap, points, settings)
        self.heatmap_image.setImage(self.heatmap, autoLevels=False, levels=settings.y_range)

        self.joint_scatter.setData(pos=joints.astype(np.float32, copy=False), color=(1.0, 0.22, 0.12, 1.0), size=10, pxMode=True)
        self.skeleton_lines.setData(pos=line_points_for_joints(joints))
        cfg_name = os.path.basename(settings.cfg_path)
        self.point_title.setText(f'Front depth heatmap - frame {frame_number}, pts {points.shape[0]}')
        self.skeleton_title.setText(f'Point-driven skeleton - {EXPERIMENT_VERSION}')
        self.status.setText(
            f'cfg {cfg_name} | radius {settings.influence_radius:.2f} m | '
            f'pull {settings.pull_gain:.2f} | smoothing {settings.smoothing:.2f} | '
            f'left color = Y depth | left axes = X/Z'
        )
        self.app.processEvents()

    def is_open(self) -> bool:
        return self.window.isVisible()

    def process_events(self) -> None:
        self.app.processEvents()


def demo_points(frame_index: int, base_joints: np.ndarray) -> np.ndarray:
    t = frame_index * 0.12
    moving = base_joints.copy()
    moving[:, 0] += 0.08 * math.sin(t)
    moving[6, 0] -= 0.16 * math.sin(t * 1.4)
    moving[9, 0] += 0.16 * math.sin(t * 1.3)
    moving[12, 1] += 0.12 * math.sin(t * 1.1)
    moving[16, 1] -= 0.12 * math.sin(t * 1.2)

    rng = np.random.default_rng(frame_index)
    sample_indices = rng.choice(np.arange(moving.shape[0]), size=28, replace=True)
    coords = moving[sample_indices] + rng.normal(0.0, 0.055, size=(len(sample_indices), 3))
    doppler = rng.normal(0.0, 0.15, size=(len(sample_indices), 1))
    intensity = rng.uniform(8.0, 24.0, size=(len(sample_indices), 1))
    background = np.column_stack((
        rng.uniform(PC_X[0], PC_X[1], size=16),
        rng.uniform(PC_Y[0], PC_Y[1], size=16),
        rng.uniform(PC_Z[0], PC_Z[1], size=16),
        rng.normal(0.0, 0.04, size=16),
        rng.uniform(1.0, 7.0, size=16),
    ))
    return np.vstack((np.column_stack((coords, doppler, intensity)), background)).astype(np.float64)


def iter_live_frames(settings: ExperimentSettings) -> Iterable[Tuple[int, np.ndarray]]:
    uart_settings = RadarUARTSettings(
        cfg_path=settings.cfg_path,
        port_cfg=settings.port_cfg,
        port_data=settings.port_data,
        baudrate_cfg=settings.baudrate_cfg,
        baudrate_data=settings.baudrate_data,
        intensity_mode=settings.intensity_mode,
        snr_norm_mean=settings.snr_norm_mean,
        snr_norm_std=settings.snr_norm_std,
        side_info_db_limit=settings.side_info_db_limit,
    )
    capture = RadarUARTCapture(uart_settings)
    try:
        if settings.send_config:
            capture.send_config()
        else:
            capture.data_port.reset_input_buffer()
            capture.byte_buffer.clear()

        frame_count = 0
        while settings.frames < 0 or frame_count < settings.frames:
            frame_data = capture.read_frame(timeout_s=0.05)
            if frame_data is None:
                yield frame_count, np.zeros((0, 5), dtype=np.float64)
                continue
            frame_count += 1
            yield frame_data.frame_number, frame_data.points
    finally:
        capture.close()


def iter_demo_frames(settings: ExperimentSettings, base_joints: np.ndarray) -> Iterable[Tuple[int, np.ndarray]]:
    frame_count = 0
    while settings.frames < 0 or frame_count < settings.frames:
        frame_count += 1
        yield frame_count, demo_points(frame_count, base_joints)


def run_experiment(settings: ExperimentSettings) -> int:
    print(f'[INFO] {EXPERIMENT_VERSION}')
    print(f'[INFO] cfg: {settings.cfg_path}')
    print(f'[INFO] mode: {"demo synthetic points" if settings.demo else "live UART"}')
    print(f'[INFO] point view + skeleton view will open in one window')

    base_joints = make_base_skeleton(settings)
    displayed_joints = base_joints.copy()
    viewer = DualSceneViewer(settings)
    source = iter_demo_frames(settings, base_joints) if settings.demo else iter_live_frames(settings)
    update_interval_s = 1.0 / settings.update_hz if settings.update_hz > 0 else 0.0
    next_update_at = 0.0

    try:
        for frame_number, raw_points in source:
            points = clip_points(raw_points, settings)
            target_joints = dynamic_skeleton(base_joints, points, settings)
            displayed_joints = blend_joints(displayed_joints, target_joints, settings.smoothing)
            now = time.monotonic()
            if now >= next_update_at:
                viewer.update(frame_number, points, displayed_joints, settings)
                next_update_at = now + update_interval_s
            else:
                viewer.process_events()
            if not viewer.is_open():
                break
    except KeyboardInterrupt:
        print('\n[INFO] stopped by user')
    return 0


def parse_range(raw: str, label: str) -> Tuple[float, float]:
    parts = [part.strip() for part in raw.split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f'{label} range must be "min,max"')
    low, high = float(parts[0]), float(parts[1])
    if not low < high:
        raise argparse.ArgumentTypeError(f'{label} range must satisfy min < max')
    return low, high


def run_self_test() -> int:
    settings = ExperimentSettings(
        cfg_path=DEFAULT_CFG_PATH,
        port_cfg=DEFAULT_CONFIG_PORT,
        port_data=DEFAULT_DATA_PORT,
        baudrate_cfg=DEFAULT_BAUDRATE_CFG,
        baudrate_data=DEFAULT_BAUDRATE_DATA,
        frames=1,
        send_config=False,
        demo=True,
        update_hz=10.0,
        bins_x=160,
        bins_z=180,
        heatmap_decay=0.90,
        influence_radius=0.55,
        pull_gain=0.55,
        smoothing=0.65,
        min_intensity=0.0,
        x_range=PC_X,
        y_range=PC_Y,
        z_range=PC_Z,
        intensity_mode=INTENSITY_MODE,
        snr_norm_mean=SNR_NORM_MEAN,
        snr_norm_std=SNR_NORM_STD,
        side_info_db_limit=SIDE_INFO_DB_LIMIT,
    )
    base = make_base_skeleton(settings)
    points = demo_points(3, base)
    clipped = clip_points(points, settings)
    moved = dynamic_skeleton(base, clipped, settings)
    displacement = float(np.max(np.linalg.norm(moved - base, axis=1)))
    print(f'self_test points={clipped.shape[0]} max_joint_displacement={displacement:.4f}')
    return 0 if clipped.shape[0] > 0 and displacement > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description='Show radar points and an experimental point-driven skeleton.')
    parser.add_argument('--cfg', default=DEFAULT_CFG_PATH, help='Radar cfg file path or file name under cfg/')
    parser.add_argument('--port_cfg', default=DEFAULT_CONFIG_PORT, help='Config serial port')
    parser.add_argument('--port_data', default=DEFAULT_DATA_PORT, help='Data serial port')
    parser.add_argument('--baudrate_cfg', type=int, default=DEFAULT_BAUDRATE_CFG)
    parser.add_argument('--baudrate_data', type=int, default=DEFAULT_BAUDRATE_DATA)
    parser.add_argument('--frames', type=int, default=-1, help='Number of frames; -1 runs until closed')
    parser.add_argument('--no_send_config', action='store_false', dest='send_config', help='Do not send radar cfg before reading')
    parser.set_defaults(send_config=True)
    parser.add_argument('--demo', action='store_true', help='Use synthetic moving points instead of UART hardware')
    parser.add_argument('--self_test', action='store_true', help='Run non-GUI skeleton math test and exit')
    parser.add_argument('--update_hz', type=float, default=15.0)
    parser.add_argument('--bins_x', type=int, default=160, help='Front heatmap horizontal bins')
    parser.add_argument('--bins_z', type=int, default=180, help='Front heatmap height bins')
    parser.add_argument('--heatmap_decay', type=float, default=0.90, help='Front heatmap temporal decay from 0.0 to 1.0')
    parser.add_argument('--influence_radius', type=float, default=0.55, help='Meters around each joint that can pull it')
    parser.add_argument('--pull_gain', type=float, default=0.55, help='How strongly local points pull each joint')
    parser.add_argument('--smoothing', type=float, default=0.65, help='0=no smoothing, 0.95=very slow skeleton')
    parser.add_argument('--min_intensity', type=float, default=0.0, help='Drop points below this SNR/intensity')
    parser.add_argument('--x_range', default=f'{PC_X[0]},{PC_X[1]}')
    parser.add_argument('--y_range', default=f'{PC_Y[0]},{PC_Y[1]}')
    parser.add_argument('--z_range', default=f'{PC_Z[0]},{PC_Z[1]}')
    parser.add_argument('--intensity_mode', choices=['snr_db', 'snr_raw', 'snr_norm'], default=INTENSITY_MODE)
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    cfg_path = resolve_cfg_path(args.cfg)
    if not os.path.isfile(cfg_path):
        print(f'[ERROR] Config file not found: {cfg_path}')
        return 1
    if not 0.0 <= args.smoothing < 1.0:
        print('[ERROR] smoothing must satisfy 0.0 <= smoothing < 1.0')
        return 1
    if args.influence_radius <= 0 or args.pull_gain < 0:
        print('[ERROR] influence_radius must be positive and pull_gain must be non-negative')
        return 1
    if args.bins_x <= 0 or args.bins_z <= 0:
        print('[ERROR] bins_x and bins_z must be positive')
        return 1
    if not 0.0 <= args.heatmap_decay <= 1.0:
        print('[ERROR] heatmap_decay must satisfy 0.0 <= heatmap_decay <= 1.0')
        return 1

    settings = ExperimentSettings(
        cfg_path=cfg_path,
        port_cfg=args.port_cfg,
        port_data=args.port_data,
        baudrate_cfg=args.baudrate_cfg,
        baudrate_data=args.baudrate_data,
        frames=args.frames,
        send_config=args.send_config,
        demo=args.demo,
        update_hz=args.update_hz,
        bins_x=args.bins_x,
        bins_z=args.bins_z,
        heatmap_decay=args.heatmap_decay,
        influence_radius=args.influence_radius,
        pull_gain=args.pull_gain,
        smoothing=args.smoothing,
        min_intensity=args.min_intensity,
        x_range=parse_range(args.x_range, 'x'),
        y_range=parse_range(args.y_range, 'y'),
        z_range=parse_range(args.z_range, 'z'),
        intensity_mode=args.intensity_mode,
        snr_norm_mean=SNR_NORM_MEAN,
        snr_norm_std=SNR_NORM_STD,
        side_info_db_limit=SIDE_INFO_DB_LIMIT,
    )
    return run_experiment(settings)


if __name__ == '__main__':
    raise SystemExit(main())
