"""Realtime UART capture plus MARS pose prediction.

This script combines the live UART preprocessing from uart_capture.py with the
MARS model inference / skeleton format used by mars_predict.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from dataclasses import replace
from typing import Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from radar_uart import MarsFeatureMapProcessor
from radar_uart import PointFilterSettings
from radar_uart import point_filter_settings_from_config
from radar_uart import PointTransformSettings
from radar_uart import point_transform_settings_from_config
from radar_uart import parse_transform_axes
from radar_uart import RadarUARTCapture
from radar_uart import RadarUARTSettings
from util.radar_config import cfg_get
from util.radar_config import cfg_range
from util.radar_config import load_radar_config
from util.radar_config import resolve_cfg_path


RADAR_CONFIG = load_radar_config()
POINT_FILTER_SETTINGS = point_filter_settings_from_config(RADAR_CONFIG)
POINT_TRANSFORM_SETTINGS = point_transform_settings_from_config(RADAR_CONFIG)

DEFAULT_CONFIG_PORT = str(cfg_get(RADAR_CONFIG, 'radar', 'config_port', default='COM19'))
DEFAULT_DATA_PORT = str(cfg_get(RADAR_CONFIG, 'radar', 'data_port', default='COM20'))
DEFAULT_CFG_FILE = str(cfg_get(RADAR_CONFIG, 'radar', 'cfg_file', default='IWRL6844_4T4R_record_high_accuracy.cfg'))
DEFAULT_CFG_PATH = resolve_cfg_path(DEFAULT_CFG_FILE)
DEFAULT_BAUDRATE_CFG = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_cfg', default=115200))
DEFAULT_BAUDRATE_DATA = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_data', default=1250000))
DEFAULT_FRAMES = int(cfg_get(RADAR_CONFIG, 'radar', 'frames', default=-1))

INTENSITY_MODE = str(cfg_get(RADAR_CONFIG, 'point_output', 'intensity_mode', default='snr_db'))
SNR_NORM_MEAN = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_mean', default=20.0))
SNR_NORM_STD = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_std', default=10.0))
SIDE_INFO_DB_LIMIT = float(cfg_get(RADAR_CONFIG, 'point_output', 'side_info_db_limit', default=100.0))
MAX_POINTS = int(cfg_get(RADAR_CONFIG, 'feature_map', 'max_points', default=64))
MARS_FEATURE_POINTS = 64
FEATURE_DTYPE = str(cfg_get(RADAR_CONFIG, 'feature_map', 'dtype', default='float64'))
TRUNCATE_BEFORE_SORT = bool(cfg_get(RADAR_CONFIG, 'feature_map', 'truncate_before_sort', default=True))
MODEL_FILE = str(cfg_get(RADAR_CONFIG, 'paths', 'model_file', default='MARS.h5'))

PC_X, PC_Y, PC_Z = cfg_range(RADAR_CONFIG, 'point_cloud')
LBL_X, LBL_Y, LBL_Z = cfg_range(RADAR_CONFIG, 'label')

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

JOINT_ANGLES = {
    'Left elbow': (4, 5, 6),
    'Right elbow': (7, 8, 9),
    'Left knee': (10, 11, 12),
    'Right knee': (14, 15, 16),
}


@dataclass(frozen=True)
class RealtimePredictSettings:
    cfg_path: str
    model_path: str
    port_cfg: str
    port_data: str
    baudrate_cfg: int
    baudrate_data: int
    frames: int
    send_config: bool
    update_hz: float
    predict_every: int
    point_filter: PointFilterSettings
    point_transform: PointTransformSettings
    max_points: int
    truncate_before_sort: bool
    dtype: str
    intensity_mode: str
    snr_norm_mean: float
    snr_norm_std: float
    side_info_db_limit: float
    display_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    label_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    print_summary: bool
    summary_every: int
    print_points: bool


def range_center(value_range: Tuple[float, float]) -> float:
    return (value_range[0] + value_range[1]) * 0.5


def range_span(value_range: Tuple[float, float]) -> float:
    return max(value_range[1] - value_range[0], 1e-6)


def label_to_joints(label_57: np.ndarray) -> np.ndarray:
    return np.stack([label_57[0:19], label_57[19:38], label_57[38:57]], axis=1)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def get_angles(joints: np.ndarray) -> dict:
    return {
        name: joint_angle(joints[a_idx], joints[b_idx], joints[c_idx])
        for name, (a_idx, b_idx, c_idx) in JOINT_ANGLES.items()
    }


def line_points_for_joints(joints: np.ndarray) -> np.ndarray:
    return np.asarray([joints[index] for edge in SKELETON_EDGES for index in edge], dtype=np.float32)


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


def format_point_line(point_index: int, point: np.ndarray) -> str:
    x, y, z, doppler, intensity = point
    return f'{point_index:02d}: x={x: .4f}, y={y: .4f}, z={z: .4f}, doppler={doppler: .4f}, intensity={intensity: .2f}'


def format_value_range(values: np.ndarray) -> str:
    return f'{np.min(values): .3f}..{np.max(values): .3f}'


def print_point_summary(frame_index: int, radar_frame_number: int, raw_count: int, points: np.ndarray, inference_ms: float) -> None:
    if points.shape[0] == 0:
        print(f'[Frame {frame_index:04d}] radar={radar_frame_number} raw={raw_count} valid=0 no points inference={inference_ms:.1f}ms')
        return

    print(
        f'[Frame {frame_index:04d}] '
        f'radar={radar_frame_number} raw={raw_count} valid={points.shape[0]} '
        f'x={format_value_range(points[:, 0])} '
        f'y={format_value_range(points[:, 1])} '
        f'z={format_value_range(points[:, 2])} '
        f'doppler={format_value_range(points[:, 3])} '
        f'intensity_avg={np.mean(points[:, 4]):.2f} '
        f'intensity_max={np.max(points[:, 4]):.2f} '
        f'inference={inference_ms:.1f}ms'
    )


def print_point_details(frame_index: int, radar_frame_number: int, raw_count: int, points: np.ndarray, inference_ms: float) -> None:
    print()
    print(f'========== Frame {frame_index:04d} / radar {radar_frame_number} ==========')
    print(f'raw_points     : {raw_count}')
    print(f'valid_points   : {points.shape[0]}')
    print(f'inference_ms   : {inference_ms:.1f}')
    if points.shape[0] == 0:
        print('(no points)')
        return
    for index, point in enumerate(points, start=1):
        print(format_point_line(index, point))


def _view_center_tuple(view) -> tuple:
    center = view.opts['center']
    return float(center.x()), float(center.y()), float(center.z())


def _camera_state(view, base_center: tuple) -> tuple:
    center_x, center_y, center_z = _view_center_tuple(view)
    return (
        float(view.opts.get('distance', 0.0)),
        float(view.opts.get('elevation', 0.0)),
        float(view.opts.get('azimuth', 0.0)),
        center_x - base_center[0],
        center_y - base_center[1],
        center_z - base_center[2],
    )


def _apply_camera_state(view, base_center: tuple, state: tuple) -> None:
    distance, elevation, azimuth, offset_x, offset_y, offset_z = state
    view.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)
    view.opts['center'].setX(base_center[0] + offset_x)
    view.opts['center'].setY(base_center[1] + offset_y)
    view.opts['center'].setZ(base_center[2] + offset_z)
    view.update()


class LinkedGLCamera:
    def __init__(self, views, qt_core, interval_ms: int = 40):
        self.views = list(views)
        self.base_centers = [_view_center_tuple(view) for view in self.views]
        self._syncing = False
        self._states = self._current_states()
        for index, view in enumerate(self.views):
            self._wrap_view_events(index, view)
        self.timer = qt_core.QTimer()
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self.sync_from(0)

    def _current_states(self) -> list:
        return [
            _camera_state(view, self.base_centers[index])
            for index, view in enumerate(self.views)
        ]

    def _wrap_view_events(self, index: int, view) -> None:
        for event_name in ('mouseMoveEvent', 'mouseReleaseEvent', 'wheelEvent'):
            original = getattr(view, event_name)

            def wrapped(event, original=original, source_index=index):
                result = original(event)
                self.sync_from(source_index)
                return result

            setattr(view, event_name, wrapped)

    def sync_from(self, source_index: int) -> None:
        if self._syncing:
            return

        source_state = _camera_state(self.views[source_index], self.base_centers[source_index])
        self._syncing = True
        try:
            for index, view in enumerate(self.views):
                if index != source_index:
                    _apply_camera_state(view, self.base_centers[index], source_state)
            self._states = self._current_states()
        finally:
            self._syncing = False

    def _poll(self) -> None:
        if self._syncing:
            return

        states = self._current_states()
        changed_index = None
        for index, (previous, current) in enumerate(zip(self._states, states)):
            if any(abs(a - b) > 1e-5 for a, b in zip(previous, current)):
                changed_index = index
                break

        if changed_index is None:
            return

        self.sync_from(changed_index)


def featuremap_to_model_input(fmap: np.ndarray) -> np.ndarray:
    return fmap[np.newaxis, ...].astype(np.float32, copy=False)


def create_feature_processor(settings: RealtimePredictSettings) -> MarsFeatureMapProcessor:
    dtype = np.float64 if settings.dtype == 'float64' else np.float32
    return MarsFeatureMapProcessor(
        point_filter=settings.point_filter,
        point_transform=settings.point_transform,
        max_points=settings.max_points,
        truncate_before_sort=settings.truncate_before_sort,
        dtype=dtype,
    )


def load_mars_model(model_path: str):
    try:
        from keras.models import load_model
    except ImportError:
        from tensorflow.keras.models import load_model

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'找不到模型: {model_path}')
    print(f'[INFO] 載入模型: {model_path}')
    model = load_model(model_path, compile=False)
    expected = tuple(model.input_shape[1:])
    if expected != (8, 8, 5):
        print(f'[WARN] 模型 input_shape={expected}，目前即時輸入會是 (8, 8, 5)')
    return model


class RealtimeMARSViewer:
    def __init__(self, settings: RealtimePredictSettings):
        try:
            import pyqtgraph as pg
            import pyqtgraph.opengl as gl
            from pyqtgraph.Qt import QtCore
            from pyqtgraph.Qt import QtWidgets
        except ImportError as exc:
            raise RuntimeError('需要安裝 pyqtgraph、PyQt5、PyOpenGL 才能顯示即時視窗。') from exc

        self.gl = gl
        self.app = QtWidgets.QApplication.instance() or pg.mkQApp('UART Realtime MARS Predict')
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle('UART Realtime MARS Predict')
        self.window.resize(1360, 760)
        self.window.setStyleSheet(
            'QWidget { background-color: #111318; color: #f2f4f8; }'
            'QLabel { font-family: Consolas, Microsoft JhengHei, sans-serif; }'
        )

        self.point_title = QtWidgets.QLabel('Radar point cloud')
        self.pose_title = QtWidgets.QLabel('MARS realtime estimation')
        for label in (self.point_title, self.pose_title):
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet('font-size: 17px; font-weight: 600; padding: 8px;')

        self.point_view = self._make_view(settings.display_ranges)
        self.pose_view = self._make_view(settings.label_ranges)

        self.point_scatter = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
            size=8,
            pxMode=True,
        )
        self.point_view.addItem(self.point_scatter)

        self.joint_scatter = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(1.0, 0.22, 0.12, 1.0),
            size=10,
            pxMode=True,
        )
        self.pose_view.addItem(self.joint_scatter)

        self.skeleton_lines = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(1.0, 0.12, 0.06, 1.0),
            width=2.6,
            mode='lines',
            antialias=True,
        )
        self.pose_view.addItem(self.skeleton_lines)

        self.status = QtWidgets.QLabel('')
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        self.status.setMinimumHeight(54)
        self.status.setStyleSheet('font-size: 14px; padding: 7px; border-top: 1px solid rgba(255,255,255,45);')

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.point_title)
        left.addWidget(self.point_view, stretch=1)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.pose_title)
        right.addWidget(self.pose_view, stretch=1)
        top = QtWidgets.QHBoxLayout()
        top.addLayout(left, stretch=1)
        top.addLayout(right, stretch=1)
        layout = QtWidgets.QVBoxLayout(self.window)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(top, stretch=1)
        layout.addWidget(self.status)

        self.camera_sync = LinkedGLCamera((self.point_view, self.pose_view), QtCore)
        self.window.show()
        self.app.processEvents()

    def _make_view(self, ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]):
        x_range, y_range, z_range = ranges
        view = self.gl.GLViewWidget()
        view.setBackgroundColor((17, 19, 24))
        view.setCameraPosition(
            distance=max(range_span(x_range), range_span(y_range), range_span(z_range)) * 1.9,
            elevation=18,
            azimuth=-62,
        )
        view.opts['center'].setX(range_center(x_range))
        view.opts['center'].setY(range_center(y_range))
        view.opts['center'].setZ(range_center(z_range))

        grid = self.gl.GLGridItem()
        grid.setSize(x=range_span(x_range), y=range_span(y_range))
        grid.setSpacing(x=max(range_span(x_range) / 8.0, 0.25), y=max(range_span(y_range) / 8.0, 0.25))
        grid.translate(range_center(x_range), range_center(y_range), z_range[0])
        view.addItem(grid)
        return view

    def update(self, frame_number: int, points: np.ndarray, joints: np.ndarray, inference_ms: float, settings: RealtimePredictSettings) -> None:
        if points.shape[0] == 0:
            self.point_scatter.setData(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
            )
        else:
            self.point_scatter.setData(
                pos=points[:, 0:3].astype(np.float32, copy=False),
                color=y_to_rgba(points[:, 1], settings.display_ranges[1]),
                size=8,
                pxMode=True,
            )

        self.joint_scatter.setData(
            pos=joints.astype(np.float32, copy=False),
            color=(1.0, 0.22, 0.12, 1.0),
            size=10,
            pxMode=True,
        )
        self.skeleton_lines.setData(pos=line_points_for_joints(joints))

        angles = get_angles(joints)
        self.point_title.setText(f'Radar point cloud - frame {frame_number}, pts {points.shape[0]}')
        self.pose_title.setText('MARS realtime estimation')
        self.status.setText(
            f'inference {inference_ms:.1f} ms | '
            f'LeftElbow {angles["Left elbow"]:.0f} | RightElbow {angles["Right elbow"]:.0f} | '
            f'LeftKnee {angles["Left knee"]:.0f} | RightKnee {angles["Right knee"]:.0f}'
        )
        self.app.processEvents()

    def process_events(self) -> None:
        self.app.processEvents()

    def is_open(self) -> bool:
        return self.window.isVisible()


def run_realtime(settings: RealtimePredictSettings) -> int:
    print(f'[INFO] Config: {settings.cfg_path}')
    print(f'[INFO] Model: {settings.model_path}')
    print(f'[INFO] Config serial port: {settings.port_cfg}')
    print(f'[INFO] Data serial port: {settings.port_data}')
    print(f'[INFO] send_config: {settings.send_config}')
    print(f'[INFO] update_hz: {settings.update_hz}, predict_every: {settings.predict_every}')
    print(f'[INFO] filter_roi: {settings.point_filter.roi_enabled}')
    print(f'[INFO] print_summary: {settings.print_summary}, summary_every: {settings.summary_every}, print_points: {settings.print_points}')
    print(f'[INFO] height_normalization: {settings.point_transform.height_normalize_enabled}')
    if settings.point_transform.height_normalize_enabled:
        scale = settings.point_transform.target_height_m / settings.point_transform.source_height_m
        print(f'[INFO] height_scale: {settings.point_transform.source_height_m:.2f}m -> {settings.point_transform.target_height_m:.2f}m ({scale:.3f}), axes={settings.point_transform.height_axes}, floor_z={settings.point_transform.floor_z_m:.2f}, x_center={settings.point_transform.x_center_m:.2f}')
    print(f'[INFO] capture max_points: {settings.max_points}')

    if settings.max_points != MARS_FEATURE_POINTS:
        print(f'[ERROR] 即時 MARS 推論需要 feature_map.max_points={MARS_FEATURE_POINTS}，目前是 {settings.max_points}')
        print('[ERROR] 請修改 cfg/radar_uart_config.yaml，或保持 capture 輸出為 (8,8,5)。')
        return 1

    model = load_mars_model(settings.model_path)
    processor = create_feature_processor(settings)
    viewer = RealtimeMARSViewer(settings)
    capture = None

    try:
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
        if settings.send_config:
            capture.send_config()
        else:
            capture.data_port.reset_input_buffer()
            capture.byte_buffer.clear()
            print('[INFO] 略過送出 config，直接讀取目前 data stream。')

        frame_count = 0
        update_interval_s = 1.0 / settings.update_hz if settings.update_hz > 0 else 0.0
        next_update_at = 0.0
        print('[INFO] 開始即時推論。關閉視窗或 Ctrl+C 可停止。')

        while settings.frames < 0 or frame_count < settings.frames:
            frame_data = capture.read_frame(timeout_s=0.02)
            if frame_data is None:
                viewer.process_events()
                if not viewer.is_open():
                    break
                continue

            frame_count += 1
            points, fmap = processor.process_frame(frame_data)
            raw_count = 0 if frame_data.points is None else frame_data.points.shape[0]
            did_predict = (frame_count - 1) % settings.predict_every == 0

            if did_predict:
                model_input = featuremap_to_model_input(fmap)
                start = time.monotonic()
                pred = model.predict(model_input, batch_size=1, verbose=0)[0]
                inference_ms = (time.monotonic() - start) * 1000.0
                joints = label_to_joints(pred).astype(np.float32, copy=False)

                if settings.print_points and frame_count % settings.summary_every == 0:
                    print_point_details(frame_count, frame_data.frame_number, raw_count, points, inference_ms)
                elif settings.print_summary and frame_count % settings.summary_every == 0:
                    print_point_summary(frame_count, frame_data.frame_number, raw_count, points, inference_ms)

                now = time.monotonic()
                if now >= next_update_at:
                    viewer.update(frame_data.frame_number, points, joints, inference_ms, settings)
                    next_update_at = now + update_interval_s
                else:
                    viewer.process_events()
            else:
                viewer.process_events()

            if not viewer.is_open():
                break

    except KeyboardInterrupt:
        print('\n[INFO] 使用者中斷。')
    finally:
        if capture is not None:
            capture.close()

    print('[INFO] 即時推論結束。')
    return 0


def main() -> int:
    default_model_path = os.path.join(PROJECT_ROOT, 'model', MODEL_FILE)

    parser = argparse.ArgumentParser(description='UART 即時擷取並執行 MARS 姿態推論')
    parser.add_argument('--cfg', default=DEFAULT_CFG_PATH, help='Radar cfg 檔案路徑或 cfg/ 底下的檔名')
    parser.add_argument('--model', default=default_model_path, help='MARS .h5 模型路徑')
    parser.add_argument('--port_cfg', default=DEFAULT_CONFIG_PORT, help='Config serial port')
    parser.add_argument('--port_data', default=DEFAULT_DATA_PORT, help='Data serial port')
    parser.add_argument('--baudrate_cfg', type=int, default=DEFAULT_BAUDRATE_CFG)
    parser.add_argument('--baudrate_data', type=int, default=DEFAULT_BAUDRATE_DATA)
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES, help='要處理的 frame 數，-1 代表持續執行')
    parser.add_argument('--no_send_config', action='store_false', dest='send_config', help='不送出 radar cfg，直接讀目前 data stream')
    parser.set_defaults(send_config=True)
    parser.add_argument('--update_hz', type=float, default=10.0, help='畫面更新率')
    parser.add_argument('--predict_every', type=int, default=1, help='每幾個 frame 做一次模型推論')
    parser.add_argument('--no_print_summary', action='store_false', dest='print_summary', help='關閉每個 frame 的 cmd 點雲摘要')
    parser.add_argument('--summary_every', type=int, default=1, help='每幾個 frame 印一次 cmd 點雲資訊')
    parser.add_argument('--print_points', action='store_true', help='逐 frame 印出所有有效點；會降低即時推論流暢度')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default=FEATURE_DTYPE)
    parser.add_argument(
        '--intensity_mode',
        choices=['snr_db', 'snr_raw', 'snr_norm'],
        default=INTENSITY_MODE,
        help='第 5 維 intensity 的來源',
    )
    parser.add_argument('--snr_norm_mean', type=float, default=SNR_NORM_MEAN)
    parser.add_argument('--snr_norm_std', type=float, default=SNR_NORM_STD)
    parser.add_argument('--filter_roi', action='store_true', default=POINT_FILTER_SETTINGS.roi_enabled)
    parser.add_argument('--no_filter_roi', action='store_false', dest='filter_roi')
    parser.add_argument('--height_normalize', action='store_true', default=POINT_TRANSFORM_SETTINGS.height_normalize_enabled, help='啟用身高正規化縮放')
    parser.add_argument('--no_height_normalize', action='store_false', dest='height_normalize', help='關閉身高正規化縮放')
    parser.add_argument('--source_height_m', type=float, default=POINT_TRANSFORM_SETTINGS.source_height_m, help='身高正規化來源身高，例如 1.90')
    parser.add_argument('--target_height_m', type=float, default=POINT_TRANSFORM_SETTINGS.target_height_m, help='身高正規化目標身高，例如 1.75')
    parser.add_argument('--height_axes', default=','.join(POINT_TRANSFORM_SETTINGS.height_axes), help='要縮放的座標軸，例如 x,z 或 z；預設不縮放 y 深度')
    parser.add_argument('--floor_z_m', type=float, default=POINT_TRANSFORM_SETTINGS.floor_z_m, help='Z 軸縮放錨點；地板高度，縮放時地板不動')
    parser.add_argument('--x_center_m', type=float, default=POINT_TRANSFORM_SETTINGS.x_center_m, help='X 軸縮放中心線；預設 0.0')
    args = parser.parse_args()

    cfg_path = resolve_cfg_path(args.cfg)
    model_path = args.model if os.path.isabs(args.model) else os.path.join(PROJECT_ROOT, args.model)
    if not os.path.isfile(cfg_path):
        print(f'[ERROR] Config 檔案不存在: {cfg_path}')
        return 1
    if not os.path.isfile(model_path):
        print(f'[ERROR] Model 檔案不存在: {model_path}')
        return 1
    if args.predict_every <= 0:
        print('[ERROR] predict_every must be positive.')
        return 1
    if args.summary_every <= 0:
        print('[ERROR] summary_every must be positive.')
        return 1

    settings = RealtimePredictSettings(
        cfg_path=cfg_path,
        model_path=model_path,
        port_cfg=args.port_cfg,
        port_data=args.port_data,
        baudrate_cfg=args.baudrate_cfg,
        baudrate_data=args.baudrate_data,
        frames=args.frames,
        send_config=args.send_config,
        update_hz=args.update_hz,
        predict_every=args.predict_every,
        point_filter=replace(POINT_FILTER_SETTINGS, roi_enabled=args.filter_roi),
        point_transform=PointTransformSettings(
            height_normalize_enabled=args.height_normalize,
            source_height_m=args.source_height_m,
            target_height_m=args.target_height_m,
            height_axes=parse_transform_axes(args.height_axes),
            floor_z_m=args.floor_z_m,
            x_center_m=args.x_center_m,
        ),
        max_points=MAX_POINTS,
        truncate_before_sort=TRUNCATE_BEFORE_SORT,
        dtype=args.dtype,
        intensity_mode=args.intensity_mode,
        snr_norm_mean=args.snr_norm_mean,
        snr_norm_std=args.snr_norm_std,
        side_info_db_limit=SIDE_INFO_DB_LIMIT,
        display_ranges=(PC_X, PC_Y, PC_Z),
        label_ranges=(LBL_X, LBL_Y, LBL_Z),
        print_summary=args.print_summary,
        summary_every=args.summary_every,
        print_points=args.print_points,
    )
    return run_realtime(settings)


if __name__ == '__main__':
    raise SystemExit(main())
