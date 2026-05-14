# -*- coding: utf-8 -*-
"""
show_feature.py
===============
讀入轉換後的 feature map (.npy) 並互動顯示 3D 點雲。

用法：
    # 顯示預設檔案（自動選擇最近一次的 radar_capture_*.npy）
  python show_feature.py

  # 指定自訂檔案
  python show_feature.py --input feature/reference/mars_pointcloud_0506.npy

  # 快速切換類別
  python show_feature.py --file_class reference

預設檔案位置：
    feature/test/radar_capture_*.npy

互動控制：
  ← → 鍵     : 逐 frame 切換
  A / D 鍵   : 逐 frame 切換
  PageUp     : 往前跳 50 frames
  PageDown   : 往後跳 50 frames
  Qt 滑塊    : 直接選擇 frame

顯示內容：
  - pyqtgraph OpenGL 3D 點雲散點圖
  - x 軸：左右（m）
  - y 軸：深度（m）
  - z 軸：高度（m）
  - 顏色：強度值（turbo colormap）

輸入格式：
  feature map (.npy)
  形狀：(N, 8, 8, 5)  其中 5 = [x, y, z, doppler, intensity]
"""

import os, sys, argparse
import numpy as np

from pointcloud_pyqtgraph import add_floor_plane
from pointcloud_pyqtgraph import clip_display_points
from pointcloud_pyqtgraph import range_center
from pointcloud_pyqtgraph import range_span
from util.AbsDir import AbsDir
from util.AbsDir import FileClass

from util.find_file import find_default_feature_file
from util.radar_config import as_bool
from util.radar_config import cfg_get
from util.radar_config import cfg_range
from util.radar_config import ensure_suffix
from util.radar_config import load_radar_config
from util.radar_config import resolve_under_root

RADAR_CONFIG = load_radar_config()
DEFAULT_FILE_CLASS = str(cfg_get(RADAR_CONFIG, 'paths', 'default_file_class', default='test'))
FEATURE_FILE = str(cfg_get(RADAR_CONFIG, 'paths', 'default_feature_file', default='radar_capture_0.npy'))
SCATTER_SIZE = float(cfg_get(RADAR_CONFIG, 'display', 'scatter', 'size', default=45))
SCATTER_ALPHA = float(cfg_get(RADAR_CONFIG, 'display', 'scatter', 'alpha', default=0.95))

# 軸範圍
PC_X, PC_Y, PC_Z = cfg_range(RADAR_CONFIG, 'point_cloud')


def file_class_from_value(value):
    if isinstance(value, FileClass):
        return value
    text = str(value).strip().lower()
    mapping = {
        'test': FileClass.TEST,
        'standard': FileClass.STANDARD,
        'reference': FileClass.REFERENCE,
        '0': FileClass.TEST,
        '1': FileClass.STANDARD,
        '2': FileClass.REFERENCE,
    }
    return mapping.get(text, FileClass.TEST)


def resolve_feature_input(abs_dir: AbsDir, file_class: FileClass, input_path: str) -> str:
    if os.path.isabs(input_path) and os.path.isfile(input_path):
        return os.path.normpath(input_path)

    project_candidate = resolve_under_root(input_path)
    if os.path.isfile(project_candidate):
        return project_candidate

    return os.path.join(abs_dir.get_feature_dir_by_class(file_class), os.path.basename(input_path))


def fmap_to_pts(fmap):
    """轉換 (8,8,5) feature map 為點雲 (N,5)"""
    pts = fmap.reshape(-1, 5)
    return pts[np.any(pts != 0, axis=1)]


def intensity_to_rgba(intensity: np.ndarray) -> np.ndarray:
    """Map point intensity to a compact turbo-like RGBA ramp for pyqtgraph."""
    if intensity.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    p1, p99 = np.percentile(intensity.astype(np.float32, copy=False), [1, 99])
    if p99 - p1 < 1e-6:
        t = np.zeros_like(intensity, dtype=np.float32)
    else:
        t = np.clip((intensity - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)

    return np.column_stack((
        np.clip(1.5 * t, 0.0, 1.0),
        np.clip(1.5 - np.abs(t - 0.5) * 2.2, 0.0, 1.0),
        np.clip(1.2 - 1.7 * t, 0.0, 1.0),
        np.full_like(t, SCATTER_ALPHA),
    )).astype(np.float32)



class PointCloudViewer:
    def __init__(self, fmaps, title_suffix=''):
        self.fmaps = fmaps
        self.N = len(fmaps)
        self.idx = 0
        self.title_suffix = title_suffix
        self._build()
        self._update(0)

    def _build(self):
        try:
            import pyqtgraph as pg
            import pyqtgraph.opengl as gl
            from pyqtgraph.Qt import QtCore
            from pyqtgraph.Qt import QtGui
            from pyqtgraph.Qt import QtWidgets
        except ImportError as exc:
            raise RuntimeError(
                '需要先安裝 pyqtgraph 3D 依賴：python -m pip install pyqtgraph PyQt5 PyOpenGL'
            ) from exc

        self.gl = gl
        self.qt_core = QtCore
        self.app = QtWidgets.QApplication.instance() or pg.mkQApp('Point Cloud Viewer')

        window_title = 'Point Cloud Viewer'
        if self.title_suffix:
            window_title += f' - {self.title_suffix}'

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle(window_title)
        self.window.resize(1040, 820)
        self.window.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.window.keyPressEvent = self._on_key
        self.window.setStyleSheet(
            'QWidget { background-color: #121418; color: #f4f4f4; }'
            'QLabel { font-family: Consolas, Microsoft JhengHei, sans-serif; }'
        )

        self.title_label = QtWidgets.QLabel('')
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet('font-size: 18px; font-weight: 600; padding: 8px;')

        self.view = self._make_view(gl)
        self.scatter = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
            size=max(SCATTER_SIZE * 0.18, 4.0),
            pxMode=True,
        )
        self.view.addItem(self.scatter)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(self.N - 1, 0))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(50)
        self.slider.valueChanged.connect(self._update)

        self.status_label = QtWidgets.QLabel('')
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet('font-size: 13px; padding: 4px;')

        layout = QtWidgets.QVBoxLayout(self.window)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.slider)
        layout.addWidget(self.status_label)

    def _make_view(self, gl):
        x_center = range_center(PC_X)
        y_center = range_center(PC_Y)
        z_center = range_center(PC_Z)
        x_span = range_span(PC_X)
        y_span = range_span(PC_Y)
        z_span = range_span(PC_Z)

        view = gl.GLViewWidget()
        view.setBackgroundColor((18, 20, 24))
        view.setCameraPosition(distance=max(x_span, y_span, z_span) * 1.8, elevation=18, azimuth=-62)
        view.opts['center'].setX(x_center)
        view.opts['center'].setY(y_center)
        view.opts['center'].setZ(z_center)

        add_floor_plane(view, gl, (PC_X, PC_Y, PC_Z))
        return view

    def _update(self, idx):
        idx = int(np.clip(idx, 0, self.N-1))
        self.idx = idx
        if self.slider.value() != idx:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)

        pts = clip_display_points(fmap_to_pts(self.fmaps[idx]), (PC_X, PC_Y, PC_Z))
        if pts.size == 0:
            self.scatter.setData(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
            )
        else:
            self.scatter.setData(
                pos=pts[:, 0:3].astype(np.float32, copy=False),
                color=intensity_to_rgba(pts[:, 4]),
                size=max(SCATTER_SIZE * 0.18, 4.0),
                pxMode=True,
            )

        self.title_label.setText(f'Radar Point Cloud - Frame {self.idx}')
        self.status_label.setText(f'Frame {self.idx + 1}/{self.N}    pts {len(pts)}')
        self.app.processEvents()

    def _on_key(self, event):
        key = event.key()
        text = event.text().lower()
        step = 0
        if key == self.qt_core.Qt.Key_Right or text == 'd':
            step = 1
        elif key == self.qt_core.Qt.Key_Left or text == 'a':
            step = -1
        elif key == self.qt_core.Qt.Key_PageDown:
            step = 50
        elif key == self.qt_core.Qt.Key_PageUp:
            step = -50
        if step:
            self._update(int(np.clip(self.idx + step, 0, self.N-1)))
            event.accept()
        else:
            event.ignore()

    def show(self):
        self.window.show()
        self.window.activateWindow()
        self.window.setFocus()
        self.app.exec_()


def main():    
    parser = argparse.ArgumentParser(description='顯示雷達點雲')
    parser.add_argument('--input', default=None, help='輸入 .npy 檔案名稱')
    parser.add_argument('--auto', default=False, help='自動尋找最新的 radar_capture_*.npy')
    parser.add_argument('--file_class', default=None, help='檔案類別：test / standard / reference 或 0 / 1 / 2')
    args = parser.parse_args()
    
    absDir = AbsDir()
    default_file_class = file_class_from_value(DEFAULT_FILE_CLASS)
    
    args.file_class = default_file_class if args.file_class is None else file_class_from_value(args.file_class)
    print(f'[INFO] 使用的 file_class: {args.file_class}')
    path_feature_dir = absDir.get_feature_dir_by_class(args.file_class)
    
    if as_bool(args.auto):
        args.input = find_default_feature_file(path_feature_dir)
        if args.input is None:
            print(f'[ERROR] 在 \"{path_feature_dir}\" 找不到任何 .npy 檔案')
            os._exit(0)
    else:
        if args.input is None:
            args.input = ensure_suffix(FEATURE_FILE, '.npy')
        args.input = resolve_feature_input(absDir, args.file_class, args.input)
        if not os.path.isfile(args.input):
            print(f'[WARNING] 輸入檔案 \"{args.input}\" 不存在')
            os._exit(0)
            
    print(f'[輸入] {args.input}')
    fmaps = np.load(args.input).astype(np.float32)
    print(f'Shape: {fmaps.shape}')

    if fmaps.ndim != 4 or fmaps.shape[1:] != (8, 8, 5):
        print(f'[WARNING] 期望形狀 (N,8,8,5)，實際 {fmaps.shape}')

    file_name = os.path.splitext(os.path.basename(args.input))[0]
    print(f'\n[控制] ← → 換 frame｜A/D 換 frame｜PageUp/Down 跳 50 frames')
    PointCloudViewer(fmaps, title_suffix=file_name).show()


if __name__ == '__main__':
    main()
