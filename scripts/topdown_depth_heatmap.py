"""Pseudo-depth heatmap viewer for TI IWRL6844 UART point clouds.

This script intentionally lives on its own and reuses the existing UART parser.
It visualizes detected points as a fading heatmap. The top-down view uses X/Y;
the front view uses X/Z for a thermal-camera-like projection.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
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


RADAR_CONFIG = load_radar_config()

DEFAULT_CONFIG_PORT = str(cfg_get(RADAR_CONFIG, 'radar', 'config_port', default='COM19'))
DEFAULT_DATA_PORT = str(cfg_get(RADAR_CONFIG, 'radar', 'data_port', default='COM20'))
DEFAULT_CFG_FILE = str(cfg_get(RADAR_CONFIG, 'radar', 'cfg_file', default='IWRL6844_4T4R_record_high_accuracy.cfg'))
DEFAULT_CFG_PATH = resolve_cfg_path(DEFAULT_CFG_FILE)
DEFAULT_BAUDRATE_CFG = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_cfg', default=115200))
DEFAULT_BAUDRATE_DATA = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_data', default=1250000))
DEFAULT_DISPLAY_X, DEFAULT_DISPLAY_Y, DEFAULT_DISPLAY_Z = cfg_range(RADAR_CONFIG, 'point_cloud')

INTENSITY_MODE = str(cfg_get(RADAR_CONFIG, 'point_output', 'intensity_mode', default='snr_db'))
SNR_NORM_MEAN = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_mean', default=20.0))
SNR_NORM_STD = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_std', default=10.0))
SIDE_INFO_DB_LIMIT = float(cfg_get(RADAR_CONFIG, 'point_output', 'side_info_db_limit', default=100.0))
VIEWER_VERSION = 'v3-depth-color'


@dataclass(frozen=True)
class ViewerSettings:
    cfg_path: str
    port_cfg: str
    port_data: str
    baudrate_cfg: int
    baudrate_data: int
    frames: int
    send_config: bool
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    bins_x: int
    bins_y: int
    fade_step: float
    fade_floor: float
    update_hz: float
    view_mode: str
    value_mode: str
    moving_only: bool
    min_abs_doppler: float
    min_intensity: Optional[float]
    intensity_mode: str
    snr_norm_mean: float
    snr_norm_std: float
    side_info_db_limit: float


def parse_range(raw: str, label: str) -> Tuple[float, float]:
    parts = [part.strip() for part in raw.split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f'{label} range must be "min,max"')
    low, high = float(parts[0]), float(parts[1])
    if not low < high:
        raise argparse.ArgumentTypeError(f'{label} range must satisfy min < max')
    return low, high


def points_in_roi(points: np.ndarray, settings: ViewerSettings) -> np.ndarray:
    if points is None or points.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float64)

    mask = np.any(points != 0, axis=1)
    mask &= (points[:, 0] >= settings.x_range[0]) & (points[:, 0] <= settings.x_range[1])
    mask &= (points[:, 1] >= settings.y_range[0]) & (points[:, 1] <= settings.y_range[1])
    mask &= (points[:, 2] >= settings.z_range[0]) & (points[:, 2] <= settings.z_range[1])

    if settings.moving_only:
        mask &= np.abs(points[:, 3]) >= settings.min_abs_doppler

    if settings.min_intensity is not None:
        mask &= points[:, 4] >= settings.min_intensity

    return points[mask]


def point_values(points: np.ndarray, mode: str) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    if mode == 'depth':
        return points[:, 1].astype(np.float32, copy=False)
    if mode == 'occupancy':
        return np.ones((points.shape[0],), dtype=np.float32)
    if mode == 'doppler':
        return np.abs(points[:, 3]).astype(np.float32, copy=False)
    if mode == 'nearest':
        depth = points[:, 1]
        return (1.0 / np.maximum(depth, 0.05)).astype(np.float32, copy=False)

    intensity = points[:, 4].astype(np.float32, copy=False)
    return np.clip(intensity, 0.0, None)


def value_label(mode: str) -> str:
    if mode == 'depth':
        return 'Y depth/range (m)'
    if mode == 'nearest':
        return 'nearer-is-brighter (1 / Y)'
    if mode == 'doppler':
        return 'abs doppler'
    if mode == 'occupancy':
        return 'occupancy'
    return 'intensity/SNR'


def projection_axis(settings: ViewerSettings) -> Tuple[int, Tuple[float, float], str]:
    if settings.view_mode == 'front':
        return 2, settings.z_range, 'Z height (m)'
    return 1, settings.y_range, 'Y depth/range (m)'


def add_points_to_heatmap(heatmap: np.ndarray, points: np.ndarray, settings: ViewerSettings) -> None:
    if points.shape[0] == 0:
        return

    x_min, x_max = settings.x_range
    vertical_axis, vertical_range, _vertical_label = projection_axis(settings)
    vertical_min, vertical_max = vertical_range
    x_idx = ((points[:, 0] - x_min) / (x_max - x_min) * settings.bins_x).astype(np.int32)
    y_idx = ((points[:, vertical_axis] - vertical_min) / (vertical_max - vertical_min) * settings.bins_y).astype(np.int32)

    valid = (
        (x_idx >= 0) & (x_idx < settings.bins_x) &
        (y_idx >= 0) & (y_idx < settings.bins_y)
    )
    if not np.any(valid):
        return

    values = point_values(points, settings.value_mode)
    np.maximum.at(heatmap, (y_idx[valid], x_idx[valid]), values[valid])


def fade_heatmap(heatmap: np.ndarray, settings: ViewerSettings) -> None:
    if settings.fade_step > 0.0:
        np.subtract(heatmap, settings.fade_step, out=heatmap)
        np.maximum(heatmap, 0.0, out=heatmap)
    if settings.fade_floor > 0.0:
        heatmap[heatmap < settings.fade_floor] = 0.0


def init_plot(settings: ViewerSettings):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = np.zeros((settings.bins_y, settings.bins_x), dtype=np.float32)
    _vertical_axis, vertical_range, vertical_label = projection_axis(settings)
    cmap = 'inferno' if settings.view_mode == 'front' else 'turbo'
    title = (
        'IWRL6844 4T4R Front Pseudo-IR Heatmap'
        if settings.view_mode == 'front'
        else 'IWRL6844 4T4R Top-Down Depth Heatmap'
    )
    image = ax.imshow(
        heatmap,
        origin='lower',
        extent=(settings.x_range[0], settings.x_range[1], vertical_range[0], vertical_range[1]),
        aspect='auto',
        cmap=cmap,
        vmin=0.0,
        interpolation='nearest',
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(value_label(settings.value_mode))
    ax.set_title(title)
    ax.set_xlabel('X left/right (m)')
    ax.set_ylabel(vertical_label)
    ax.grid(color='white', alpha=0.18, linewidth=0.5)
    status = ax.text(
        0.01,
        1.01,
        'waiting for frames...',
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=9,
    )
    fig.tight_layout()
    return fig, image, status, heatmap


def update_plot(fig, image, status, heatmap: np.ndarray, frame_index: int, points: np.ndarray, settings: ViewerSettings) -> None:
    image.set_data(heatmap)
    vmax = max(float(np.percentile(heatmap, 99.0)), 1.0)
    image.set_clim(0.0, vmax)
    _vertical_axis, vertical_range, vertical_label = projection_axis(settings)
    cfg_name = os.path.basename(settings.cfg_path)
    status.set_text(
        f'{VIEWER_VERSION} | cfg {cfg_name} | '
        f'{settings.view_mode} | color {value_label(settings.value_mode)} | '
        f'frame {frame_index} | points {points.shape[0]} | '
        f'X {settings.x_range[0]:.1f}..{settings.x_range[1]:.1f} m | '
        f'{vertical_label.split()[0]} {vertical_range[0]:.1f}..{vertical_range[1]:.1f} m'
    )
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def run_viewer(settings: ViewerSettings) -> int:
    print(f'[INFO] Config file: {settings.cfg_path}')
    print(f'[INFO] Config serial port: {settings.port_cfg}')
    print(f'[INFO] Data serial port: {settings.port_data}')
    print(f'[INFO] send_config: {settings.send_config}')
    print(f'[INFO] view_mode: {settings.view_mode}')
    print(f'[INFO] value_mode: {settings.value_mode}')
    print(f'[INFO] bins: {settings.bins_x}x{settings.bins_y}')
    print(f'[INFO] fade_step: {settings.fade_step}, fade_floor: {settings.fade_floor}, update_hz: {settings.update_hz}')
    print(f'[INFO] ROI: X{settings.x_range} Y{settings.y_range} Z{settings.z_range}')

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

    capture = None
    try:
        capture = RadarUARTCapture(uart_settings)
        if settings.send_config:
            capture.send_config()
        else:
            capture.data_port.reset_input_buffer()
            capture.byte_buffer.clear()

        fig, image, status, heatmap = init_plot(settings)
        frame_index = 0
        update_interval_s = 1.0 / settings.update_hz if settings.update_hz > 0 else 0.0
        next_update_at = 0.0

        while settings.frames < 0 or frame_index < settings.frames:
            frame_data = capture.read_frame(timeout_s=0.05)
            if frame_data is None:
                plt.pause(0.001)
                if not plt.fignum_exists(fig.number):
                    break
                continue

            frame_index += 1
            points = points_in_roi(frame_data.points, settings)
            fade_heatmap(heatmap, settings)
            add_points_to_heatmap(heatmap, points, settings)

            now = time.monotonic()
            if now >= next_update_at:
                update_plot(fig, image, status, heatmap, frame_index, points, settings)
                next_update_at = now + update_interval_s

            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        print('\n[INFO] Stopped by user.')
    finally:
        if capture is not None:
            capture.close()
        plt.ioff()

    print('[INFO] Viewer closed.')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Show pseudo-depth heatmaps from IWRL6844 UART detected points.'
    )
    parser.add_argument('--cfg', default=DEFAULT_CFG_PATH, help='Radar cfg file path or file name under cfg/')
    parser.add_argument('--port_cfg', default=DEFAULT_CONFIG_PORT, help='Config serial port')
    parser.add_argument('--port_data', default=DEFAULT_DATA_PORT, help='Data serial port')
    parser.add_argument('--baudrate_cfg', type=int, default=DEFAULT_BAUDRATE_CFG, help='Config serial baudrate')
    parser.add_argument('--baudrate_data', type=int, default=DEFAULT_BAUDRATE_DATA, help='Data serial baudrate')
    parser.add_argument('--frames', type=int, default=-1, help='Number of frames to view; -1 runs until closed')
    parser.add_argument('--no_send_config', action='store_false', dest='send_config', help='Do not send radar cfg before reading data')
    parser.set_defaults(send_config=True)
    parser.add_argument('--x_range', default=f'{DEFAULT_DISPLAY_X[0]},{DEFAULT_DISPLAY_X[1]}', help='X range in meters: min,max')
    parser.add_argument('--y_range', default=f'{DEFAULT_DISPLAY_Y[0]},{DEFAULT_DISPLAY_Y[1]}', help='Y/depth range in meters: min,max')
    parser.add_argument('--z_range', default=f'{DEFAULT_DISPLAY_Z[0]},{DEFAULT_DISPLAY_Z[1]}', help='Z range in meters: min,max')
    parser.add_argument('--bins_x', type=int, default=120, help='Horizontal heatmap bins')
    parser.add_argument('--bins_y', type=int, default=120, help='Depth/height heatmap bins')
    parser.add_argument('--fade_step', type=float, default=0.35, help='Subtract this value from every heatmap cell each frame; higher values shorten point trails')
    parser.add_argument('--fade_floor', type=float, default=0.05, help='Clear heatmap cells below this value after fading to reduce render load')
    parser.add_argument('--update_hz', type=float, default=8.0, help='Plot refresh rate')
    parser.add_argument(
        '--view',
        choices=['topdown', 'front'],
        default='topdown',
        help='Projection view: topdown uses X/Y, front uses X/Z like a sparse pseudo-IR image',
    )
    parser.add_argument(
        '--value_mode',
        choices=['depth', 'intensity', 'occupancy', 'doppler', 'nearest'],
        default='depth',
        help='Heatmap value: Y depth, SNR intensity, hit count, speed, or nearer-is-brighter depth',
    )
    parser.add_argument('--moving_only', action='store_true', help='Keep only points above --min_abs_doppler')
    parser.add_argument('--min_abs_doppler', type=float, default=0.2870, help='Moving-point Doppler threshold')
    parser.add_argument('--min_intensity', type=float, default=None, help='Drop points below this intensity/SNR value')
    parser.add_argument(
        '--intensity_mode',
        choices=['snr_db', 'snr_raw', 'snr_norm'],
        default=INTENSITY_MODE,
        help='How side-info SNR is converted by the existing UART parser',
    )
    args = parser.parse_args()

    cfg_path = resolve_cfg_path(args.cfg)
    if not os.path.isfile(cfg_path):
        print(f'[ERROR] Config file not found: {cfg_path}')
        return 1

    if args.bins_x <= 0 or args.bins_y <= 0:
        print('[ERROR] bins_x and bins_y must be positive.')
        return 1
    if args.fade_step < 0.0:
        print('[ERROR] fade_step must be non-negative.')
        return 1
    if args.fade_floor < 0.0:
        print('[ERROR] fade_floor must be non-negative.')
        return 1

    settings = ViewerSettings(
        cfg_path=cfg_path,
        port_cfg=args.port_cfg,
        port_data=args.port_data,
        baudrate_cfg=args.baudrate_cfg,
        baudrate_data=args.baudrate_data,
        frames=args.frames,
        send_config=args.send_config,
        x_range=parse_range(args.x_range, 'x'),
        y_range=parse_range(args.y_range, 'y'),
        z_range=parse_range(args.z_range, 'z'),
        bins_x=args.bins_x,
        bins_y=args.bins_y,
        fade_step=args.fade_step,
        fade_floor=args.fade_floor,
        update_hz=args.update_hz,
        view_mode=args.view,
        value_mode=args.value_mode,
        moving_only=args.moving_only,
        min_abs_doppler=args.min_abs_doppler,
        min_intensity=args.min_intensity,
        intensity_mode=args.intensity_mode,
        snr_norm_mean=SNR_NORM_MEAN,
        snr_norm_std=SNR_NORM_STD,
        side_info_db_limit=SIDE_INFO_DB_LIMIT,
    )
    return run_viewer(settings)


if __name__ == '__main__':
    raise SystemExit(main())
