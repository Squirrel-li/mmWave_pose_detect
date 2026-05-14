"""MARS UART 擷取工具：依照 MARS 論文格式錄製點雲並儲存 NPY。

輸出格式：
    (N, 8, 8, 5)

每個 frame 的處理流程：
    1. 從 UART TLV 取得點雲 [x, y, z, doppler, intensity]
    2. 不做 ROI 篩選，保留 out-of-range / ghost points
    3. 依 x -> y -> z 升冪排序
    4. 截斷或補零到 64 points
    5. row-major reshape 成 (8, 8, 5)

第 5 維 intensity 來源：
    TI xWRL6844 OOB point cloud 沒有直接輸出 MARS 論文中的 reflection intensity I_i，
    這裡使用 side-info SNR 作為 intensity channel 的可用近似值。
"""


#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from pointcloud_pyqtgraph import init_plot
from pointcloud_pyqtgraph import process_plot_events
from pointcloud_pyqtgraph import update_plot_points
from radar_uart import MarsFeatureMapProcessor
from radar_uart import PointFilterSettings
from radar_uart import point_filter_settings_from_config
from radar_uart import PointTransformSettings
from radar_uart import point_transform_settings_from_config
from radar_uart import parse_transform_axes
from radar_uart import RadarUARTCapture
from radar_uart import RadarUARTSettings
from util.AbsDir import AbsDir
from util.AbsDir import FileClass
from util.radar_config import cfg_get
from util.radar_config import cfg_range
from util.radar_config import load_radar_config
from util.radar_config import resolve_cfg_path

RADAR_CONFIG = load_radar_config()
POINT_FILTER_SETTINGS = point_filter_settings_from_config(RADAR_CONFIG)
POINT_TRANSFORM_SETTINGS = point_transform_settings_from_config(RADAR_CONFIG)

COM_PORT_CONFIG = str(cfg_get(RADAR_CONFIG, 'radar', 'config_port', default='COM3'))
COM_PORT_DATA = str(cfg_get(RADAR_CONFIG, 'radar', 'data_port', default='COM4'))
CFG_FILE = str(cfg_get(RADAR_CONFIG, 'radar', 'cfg_file', default='IWRL6844_4T4R_record.cfg'))
FRAMES = int(cfg_get(RADAR_CONFIG, 'radar', 'frames', default=200))
BAUDRATE_CFG = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_cfg', default=115200))
BAUDRATE_DATA = int(cfg_get(RADAR_CONFIG, 'radar', 'baudrate_data', default=1250000))
INTENSITY_MODE = str(cfg_get(RADAR_CONFIG, 'point_output', 'intensity_mode', default='snr_db'))
SNR_NORM_MEAN = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_mean', default=20.0))
SNR_NORM_STD = float(cfg_get(RADAR_CONFIG, 'point_output', 'snr_norm_std', default=10.0))
SIDE_INFO_DB_LIMIT = float(cfg_get(RADAR_CONFIG, 'point_output', 'side_info_db_limit', default=100.0))
MAX_POINTS = int(cfg_get(RADAR_CONFIG, 'feature_map', 'max_points', default=64))
FEATURE_DTYPE = str(cfg_get(RADAR_CONFIG, 'feature_map', 'dtype', default='float64'))
TRUNCATE_BEFORE_SORT = bool(cfg_get(RADAR_CONFIG, 'feature_map', 'truncate_before_sort', default=True))
DISPLAY_X, DISPLAY_Y, DISPLAY_Z = cfg_range(RADAR_CONFIG, 'point_cloud')

absDir = AbsDir()
path_project_root = absDir.path_project_root

IntensityMode = str


@dataclass(frozen=True)
class CaptureSettings:
    cfg_path: str
    port_cfg: str
    port_data: str
    frames: int
    out_path: str
    send_config: bool
    baudrate_cfg: int
    baudrate_data: int
    intensity_mode: IntensityMode = 'snr_db'
    snr_norm_mean: float = 20.0
    snr_norm_std: float = 10.0
    side_info_db_limit: float = 100.0
    point_filter: PointFilterSettings = POINT_FILTER_SETTINGS
    point_transform: PointTransformSettings = POINT_TRANSFORM_SETTINGS
    max_points: int = 64
    truncate_before_sort: bool = True
    display_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (DISPLAY_X, DISPLAY_Y, DISPLAY_Z)
    show_plot: bool = True
    plot_hz: float = 10.0
    dtype: str = 'float64'


# def resolve_cfg_path(cfg_path: str) -> str:
# 	"""支援只輸入檔名時，優先到專案 cfg 目錄找。"""
# 	if os.path.isabs(cfg_path) and os.path.isfile(cfg_path):
# 		return cfg_path

# 	candidates = [
# 		cfg_path,
# 		os.path.join(os.path.dirname(__file__), cfg_path),
# 		os.path.join(path_project_root, 'get_pointcloud', 'cfg', os.path.basename(cfg_path)),
# 	]

# 	for candidate in candidates:
# 		candidate = os.path.normpath(candidate)
# 		if os.path.isfile(candidate):
# 			return candidate

# 	return os.path.normpath(os.path.join(path_project_root, 'get_pointcloud', 'cfg', os.path.basename(cfg_path)))

def frame_to_featuremap(frame: np.ndarray, max_points: int = 64, truncate_before_sort: bool = True, dtype=np.float64) -> np.ndarray:
    """依 MARS 論文產生 (8,8,5) feature map。

    MARS 論文流程：
    - 每點 5 維：[x, y, z, Doppler, intensity]
    - 每 frame 統一為 64 x 5；不足補零，超過截斷
    - 保留雷達 frame 中前 64 個 reflected points
    - 依 x -> y -> z 升冪排序
    - row-major reshape 成 8 x 8 x 5

    ROI 與有效點篩選已由 shared radar_uart.filter_points() 控制。
    """
    processor = MarsFeatureMapProcessor(
        point_filter=PointFilterSettings(remove_all_zero=False),
        point_transform=PointTransformSettings(),
        max_points=max_points,
        truncate_before_sort=truncate_before_sort,
        dtype=dtype,
    )
    return processor.points_to_featuremap(frame)


def capture_to_npy(settings: CaptureSettings) -> int:
    print(f'[INFO] Config 檔案: {settings.cfg_path}')
    print(f'[INFO] Config serial port: {settings.port_cfg}')
    print(f'[INFO] Data serial port: {settings.port_data}')
    print(f'[INFO] send_config: {settings.send_config}')
    print(f'[INFO] 輸出檔案: {settings.out_path}')
    print(f'[INFO] intensity_mode: {settings.intensity_mode}')
    print(f'[INFO] filter_roi: {settings.point_filter.roi_enabled}')
    print(f'[INFO] height_normalization: {settings.point_transform.height_normalize_enabled}')
    if settings.point_transform.height_normalize_enabled:
        scale = settings.point_transform.target_height_m / settings.point_transform.source_height_m
        print(f'[INFO] height_scale: {settings.point_transform.source_height_m:.2f}m -> {settings.point_transform.target_height_m:.2f}m ({scale:.3f}), axes={settings.point_transform.height_axes}, floor_z={settings.point_transform.floor_z_m:.2f}, x_center={settings.point_transform.x_center_m:.2f}')
    print(f'[INFO] show_plot: {settings.show_plot}')
    if settings.show_plot:
        print(f'[INFO] display_ranges: X{settings.display_ranges[0]} Y{settings.display_ranges[1]} Z{settings.display_ranges[2]}')
        print(f'[INFO] plot_backend: pyqtgraph OpenGL, plot_hz: {settings.plot_hz}')
    print(f'[INFO] dtype: {settings.dtype}')

    dtype = np.float64 if settings.dtype == 'float64' else np.float32
    processor = MarsFeatureMapProcessor(
        point_filter=settings.point_filter,
        point_transform=settings.point_transform,
        max_points=settings.max_points,
        truncate_before_sort=settings.truncate_before_sort,
        dtype=dtype,
    )
    fmaps: list[np.ndarray] = []
    capture = None
    plot = None

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
            print('[INFO] 略過送出 config，直接開始讀取 data serial port。')

        if settings.show_plot:
            plot = init_plot(settings.display_ranges, title='MARS UART Capture 3D Preview')

        print(f'[INFO] 開始錄製，目標 {settings.frames} frames。')

        frame_count = 0
        plot_interval_s = 1.0 / settings.plot_hz if settings.plot_hz > 0 else 0.0
        next_plot_at = 0.0
        while settings.frames < 0 or frame_count < settings.frames:
            frame_data = capture.read_frame(timeout_s=0.02 if settings.show_plot else None)
            if frame_data is None:
                if plot is not None:
                    process_plot_events(plot)
                continue

            frame_count += 1
            valid_points, fmap = processor.process_frame(frame_data)
            fmaps.append(fmap.astype(dtype, copy=False))

            if plot is not None:
                import time
                now = time.monotonic()
                if now >= next_plot_at:
                    update_plot_points(valid_points, plot, settings.display_ranges)
                    next_plot_at = now + plot_interval_s

            if frame_count % 10 == 0:
                nonzero_points = np.count_nonzero(np.any(fmap.reshape(-1, 5) != 0, axis=1))
                print(f'[INFO] 已擷取 {frame_count} frames, 本幀有效點 {nonzero_points}, 累積 featuremap {len(fmaps)}')

    except KeyboardInterrupt:
        print('\n[INFO] 使用者中斷。')
    finally:
        if capture is not None:
            capture.close()

    if fmaps:
        captured = np.stack(fmaps).astype(dtype, copy=False)
    else:
        captured = np.zeros((0, 8, 8, 5), dtype=dtype)

    np.save(settings.out_path, captured)
    print(f'[INFO] 已儲存 featuremap NPY: {settings.out_path}')
    print(f'[INFO] 資料維度: {captured.shape} (frames,8,8,5)')
    print(f'[INFO] dtype: {captured.dtype}')

    if captured.size > 0:
        for ch in range(captured.shape[-1]):
            data = captured[..., ch]
            print(
                f'[INFO] channel {ch}: '
                f'min={np.min(data):.6f}, max={np.max(data):.6f}, '
                f'mean={np.mean(data):.6f}, std={np.std(data):.6f}, '
                f'nonzero={np.count_nonzero(data)}'
            )

    return 0


def auto_capture_path(folder: str, prefix: str = 'radar_capture_', ext: str = '.npy') -> str:
    idx = 0
    while True:
        path = os.path.join(folder, f'{prefix}{idx}{ext}')
        if not os.path.exists(path):
            return path
        idx += 1

def main() -> int:
    parser = argparse.ArgumentParser(description='MARS UART 擷取並輸出 MARS 論文格式 NPY')
    parser.add_argument('--cfg', default=CFG_FILE, help='Config 檔案名稱')
    parser.add_argument('--port_cfg', default=COM_PORT_CONFIG, help='Config serial port')
    parser.add_argument('--port_data', default=COM_PORT_DATA, help='Data serial port')
    parser.add_argument('--frames', type=int, default=FRAMES, help='要擷取的 frame 數，-1 代表持續錄製')
    parser.add_argument('--output', default=None, help='輸出 NPY 檔案路徑')
    parser.add_argument('--send_config', action='store_true', default=True, help='送出 radar cfg；預設啟用')
    parser.add_argument('--no_send_config', action='store_false', dest='send_config', help='不送出 radar cfg，直接擷取目前 radar 輸出的資料')
    parser.add_argument('--baudrate_cfg', type=int, default=BAUDRATE_CFG, help='Config serial baudrate')
    parser.add_argument('--baudrate_data', type=int, default=BAUDRATE_DATA, help='Data serial baudrate')
    parser.add_argument(
        '--intensity_mode',
        choices=['snr_db', 'snr_raw', 'snr_norm'],
        default=INTENSITY_MODE,
        help='第 5 維 intensity 的來源：snr_db=接近論文 Intensity；snr_raw=原始 side-info；snr_norm=給舊模型近似標準化',
    )
    parser.add_argument('--snr_norm_mean', type=float, default=SNR_NORM_MEAN, help='snr_norm 模式使用的 mean')
    parser.add_argument('--snr_norm_std', type=float, default=SNR_NORM_STD, help='snr_norm 模式使用的 std')
    parser.add_argument('--filter_roi', action='store_true', default=POINT_FILTER_SETTINGS.roi_enabled, help='啟用 ROI 篩選；預設跟 cfg/radar_uart_config.yaml 一致')
    parser.add_argument('--no_filter_roi', action='store_false', dest='filter_roi', help='關閉 ROI 篩選')
    parser.add_argument('--height_normalize', action='store_true', default=POINT_TRANSFORM_SETTINGS.height_normalize_enabled, help='啟用身高正規化縮放')
    parser.add_argument('--no_height_normalize', action='store_false', dest='height_normalize', help='關閉身高正規化縮放')
    parser.add_argument('--source_height_m', type=float, default=POINT_TRANSFORM_SETTINGS.source_height_m, help='身高正規化來源身高，例如 1.90')
    parser.add_argument('--target_height_m', type=float, default=POINT_TRANSFORM_SETTINGS.target_height_m, help='身高正規化目標身高，例如 1.75')
    parser.add_argument('--height_axes', default=','.join(POINT_TRANSFORM_SETTINGS.height_axes), help='要縮放的座標軸，例如 x,z 或 z；預設不縮放 y 深度')
    parser.add_argument('--floor_z_m', type=float, default=POINT_TRANSFORM_SETTINGS.floor_z_m, help='Z 軸縮放錨點；地板高度，縮放時地板不動')
    parser.add_argument('--x_center_m', type=float, default=POINT_TRANSFORM_SETTINGS.x_center_m, help='X 軸縮放中心線；預設 0.0')
    parser.add_argument('--show_plot', action='store_true', default=True, help='錄製時同步顯示 pyqtgraph 3D 點雲；預設啟用')
    parser.add_argument('--no_show_plot', action='store_false', dest='show_plot', help='錄製時不顯示 3D 點雲')
    parser.add_argument('--plot_hz', type=float, default=10.0, help='3D 點雲顯示更新頻率')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default=FEATURE_DTYPE, help='輸出 NPY dtype；模型訓練檔常見為 float64')
    args = parser.parse_args()

    cfg_path = resolve_cfg_path(args.cfg)

    # 自動命名：capture_0.npy, capture_1.npy ...
    if args.output is None:
        args.output = auto_capture_path(absDir.get_feature_dir_by_class(FileClass.TEST))

    settings = CaptureSettings(
        cfg_path=cfg_path,
        port_cfg=args.port_cfg,
        port_data=args.port_data,
        frames=args.frames,
        out_path=args.output,
        send_config=args.send_config,
        baudrate_cfg=args.baudrate_cfg,
        baudrate_data=args.baudrate_data,
        intensity_mode=args.intensity_mode,
        snr_norm_mean=args.snr_norm_mean,
        snr_norm_std=args.snr_norm_std,
        side_info_db_limit=SIDE_INFO_DB_LIMIT,
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
        display_ranges=(DISPLAY_X, DISPLAY_Y, DISPLAY_Z),
        show_plot=args.show_plot,
        plot_hz=args.plot_hz,
        dtype=args.dtype,
    )
    print("[INFO] 設定完成，開始擷取...")
    print("[INFO] 倒數3秒，請準備好雷達裝置...")
    import time
    for i in range(3, 0, -1):
        print(f"[INFO] {i}...")
        time.sleep(1)
    
    return capture_to_npy(settings)


if __name__ == '__main__':
    sys.exit(main())
