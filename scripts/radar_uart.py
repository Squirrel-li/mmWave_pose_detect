"""Shared UART control and TLV parsing for TI mmWave radar scripts."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass

import numpy as np
import serial

from util.radar_config import cfg_get
from util.radar_config import load_radar_config

RADAR_CONFIG = load_radar_config()

TLV_DETECTED_POINTS = int(cfg_get(RADAR_CONFIG, 'tlv', 'detected_points', default=1))
TLV_SIDE_INFO = int(cfg_get(RADAR_CONFIG, 'tlv', 'side_info', default=7))
FRAME_SYNC_WORD = bytes.fromhex(str(cfg_get(RADAR_CONFIG, 'tlv', 'frame_sync_word', default='0201040306050807')))

FRAME_LENGTH_OFFSET = int(cfg_get(RADAR_CONFIG, 'tlv', 'frame_length_offset', default=12))
FRAME_HEADER_SIZE = int(cfg_get(RADAR_CONFIG, 'tlv', 'frame_header_size', default=40))
POINT_DATA_SIZE = int(cfg_get(RADAR_CONFIG, 'tlv', 'point_data_size', default=16))
SIDE_INFO_DATA_SIZE = int(cfg_get(RADAR_CONFIG, 'tlv', 'side_info_data_size', default=4))

IntensityMode = str


@dataclass(frozen=True)
class RadarUARTSettings:
	cfg_path: str
	port_cfg: str
	port_data: str
	baudrate_cfg: int
	baudrate_data: int
	intensity_mode: IntensityMode = 'snr_db'
	snr_norm_mean: float = 20.0
	snr_norm_std: float = 10.0
	side_info_db_limit: float = 100.0


@dataclass(frozen=True)
class PointFilterSettings:
	remove_all_zero: bool = True
	roi_enabled: bool = False
	x_min: float = -1.0
	x_max: float = 1.0
	y_min: float = 0.0
	y_max: float = 3.0
	z_min: float = -1.0
	z_max: float = 1.0


@dataclass(frozen=True)
class PointTransformSettings:
	height_normalize_enabled: bool = False
	source_height_m: float = 1.90
	target_height_m: float = 1.75
	height_axes: tuple[str, ...] = ('x', 'z')
	floor_z_m: float = -1.0
	x_center_m: float = 0.0


@dataclass
class FrameData:
	frame_number: int
	total_length: int
	num_detected_objects: int
	num_tlv: int
	points: np.ndarray
	_debug_tlv_bytes: bytes | None = None


def find_magic_word(buffer: bytearray, magic_word: bytes) -> int:
	return buffer.find(magic_word)


def side_info_to_db(raw_value: int, max_abs_db: float = 100.0) -> float:
	"""Convert TI OOB side-info raw int16 to dB with a guard for scaled firmware output."""
	value_db = float(raw_value) * 0.1
	if abs(value_db) > max_abs_db:
		value_db = float(raw_value) * 0.01
	return value_db


def convert_intensity(snr_raw: int, settings: RadarUARTSettings) -> float:
	"""Convert side-info SNR to the MARS intensity channel."""
	if settings.intensity_mode == 'snr_raw':
		return float(snr_raw)

	snr_db = side_info_to_db(snr_raw, settings.side_info_db_limit)
	if settings.intensity_mode == 'snr_norm':
		if settings.snr_norm_std == 0:
			raise ValueError('snr_norm_std cannot be 0')
		return (snr_db - settings.snr_norm_mean) / settings.snr_norm_std

	return snr_db


def point_filter_settings_from_config(config: dict | None = None) -> PointFilterSettings:
	config = config or RADAR_CONFIG
	return PointFilterSettings(
		remove_all_zero=bool(cfg_get(config, 'point_filter', 'remove_all_zero', default=True)),
		roi_enabled=bool(cfg_get(config, 'point_filter', 'roi', 'enabled', default=False)),
		x_min=float(cfg_get(config, 'point_filter', 'roi', 'x_min', default=-1.0)),
		x_max=float(cfg_get(config, 'point_filter', 'roi', 'x_max', default=1.0)),
		y_min=float(cfg_get(config, 'point_filter', 'roi', 'y_min', default=0.0)),
		y_max=float(cfg_get(config, 'point_filter', 'roi', 'y_max', default=3.0)),
		z_min=float(cfg_get(config, 'point_filter', 'roi', 'z_min', default=-1.0)),
		z_max=float(cfg_get(config, 'point_filter', 'roi', 'z_max', default=1.0)),
	)


def point_transform_settings_from_config(config: dict | None = None) -> PointTransformSettings:
	config = config or RADAR_CONFIG
	raw_axes = cfg_get(config, 'point_transform', 'height_normalization', 'axes', default=['x', 'z'])
	return PointTransformSettings(
		height_normalize_enabled=bool(cfg_get(config, 'point_transform', 'height_normalization', 'enabled', default=False)),
		source_height_m=float(cfg_get(config, 'point_transform', 'height_normalization', 'source_height_m', default=1.90)),
		target_height_m=float(cfg_get(config, 'point_transform', 'height_normalization', 'target_height_m', default=1.75)),
		height_axes=parse_transform_axes(raw_axes),
		floor_z_m=float(cfg_get(config, 'point_transform', 'height_normalization', 'floor_z_m', default=-1.0)),
		x_center_m=float(cfg_get(config, 'point_transform', 'height_normalization', 'x_center_m', default=0.0)),
	)


def parse_transform_axes(raw_axes) -> tuple[str, ...]:
	if isinstance(raw_axes, str):
		axes = tuple(axis.strip().lower() for axis in raw_axes.split(',') if axis.strip())
	else:
		axes = tuple(str(axis).strip().lower() for axis in raw_axes)
	for axis in axes:
		if axis not in ('x', 'y', 'z'):
			raise ValueError(f'Unsupported transform axis: {axis}')
	return axes


def filter_points(points: np.ndarray, settings: PointFilterSettings) -> np.ndarray:
	if points is None or points.shape[0] == 0:
		return np.zeros((0, 5), dtype=np.float64)

	filtered = points
	if settings.remove_all_zero:
		filtered = filtered[np.any(filtered != 0, axis=1)]

	if settings.roi_enabled and filtered.shape[0] > 0:
		mask = (
			(filtered[:, 0] >= settings.x_min) & (filtered[:, 0] <= settings.x_max) &
			(filtered[:, 1] >= settings.y_min) & (filtered[:, 1] <= settings.y_max) &
			(filtered[:, 2] >= settings.z_min) & (filtered[:, 2] <= settings.z_max)
		)
		filtered = filtered[mask]


	# if filtered.shape[0] > 0:
	# 	moving_mask = np.abs(filtered[:, 3]) >= 0.2870
	# 	filtered = filtered[moving_mask]
  
	return filtered


def transform_points(points: np.ndarray, settings: PointTransformSettings) -> np.ndarray:
	if points is None or points.shape[0] == 0:
		return np.zeros((0, 5), dtype=np.float64)

	if not settings.height_normalize_enabled:
		return points

	if settings.source_height_m <= 0:
		raise ValueError('source_height_m must be positive')

	scale = settings.target_height_m / settings.source_height_m
	transformed = points.copy()
	for axis in settings.height_axes:
		if axis == 'x':
			transformed[:, 0] = settings.x_center_m + (transformed[:, 0] - settings.x_center_m) * scale
		elif axis == 'z':
			transformed[:, 2] = settings.floor_z_m + (transformed[:, 2] - settings.floor_z_m) * scale
		elif axis == 'y':
			transformed[:, 1] *= scale
		else:
			raise ValueError(f'Unsupported height normalization axis: {axis}')
	return transformed


def select_mars_points(points: np.ndarray, max_points: int = 64, truncate_before_sort: bool = True) -> np.ndarray:
	"""Apply MARS point ordering: keep first Np points, then sort by x, y, z."""
	if points is None or points.shape[0] == 0:
		return np.zeros((0, 5), dtype=np.float64)

	selected = points
	if truncate_before_sort and selected.shape[0] > max_points:
		selected = selected[:max_points]

	if selected.shape[0] > 0:
		selected = selected[np.lexsort((selected[:, 2], selected[:, 1], selected[:, 0]))]

	if not truncate_before_sort and selected.shape[0] > max_points:
		selected = selected[:max_points]

	return selected


class MarsFeatureMapProcessor:
	"""Shared MARS preprocessing: filter points, select/pad 64 points, reshape to (8, 8, 5)."""

	def __init__(
		self,
		point_filter: PointFilterSettings,
		point_transform: PointTransformSettings | None = None,
		max_points: int = 64,
		truncate_before_sort: bool = True,
		dtype='float64',
	):
		self.point_filter = point_filter
		self.point_transform = point_transform or PointTransformSettings()
		self.max_points = int(max_points)
		self.truncate_before_sort = bool(truncate_before_sort)
		self.dtype = np.dtype(dtype)
		self.feature_shape = (8, 8, 5)
		self.required_points = self.feature_shape[0] * self.feature_shape[1]
		if self.max_points != self.required_points:
			raise ValueError(
				f'MARS feature map requires max_points={self.required_points} '
				f'for shape {self.feature_shape}, got {self.max_points}'
			)

	def filter(self, points: np.ndarray) -> np.ndarray:
		return filter_points(points, self.point_filter)

	def select(self, points: np.ndarray) -> np.ndarray:
		return select_mars_points(
			points,
			max_points=self.max_points,
			truncate_before_sort=self.truncate_before_sort,
		)

	def transform(self, points: np.ndarray) -> np.ndarray:
		return transform_points(points, self.point_transform)

	def points_to_featuremap(self, points: np.ndarray) -> np.ndarray:
		if points is None or points.shape[0] == 0:
			return np.zeros(self.feature_shape, dtype=self.dtype)

		selected = self.select(points).astype(self.dtype, copy=False)
		if selected.shape[0] < self.max_points:
			pad = np.zeros((self.max_points - selected.shape[0], 5), dtype=self.dtype)
			selected = np.vstack((selected, pad))
		return selected.reshape(self.feature_shape)

	def process_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		selected = self.select(self.transform(self.filter(points)))
		featuremap = self.points_to_featuremap(selected)
		return selected, featuremap

	def process_frame(self, frame_data: FrameData) -> tuple[np.ndarray, np.ndarray]:
		return self.process_points(frame_data.points)


def parse_frame(frame_bytes: bytes, settings: RadarUARTSettings) -> FrameData | None:
	"""Parse one complete UART frame into point-cloud data."""
	if len(frame_bytes) < FRAME_HEADER_SIZE:
		return None

	first_tlv_bytes = None
	offset = 8  # magic word
	try:
		offset += 4  # version
		total_length = struct.unpack_from('<I', frame_bytes, offset)[0]
		offset += 4  # totalLen
		offset += 4  # platform
		frame_number = struct.unpack_from('<I', frame_bytes, offset)[0]
		offset += 4  # frameNumber
		offset += 4  # timeCPUCycles
		num_detected_objects = struct.unpack_from('<I', frame_bytes, offset)[0]
		offset += 4
		num_tlv = struct.unpack_from('<I', frame_bytes, offset)[0]
		offset += 4
		offset += 4  # subFrameNumber
	except struct.error:
		return None

	if num_detected_objects == 0:
		return FrameData(
			frame_number=frame_number,
			total_length=total_length,
			num_detected_objects=0,
			num_tlv=num_tlv,
			points=np.zeros((0, 5), dtype=np.float64),
		)

	points = np.zeros((num_detected_objects, 5), dtype=np.float64)

	for _ in range(num_tlv):
		if offset + 8 > len(frame_bytes):
			break

		tlv_type, tlv_length = struct.unpack_from('<II', frame_bytes, offset)
		offset += 8
		tlv_end = offset + tlv_length

		if tlv_type == TLV_DETECTED_POINTS:
			for point_index in range(num_detected_objects):
				if offset + POINT_DATA_SIZE > len(frame_bytes):
					break
				if first_tlv_bytes is None and point_index == 0:
					first_tlv_bytes = bytes(frame_bytes[offset:offset + POINT_DATA_SIZE])
				points[point_index, 0:4] = struct.unpack_from('<ffff', frame_bytes, offset)
				offset += POINT_DATA_SIZE

		elif tlv_type == TLV_SIDE_INFO:
			for point_index in range(num_detected_objects):
				if offset + SIDE_INFO_DATA_SIZE > len(frame_bytes):
					break

				snr_raw, _noise_raw = struct.unpack_from('<hh', frame_bytes, offset)
				offset += SIDE_INFO_DATA_SIZE
				points[point_index, 4] = convert_intensity(snr_raw, settings)

		else:
			offset = tlv_end

		if offset < tlv_end:
			offset = tlv_end

	return FrameData(
		frame_number=frame_number,
		total_length=total_length,
		num_detected_objects=num_detected_objects,
		num_tlv=num_tlv,
		points=points,
		_debug_tlv_bytes=first_tlv_bytes,
	)


class RadarUARTCapture:
	def __init__(self, settings: RadarUARTSettings):
		self.settings = settings
		self.cfg_port = serial.Serial(
			settings.port_cfg,
			baudrate=settings.baudrate_cfg,
			timeout=0.1,
		)
		self.data_port = serial.Serial(
			settings.port_data,
			baudrate=settings.baudrate_data,
			timeout=0.1,
		)
		self.byte_buffer = bytearray()

	def _send_cli(self, cmd: str, delay: float = 0.12) -> str:
		if not cmd.endswith('\n'):
			cmd += '\n'
		self.cfg_port.write(cmd.encode('utf-8'))
		self.cfg_port.flush()
		time.sleep(delay)
		resp = b''
		while self.cfg_port.in_waiting:
			resp += self.cfg_port.read(self.cfg_port.in_waiting)
			time.sleep(0.1)
		return resp.decode(errors='ignore').strip()

	def send_config(self) -> None:
		print('[INFO] 送出 config...')
		self.cfg_port.reset_input_buffer()
		self.cfg_port.reset_output_buffer()
		self.data_port.reset_input_buffer()
		self.data_port.reset_output_buffer()
		self.byte_buffer.clear()

		resp = self._send_cli('sensorStop 0', delay=1.5)
		if resp:
			print('[CLI]', resp)

		with open(self.settings.cfg_path, 'r', encoding='utf-8') as file_handle:
			for line in file_handle:
				line = line.strip()
				if not line or line.startswith('%') or line.startswith('#'):
					continue
				if line.startswith('sensorStop'):
					continue

				delay = 1.5 if line.startswith('sensorStart') else 0.12
				resp = self._send_cli(line, delay=delay)
				print(f'  [CFG] {line}')
				if resp:
					print('[CLI]', resp)
				if 'error' in resp.lower():
					print(f'[WARN] 這行 cfg 可能失敗: {line}')

		time.sleep(0.5)
		self.data_port.reset_input_buffer()
		self.byte_buffer.clear()
		print('[INFO] Config 送出完成。\n')

	def read_frame(self, timeout_s: float | None = None) -> FrameData | None:
		start_time = time.monotonic()
		while True:
			if timeout_s is not None and time.monotonic() - start_time >= timeout_s:
				return None

			n_avail = self.data_port.in_waiting
			if n_avail > 0:
				self.byte_buffer.extend(self.data_port.read(n_avail))

			if len(self.byte_buffer) > 65536:
				del self.byte_buffer[:-32768]

			magic_idx = find_magic_word(self.byte_buffer, FRAME_SYNC_WORD)
			if magic_idx < 0:
				time.sleep(0.005)
				continue

			if len(self.byte_buffer) < magic_idx + FRAME_LENGTH_OFFSET + 4:
				time.sleep(0.005)
				continue

			total_len = struct.unpack_from('<I', self.byte_buffer, magic_idx + FRAME_LENGTH_OFFSET)[0]
			if total_len < FRAME_HEADER_SIZE or total_len > 65535:
				del self.byte_buffer[:magic_idx + len(FRAME_SYNC_WORD)]
				continue

			if len(self.byte_buffer) < magic_idx + total_len:
				time.sleep(0.005)
				continue

			frame_bytes = bytes(self.byte_buffer[magic_idx:magic_idx + total_len])
			del self.byte_buffer[:magic_idx + total_len]

			frame_data = parse_frame(frame_bytes, self.settings)
			if frame_data is not None:
				return frame_data

	def stop_sensor(self) -> None:
		try:
			if self.cfg_port and self.cfg_port.is_open:
				resp = self._send_cli('sensorStop 0', delay=1.5)
				if resp:
					print('[INFO] sensorStop 回應:', resp)
		except Exception as exc:
			print(f'[WARN] sensorStop 送出失敗: {exc}')

	def close(self) -> None:
		try:
			self.stop_sensor()
		finally:
			try:
				if self.data_port and self.data_port.is_open:
					self.data_port.reset_input_buffer()
					self.data_port.close()
			finally:
				if self.cfg_port and self.cfg_port.is_open:
					self.cfg_port.close()
