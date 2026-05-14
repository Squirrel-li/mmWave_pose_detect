from __future__ import annotations

import copy
import os
from typing import Any

import yaml


DEFAULT_RADAR_CONFIG = {
	'radar': {
		'config_port': 'COM3',
		'data_port': 'COM4',
		'cfg_file': 'IWRL6844_4T4R_record.cfg',
		'frames': 200,
		'baudrate_cfg': 115200,
		'baudrate_data': 1250000,
	},
	'tlv': {
		'detected_points': 1,
		'side_info': 7,
		'frame_sync_word': '0201040306050807',
		'frame_length_offset': 12,
		'frame_header_size': 40,
		'point_data_size': 16,
		'side_info_data_size': 4,
	},
	'point_output': {
		'intensity_mode': 'snr_db',
		'snr_norm_mean': 20.0,
		'snr_norm_std': 10.0,
		'side_info_db_limit': 100.0,
	},
	'point_filter': {
		'remove_all_zero': True,
		'roi': {
			'enabled': False,
			'x_min': -1.0,
			'x_max': 1.0,
			'y_min': 0.0,
			'y_max': 3.0,
			'z_min': -1.0,
			'z_max': 1.0,
		},
	},
	'point_transform': {
		'height_normalization': {
			'enabled': False,
			'source_height_m': 1.90,
			'target_height_m': 1.75,
			'axes': ['x', 'z'],
			'floor_z_m': -1.0,
			'x_center_m': 0.0,
		},
	},
	'feature_map': {
		'max_points': 64,
		'dtype': 'float64',
		'truncate_before_sort': True,
		'sort_axes': ['x', 'y', 'z'],
	},
	'paths': {
		'default_file_class': 'test',
		'model_file': 'MARS.h5',
		'capture_prefix': 'radar_capture_',
		'default_feature_file': 'radar_capture_0.npy',
		'default_pointcloud_file': 'mars_pointcloud_0506_Both_upper_limb_extension.mat',
		'default_label_file': 'labels_test.npy',
	},
	'predict': {
		'batch_size': 256,
		'save_pred': None,
	},
	'triplet': {
		'file_class': 'reference',
		'feature_file': 'featuremap_test.npy',
		'label_file': 'labels_test.npy',
	},
	'conversion': {
		'mat_key': 'marsData',
		'all_files': False,
	},
	'display': {
		'point_cloud': {
			'x_min': -1.0,
			'x_max': 1.0,
			'y_min': 0.0,
			'y_max': 3.0,
			'z_min': -1.0,
			'z_max': 1.0,
		},
		'label': {
			'x_min': -1.0,
			'x_max': 1.0,
			'y_min': 0.0,
			'y_max': 3.0,
			'z_min': -1.0,
			'z_max': 1.0,
		},
		'scatter': {
			'cmap': 'turbo',
			'size': 45,
			'alpha': 0.95,
			'edge_color': 'k',
			'line_width': 0.15,
		},
	},
}


def project_root() -> str:
	return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def default_radar_config_path() -> str:
	return os.path.join(project_root(), 'cfg', 'radar_uart_config.yaml')


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
	merged = copy.deepcopy(base)
	for key, value in override.items():
		if isinstance(value, dict) and isinstance(merged.get(key), dict):
			merged[key] = _merge_config(merged[key], value)
		else:
			merged[key] = value
	return merged


def load_radar_config(path: str | None = None) -> dict[str, Any]:
	config_path = path or default_radar_config_path()
	if not os.path.isfile(config_path):
		return copy.deepcopy(DEFAULT_RADAR_CONFIG)

	with open(config_path, 'r', encoding='utf-8') as file_handle:
		loaded = yaml.safe_load(file_handle) or {}

	if not isinstance(loaded, dict):
		raise ValueError(f'Radar config must be a YAML object: {config_path}')

	return _merge_config(DEFAULT_RADAR_CONFIG, loaded)


def cfg_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
	value: Any = config
	for key in keys:
		if not isinstance(value, dict) or key not in value:
			return default
		value = value[key]
	return value


def cfg_range(config: dict[str, Any], section: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
	return (
		(
			float(cfg_get(config, 'display', section, 'x_min', default=-1.0)),
			float(cfg_get(config, 'display', section, 'x_max', default=1.0)),
		),
		(
			float(cfg_get(config, 'display', section, 'y_min', default=0.0)),
			float(cfg_get(config, 'display', section, 'y_max', default=3.0)),
		),
		(
			float(cfg_get(config, 'display', section, 'z_min', default=-1.0)),
			float(cfg_get(config, 'display', section, 'z_max', default=1.0)),
		),
	)


def as_bool(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
	return bool(value)


def ensure_suffix(path: str, suffix: str) -> str:
	return path if path.lower().endswith(suffix.lower()) else f'{path}{suffix}'


def resolve_under_root(path: str, *parents: str) -> str:
	if os.path.isabs(path):
		return os.path.normpath(path)
	return os.path.normpath(os.path.join(project_root(), *parents, path))


def resolve_cfg_path(cfg_path: str) -> str:
	root = project_root()
	if os.path.isabs(cfg_path) and os.path.isfile(cfg_path):
		return os.path.normpath(cfg_path)

	candidates = [
		cfg_path,
		os.path.join(root, 'cfg', cfg_path),
		os.path.join(root, 'cfg', os.path.basename(cfg_path)),
		os.path.join(root, 'get_pointcloud', 'cfg', os.path.basename(cfg_path)),
	]

	for candidate in candidates:
		candidate = os.path.normpath(candidate)
		if os.path.isfile(candidate):
			return candidate

	return os.path.normpath(os.path.join(root, 'cfg', os.path.basename(cfg_path)))
