# mmWave Pose Detect

使用 TI mmWave radar UART 點雲資料進行 MARS 格式特徵轉換、即時點雲檢視、資料擷取與姿態估計的 Python 專案。

本專案主要流程是從 radar 的 config / data 兩個 UART port 讀取 frame，解析 TLV detected points 與 side info，轉成 MARS 論文相容的 `(8, 8, 5)` feature map，再使用 `model/MARS.h5` 做 19 joints 姿態推論。

## 功能

- 透過 UART 傳送 radar `.cfg` 設定並讀取即時 data stream。
- 解析 TI mmWave OOB TLV frame，包括 detected points 與 side info。
- 將點雲轉成 MARS feature map：`(N, 8, 8, 5)`。
- 使用 SNR 作為 MARS intensity channel 的近似值。
- 即時顯示 3D radar point cloud。
- 即時執行 MARS 姿態估計並顯示 skeleton。
- 將 `.mat` / `.npy` 點雲資料轉換為 feature map。
- 使用既有 feature map 與 `MARS.h5` 模型做離線推論與視覺化。

## 專案結構

```text
.
├── cfg/                         # Radar UART 與 mmWave profile 設定檔
│   ├── radar_uart_config.yaml    # 共用執行參數
│   └── *.cfg                     # TI mmWave radar profile
├── feature/                      # MARS feature map .npy 資料
│   ├── reference/
│   ├── standard/
│   └── test/
├── model/
│   └── MARS.h5                   # MARS 姿態估計模型
├── pointcloud/                   # 原始或參考點雲資料
│   ├── reference/
│   └── standard/
├── scripts/
│   ├── MARS_UART_capture.py       # UART 擷取並輸出 feature map
│   ├── MARS_UART_realtime_viewer.py
│   ├── MARS_UART_realtime_predict.py
│   ├── pc_to_featuremap_mars.py
│   ├── mars_predict.py
│   ├── mars_triplet.py
│   ├── show_feature.py
│   ├── test_uart_cfg.py
│   ├── radar_uart.py
│   └── util/
├── requirements.txt              # Python dependencies
├── LICENSE                       # GPL-3.0
└── 10334240.pdf                  # 參考文件
```

## 環境需求

### 作業系統

目前專案設定與指令以 Windows / PowerShell 為主，因為預設 serial port 使用 `COM19` / `COM20`。

### Python

建議使用 Python `3.7.x`。目前工作環境的 virtualenv metadata 顯示：

```text
Python 3.7.0
```

這點很重要，因為 `tensorflow==2.2.0`、`Keras==2.3.0`、`scipy==1.4.1` 等舊版科學運算套件與新版 Python 不一定相容。若使用 Python 3.10+，安裝很可能失敗。

### 硬體

- TI mmWave radar，設定檔包含 IWRL6844 / IWR6843 / xWR68xx profile。
- 兩個 serial port：
  - config port：傳送 `.cfg` 指令，預設 `COM19`
  - data port：接收 TLV binary stream，預設 `COM20`
- data baudrate 預設 `1250000`
- config baudrate 預設 `115200`

## 安裝

建議在專案根目錄建立 virtualenv：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果系統預設不是 Python 3.7，請明確使用 Python 3.7 建立環境，例如：

```powershell
py -3.7 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 依賴版本

完整 pinned dependencies 由 `requirements.txt` 管理：

```text
absl-py==2.1.0
astunparse==1.6.3
cachetools==4.2.4
certifi==2026.4.22
charset-normalizer==3.4.7
cycler==0.11.0
fonttools==4.38.0
gast==0.3.3
google-auth==1.35.0
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
grpcio==1.62.3
h5py==2.10.0
idna==3.10
importlib-metadata==6.7.0
joblib==1.3.2
Keras==2.3.0
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.2
kiwisolver==1.4.5
Markdown==3.4.4
MarkupSafe==2.1.5
matplotlib==3.5.3
numpy==1.21.6
oauthlib==3.2.2
opt-einsum==3.3.0
packaging==24.0
Pillow==9.5.0
protobuf==3.20.3
pyasn1==0.5.1
pyasn1-modules==0.3.0
PyOpenGL==3.1.7
pyparsing==3.1.4
PyQt5==5.15.10
PyQt5-Qt5==5.15.2
PyQt5-sip==12.13.0
pyqtgraph==0.12.4
pyserial==3.5
python-dateutil==2.9.0.post0
python-dotenv==0.21.1
PyYAML==6.0.1
requests==2.31.0
requests-oauthlib==2.0.0
rsa==4.9.1
scikit-learn==1.0.2
scipy==1.4.1
six==1.17.0
tensorboard==2.2.2
tensorboard-plugin-wit==1.8.1
tensorflow==2.2.0
tensorflow-estimator==2.2.0
termcolor==2.3.0
threadpoolctl==3.1.0
typing_extensions==4.7.1
urllib3==2.0.7
Werkzeug==2.2.3
wrapt==1.16.0
zipp==3.15.0
```

主要套件用途：

| 套件 | 用途 |
| --- | --- |
| `pyserial` | 與 radar config / data UART port 通訊 |
| `numpy` / `scipy` | 點雲、feature map、`.mat` / `.npy` 資料處理 |
| `tensorflow` / `Keras` / `h5py` | 載入與執行 `model/MARS.h5` |
| `PyQt5` / `pyqtgraph` / `PyOpenGL` | 即時 3D 點雲與姿態視覺化 |
| `matplotlib` | 離線資料與推論結果視覺化 |
| `PyYAML` | 載入 `cfg/radar_uart_config.yaml` |
| `python-dotenv` | 載入 `.env` 環境設定 |

## 共用設定

主要設定檔是：

```text
cfg/radar_uart_config.yaml
```

目前重要預設值：

```yaml
radar:
  config_port: COM19
  data_port: COM20
  cfg_file: IWRL6844_4T4R_record_high_accuracy.cfg
  frames: 200
  baudrate_cfg: 115200
  baudrate_data: 1250000

feature_map:
  max_points: 64
  dtype: float64
  truncate_before_sort: true
  sort_axes: [x, y, z]

paths:
  default_file_class: test
  model_file: MARS.h5
  default_feature_file: radar_capture_6.npy
```

如需更換 radar port 或 profile，優先修改 `cfg/radar_uart_config.yaml`：

```yaml
radar:
  config_port: COMx
  data_port: COMy
  cfg_file: your_profile.cfg
```

也可以在執行指令時用 CLI argument 覆蓋，例如：

```powershell
python scripts\MARS_UART_realtime_viewer.py --port_cfg COM19 --port_data COM20 --cfg IWRL6844_4T4R_record_high_accuracy.cfg
```

## Radar Profile

`cfg/` 目錄包含多個 radar `.cfg`：

- `IWRL6844_4T4R_record_high_accuracy.cfg`
- `IWRL6844_4T4R_record.cfg`
- `IWRL6844_4T4R_MARS_paper_like.cfg`
- `iwrl6844_uart_0.cfg`
- `iwr6843_clutter_removal.cfg`
- `xwr68xx_MARS_UART.cfg`
- `xwr68xx_MARS_UART_2.cfg`
- `xwr68xx_MARS_UART_3.cfg`
- `xwr68xx_MARS_UART_test.cfg`
- `xwr68xx_profile_2023_11_27_vitalsign_test_2T4R.cfg`
- `xwr68xx_profile_2026_04_15T09_10_07_261.cfg`
- `profile_4T4R_tdm_1843_style_output_filter.cfg`
- `template_1843.cfg`

目前共用設定預設使用：

```text
cfg/IWRL6844_4T4R_record_high_accuracy.cfg
```

## 資料格式

### UART frame

`scripts/radar_uart.py` 負責解析 UART binary stream。TLV 參數由 `cfg/radar_uart_config.yaml` 控制：

```yaml
tlv:
  detected_points: 1
  side_info: 7
  frame_sync_word: "0201040306050807"
  frame_length_offset: 12
  frame_header_size: 40
  point_data_size: 16
  side_info_data_size: 4
```

### Point cloud

每個 radar point 會整理成：

```text
[x, y, z, doppler, intensity]
```

目前 `intensity` 預設來自 side-info SNR：

```yaml
point_output:
  intensity_mode: snr_db
```

### MARS feature map

UART capture 與資料轉換輸出的 feature map shape：

```text
(N, 8, 8, 5)
```

每個 frame 的處理流程：

1. 取得點雲 `[x, y, z, doppler, intensity]`
2. 依設定決定是否套用 ROI filter
3. 保留最多 64 個點
4. 依 `x -> y -> z` 排序
5. 不足 64 點時補零
6. row-major reshape 成 `(8, 8, 5)`

## 常用指令

以下指令都在專案根目錄執行。

### 1. 測試 radar cfg 是否能輸出可解析點雲

```powershell
python scripts\test_uart_cfg.py
```

指定 profile 與 frame 數：

```powershell
python scripts\test_uart_cfg.py IWRL6844_4T4R_record_high_accuracy.cfg --frames 20 --duration 8
```

### 2. 即時顯示 UART 點雲

```powershell
python scripts\MARS_UART_realtime_viewer.py
```

常用參數：

```powershell
python scripts\MARS_UART_realtime_viewer.py --frames 200 --plot_hz 10
python scripts\MARS_UART_realtime_viewer.py --print_points
python scripts\MARS_UART_realtime_viewer.py --no_filter_roi
```

### 3. 擷取 UART 點雲並儲存 feature map

```powershell
python scripts\MARS_UART_capture.py
```

指定輸出檔案：

```powershell
python scripts\MARS_UART_capture.py --frames 200 --output feature\test\radar_capture_custom.npy
```

不送出 radar config，直接讀目前 radar data stream：

```powershell
python scripts\MARS_UART_capture.py --no_send_config
```

不顯示即時 3D preview：

```powershell
python scripts\MARS_UART_capture.py --no_show_plot
```

### 4. 即時姿態估計

```powershell
python scripts\MARS_UART_realtime_predict.py
```

指定模型、profile 與 port：

```powershell
python scripts\MARS_UART_realtime_predict.py --model model\MARS.h5 --cfg IWRL6844_4T4R_record_high_accuracy.cfg --port_cfg COM19 --port_data COM20
```

持續執行：

```powershell
python scripts\MARS_UART_realtime_predict.py --frames -1
```

### 5. 將 `.mat` 點雲轉成 MARS feature map

```powershell
python scripts\pc_to_featuremap_mars.py
```

指定輸入與輸出：

```powershell
python scripts\pc_to_featuremap_mars.py --input pointcloud\standard\mars_pointcloud_0506_squad.mat --output feature\standard\mars_pointcloud_0506_squad.npy
```

批次轉換指定資料夾類別：

```powershell
python scripts\pc_to_featuremap_mars.py --all_files true --file_class standard
```

`file_class` 可使用：

```text
test
standard
reference
```

### 6. 顯示 feature map 點雲

```powershell
python scripts\show_feature.py
```

指定檔案：

```powershell
python scripts\show_feature.py --input feature\test\radar_capture_6.npy
```

自動尋找最新 capture：

```powershell
python scripts\show_feature.py --auto true
```

### 7. 離線 MARS 推論

```powershell
python scripts\mars_predict.py
```

指定 feature map 與模型：

```powershell
python scripts\mars_predict.py --input feature\test\radar_capture_6.npy --model model\MARS.h5
```

儲存預測結果：

```powershell
python scripts\mars_predict.py --input feature\test\radar_capture_6.npy --save_pred feature\test\radar_capture_6_pred.npy
```

### 8. 三欄比較 demo

顯示 point cloud、MARS prediction、ground truth label：

```powershell
python scripts\mars_triplet.py
```

指定資料：

```powershell
python scripts\mars_triplet.py --input feature\reference\featuremap_test.npy --label feature\reference\labels_test.npy --model model\MARS.h5
```

## 常見參數

| 參數 | 說明 |
| --- | --- |
| `--cfg` | radar `.cfg` 檔案路徑或檔名 |
| `--port_cfg` | config serial port，例如 `COM19` |
| `--port_data` | data serial port，例如 `COM20` |
| `--baudrate_cfg` | config port baudrate，預設 `115200` |
| `--baudrate_data` | data port baudrate，預設 `1250000` |
| `--frames` | 擷取或推論 frame 數，部分腳本用 `-1` 表示持續執行 |
| `--filter_roi` | 啟用 ROI 篩選 |
| `--no_filter_roi` | 關閉 ROI 篩選 |
| `--model` | MARS `.h5` 模型路徑 |
| `--input` | 輸入 `.npy` 或 `.mat` |
| `--output` | 輸出 `.npy` |
| `--file_class` | 資料類別：`test`、`standard`、`reference` |

## 疑難排解

### 安裝 TensorFlow 失敗

確認使用 Python 3.7：

```powershell
python --version
```

`tensorflow==2.2.0` 不適合直接安裝在新版 Python。若目前是 Python 3.10+，請改用 Python 3.7 建立 `.venv`。

### 找不到 serial port

檢查 Windows 裝置管理員中的 COM port 編號，並更新：

```text
cfg/radar_uart_config.yaml
```

或在指令中指定：

```powershell
python scripts\MARS_UART_realtime_viewer.py --port_cfg COMx --port_data COMy
```

### 可以開 port 但沒有點雲

請依序檢查：

1. `--cfg` 是否對應目前 radar firmware / board。
2. `baudrate_data` 是否為 radar profile 設定的 data baudrate。
3. data port 與 config port 是否接反。
4. radar 是否已被其他程式占用。
5. 使用 `scripts\test_uart_cfg.py` 先短測 profile。

### 即時視窗沒有顯示或閃退

確認已安裝 GUI 相關依賴：

```text
PyQt5==5.15.10
pyqtgraph==0.12.4
PyOpenGL==3.1.7
```

若在遠端、無桌面環境或未啟用圖形介面，PyQt5 視窗可能無法正常啟動。

### 找不到模型

預設模型路徑：

```text
model/MARS.h5
```

若模型放在其他位置，使用：

```powershell
python scripts\mars_predict.py --model path\to\MARS.h5
```

## 開發備註

- 主要 UART parser 在 `scripts/radar_uart.py`。
- 共用路徑與資料類別邏輯在 `scripts/util/AbsDir.py`。
- 共用 YAML 設定讀取與 profile path resolve 在 `scripts/util/radar_config.py`。
- 即時工具多數會優先讀取 `cfg/radar_uart_config.yaml`，再由 CLI argument 覆蓋。
- feature map 預設使用 `float64`，與目前資料及模型流程一致。

## License

本專案使用 GPL-3.0。完整授權內容請見 `LICENSE`。
