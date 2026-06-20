# cham_thi

Pipeline OMR trong repo này tách bước căn layout khỏi bước đọc đáp án. Layout detector học 4 kênh `page_mask`, `marker_heatmap`, `bubble_heatmap`, `red_grid_mask`; kết quả marker/bubble sau đó được dùng để warp, local-align và decode bằng rule-based/classifier.

## Cài Đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest baseline\tests
```

## Tạo Dataset Layout V2

Nguồn dữ liệu:

- Scan gốc: `baseline/phieu_thi/*.jpg`
- Metadata: `baseline/data/labels/sheets.jsonl`
- Template: `baseline/data/labels/template_tnthpt.json`

Dataset mới mặc định ghi vào `data_train/layout_v0_v2`, giữ nguyên schema `layout_v0` nhưng tăng augmentation. Mỗi scan gồm `1 clean + 6 augment`.

```powershell
python baseline\scripts\build_layout_dataset.py --overwrite
```

Lệnh tương đương, ghi rõ tham số:

```powershell
python baseline\scripts\build_layout_dataset.py `
  --output-dir ..\data_train\layout_v0_v2 `
  --augmentations-per-scan 6 `
  --overwrite
```

Augmentation hiện mạnh hơn bản cũ: rotation tới khoảng `±30°`, perspective/crop lệch hơn, elastic warp mạnh hơn, occlude marker với xác suất `50%` và tối đa `4` marker, có occlusion một phần, contrast/brightness/noise/shadow mạnh hơn, blur, motion blur và JPEG quality thấp.

Smoke dataset để kiểm tra nhanh:

```powershell
python baseline\scripts\build_layout_dataset.py `
  --output-dir ..\data_train\layout_v0_smoke `
  --limit-base 3 `
  --augmentations-per-scan 1 `
  --overwrite
```

Với split mặc định `120/25/22` base scans và `1 + 6` sample/scan, nếu không có scan bị skip thì số sample kỳ vọng là:

```text
train: 840
val:   175
test:  154
total: 1169
```

## Train Layout Detector

Default train đã được tăng cấu hình:

```text
dataset-dir:   data_train/layout_v0_v2
output-dir:    baseline/reports/layout_v0_runs/unet_v2
epochs:        50
base_channels: 48
pos_weights:   3,50,10,6
image_size:    416 x 596
scheduler:     cosine annealing
```

Thứ tự `pos_weights` là:

```text
page_mask, marker_heatmap, bubble_heatmap, red_grid_mask
```

`marker_heatmap` đã tăng từ `30` lên `50` để giảm false negative marker.

Smoke train:

```powershell
python baseline\scripts\train_layout_model.py `
  --dataset-dir data_train\layout_v0_smoke `
  --output-dir baseline\reports\layout_v0_runs\smoke `
  --epochs 1 `
  --batch-size 1 `
  --image-width 208 `
  --image-height 298 `
  --limit-train 8 `
  --limit-val 4 `
  --overwrite
```

Train nhanh ở `416 x 596`:

```powershell
python baseline\scripts\train_layout_model.py `
  --dataset-dir data_train\layout_v0_v2 `
  --output-dir baseline\reports\layout_v0_runs\unet_v2 `
  --epochs 50 `
  --batch-size 4 `
  --image-width 416 `
  --image-height 596 `
  --base-channels 48 `
  --pos-weights 3,50,10,6 `
  --overwrite
```

Train full-size theo đúng aspect của dataset (`832 x 1192`). Nên dùng GPU và giảm batch nếu thiếu VRAM:

```powershell
python baseline\scripts\train_layout_model.py `
  --dataset-dir data_train\layout_v0_v2 `
  --output-dir baseline\reports\layout_v0_runs\unet_v2_full `
  --epochs 50 `
  --batch-size 1 `
  --image-width 832 `
  --image-height 1192 `
  --base-channels 48 `
  --pos-weights 3,50,10,6 `
  --device cuda `
  --overwrite
```

Nếu không có GPU, bỏ `--device cuda` và bắt đầu với `416 x 596`, `--batch-size 1` hoặc `2`.

Output train:

```text
baseline/reports/layout_v0_runs/unet_v2/
  args.json
  history.jsonl
  best_model.pt
  last_model.pt
  best_metrics.json
  last_metrics.json
```

## Eval Checkpoint

Eval trên test split:

```powershell
python baseline\scripts\evaluate_layout_model.py `
  --dataset-dir data_train\layout_v0_v2 `
  --checkpoint baseline\reports\layout_v0_runs\unet_v2\best_model.pt `
  --output-dir baseline\reports\layout_v0_eval\unet_v2_test `
  --split test `
  --marker-threshold 0.25 `
  --max-marker-peaks 40 `
  --overwrite
```

Metric cần xem:

- `pixel.marker_heatmap.dice`: chất lượng heatmap marker.
- `peaks.markers.recall`: marker có đủ không.
- `peaks.markers.mean_distance_px`: marker lệch bao nhiêu pixel.
- `peaks.bubbles.recall`: tâm bubble có đủ không.
- `pixel.red_grid_mask.dice`: model có bám khung/lưới không.

## Predict Và Extract

Predict overlay layout trên ảnh scan:

```powershell
python baseline\scripts\predict_layout_model.py `
  --checkpoint baseline\reports\layout_v0_runs\unet_v2\best_model.pt `
  --input-glob "baseline/phieu_thi/*.jpg" `
  --output-dir baseline\reports\layout_v0_infer\unet_v2_best `
  --marker-threshold 0.25 `
  --max-marker-peaks 40 `
  --overwrite
```

Extract batch, dùng layout detector mới:

```powershell
python baseline\scripts\extract_sheets.py `
  --run-dir reports\extract_unet_v2 `
  --all `
  --layout-checkpoint baseline\reports\layout_v0_runs\unet_v2\best_model.pt `
  --layout-marker-threshold 0.25 `
  --layout-max-marker-peaks 40 `
  --layout-marker-match-tolerance 55 `
  --bubble-classifier
```

Extract một ảnh:

```powershell
python baseline\scripts\extract_sheet_json.py baseline\phieu_thi\phieu_thi_001.jpg `
  --layout-checkpoint baseline\reports\layout_v0_runs\unet_v2\best_model.pt `
  --layout-marker-threshold 0.25 `
  --layout-max-marker-peaks 40 `
  --layout-marker-match-tolerance 55 `
  --bubble-classifier `
  --output reports\extract_one\phieu_thi_001.json
```

## Ghi Chú Vận Hành

- Nên giữ dataset cũ ở `data_train/layout_v0`; dataset cải tiến dùng `data_train/layout_v0_v2`.
- `best_model.pt` thường dùng cho eval/extract; `last_model.pt` chỉ dùng khi cần xem checkpoint cuối.
- Nếu eval thiếu marker, thử giảm `--marker-threshold` xuống `0.20`.
- Nếu nhiều peak đúng nhưng match thiếu marker ở scan skew mạnh, tăng `--layout-marker-match-tolerance` lên `60`.
