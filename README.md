# cham_thi

Pipeline OMR trong repo này tách phần căn layout ra khỏi phần đọc đáp án. Model layout v0 học dự đoán mask/heatmap để tìm trang, marker, tâm bubble và khung lưới; sau đó OMR rule-based mới dùng các điểm này để căn chỉnh và decode.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Kiểm tra nhanh:

```powershell
python -m pytest baseline\tests
```

## Tạo lại dataset layout_v0

Nguồn đầu vào:

- 167 scan: `baseline/phieu_thi/*.jpg`
- metadata: `baseline/data/labels/sheets.jsonl`
- template canonical: `baseline/data/labels/template_tnthpt.json`

Tạo dataset đầy đủ hiện tại, mỗi phiếu gồm `1 clean + 3 augment`:

```powershell
python baseline\scripts\build_layout_dataset.py --augmentations-per-scan 3 --overwrite
```

Output mặc định:

```text
data_train/layout_v0/
  images/{train,val,test}/
  labels/{train,val,test}/
  masks/page_mask/{train,val,test}/
  masks/marker_heatmap/{train,val,test}/
  masks/bubble_heatmap/{train,val,test}/
  masks/red_grid_mask/{train,val,test}/
  splits/
  manifest.jsonl
  skipped.jsonl
  summary.json
```

Smoke test tạo ít dữ liệu:

```powershell
python baseline\scripts\build_layout_dataset.py `
  --output-dir ..\data_train\layout_v0_smoke `
  --limit-base 3 `
  --augmentations-per-scan 1 `
  --overwrite
```

Chạy hiện tại sinh ra `668` sample:

```text
train: 480
val:   100
test:   88
```

## Train layout detector

Model mặc định là U-Net nhỏ, input RGB, output 4 kênh:

```text
page_mask
marker_heatmap
bubble_heatmap
red_grid_mask
```

Smoke train để kiểm tra code:

```powershell
python baseline\scripts\train_layout_model.py `
  --output-dir baseline\reports\layout_v0_runs\smoke `
  --epochs 1 `
  --batch-size 1 `
  --image-width 208 `
  --image-height 298 `
  --limit-train 8 `
  --limit-val 4 `
  --overwrite
```

Train thực tế bước đầu:

```powershell
python baseline\scripts\train_layout_model.py `
  --output-dir baseline\reports\layout_v0_runs\unet_416 `
  --epochs 30 `
  --batch-size 4 `
  --image-width 416 `
  --image-height 596 `
  --base-channels 24 `
  --overwrite
```

Nếu không có GPU, dùng `--batch-size 1` hoặc `--batch-size 2`. Nếu có GPU còn dư VRAM, tăng dần lên `832 x 1192` hoặc `base-channels 32` sau khi bản `416 x 596` đã ổn.

Output train:

```text
baseline/reports/layout_v0_runs/unet_416/
  args.json
  history.jsonl
  best_model.pt
  last_model.pt
  best_metrics.json
  last_metrics.json
```

Metric chính trong lúc train là `val.mean_dice`. Với heatmap marker/bubble, Dice chỉ là tín hiệu huấn luyện; eval peak bên dưới mới quan trọng hơn cho alignment.

## Eval checkpoint

Eval trên test split:

```powershell
python baseline\scripts\evaluate_layout_model.py `
  --checkpoint baseline\reports\layout_v0_runs\unet_416\best_model.pt `
  --output-dir baseline\reports\layout_v0_eval\unet_416_test `
  --split test `
  --overwrite
```

Output:

```text
baseline/reports/layout_v0_eval/unet_416_test/
  metrics.json
  visuals/*_pred_overlay.jpg
```

Metric cần xem:

- `pixel.page_mask.dice`: model có tìm đúng vùng trang không.
- `pixel.red_grid_mask.dice`: model có bám khung/lưới không.
- `peaks.markers.recall`, `peaks.markers.mean_distance_px`: marker có đủ và gần đúng không.
- `peaks.bubbles.recall`, `peaks.bubbles.mean_distance_px`: tâm bubble có đủ và chính xác không.

Ngưỡng gợi ý ở size `416 x 596`:

```text
marker mean distance <= 3 px: tốt
bubble mean distance <= 2 px: tốt
bubble recall >= 0.98: bắt đầu dùng để fit grid
```

Ở size `832 x 1192`, các ngưỡng pixel có thể xấp xỉ gấp đôi.

## Hướng train tiếp theo

1. Train `416 x 596` trước để kiểm tra model có học đúng mask/heatmap không.
2. Eval trên `test`, mở các ảnh trong `visuals` để xem lỗi lệch bubble/marker.
3. Nếu recall thấp vì threshold, thử eval lại với `--bubble-threshold 0.15` hoặc `--marker-threshold 0.25`.
4. Nếu peak bám đúng nhưng thiếu ở ảnh crop mạnh, tăng `--augmentations-per-scan` khi tạo data.
5. Khi bản nhỏ ổn, train/fine-tune ở `832 x 1192` để lấy độ chính xác pixel tốt hơn.
6. Sau đó mới gắn inference layout detector vào local alignment của OMR rule-based.
