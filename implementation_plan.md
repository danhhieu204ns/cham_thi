# Kế hoạch: Cải thiện Model Detect Markers

## Phân tích kết quả debug

### Tổng quan 4 phiếu test

| Phiếu | marker_source | Markers detected | match status | Vấn đề |
|-------|--------------|-----------------|--------------|--------|
| phieu_that_1 | `rule_based_fallback_after_layout_v0` | 6/10 | ok (5 matched) | ⚠️ Layout model miss `part1_left_top` — fallback sang rule-based |
| phieu_that_2 | `layout_v0` | 10/10 | ok (10 matched) | ✅ Tốt nhất, detect đủ cả 10 marker |
| phieu_that_3 | `layout_v0` | 10/10 | ok (10 matched) | ✅ Đầy đủ |
| phieu_that_4 | `layout_v0` | 7/10 | ok (7 matched) | ⚠️ Miss `part1_right_top`, `part1_right_bottom`, `part2_right_bottom` |

### Chi tiết Phiếu 1 — thất bại nghiêm trọng nhất

**Layout model** chỉ detect được 5/10 markers, **miss `part1_left_top`**:
- Layout model detect peak tại `(127.55, 769.25)` — đây là `part1_left_bottom`, không phải `part1_left_top`
- Không có peak nào cho `part1_left_top` (khoảng `x~160, y~461`)
- Fallback sang `rule_based` phát hiện thêm `part1_left_bottom` và `part1_left_top` nhờ threshold-based contour detection
- **`mean_marker_match_distance_px`: 7.97px, max: 39.85px** — cao hơn hẳn so với phiếu 2/3

**Marker source = `rule_based_fallback_after_layout_v0`** là dấu hiệu layout model hoạt động kém.

### Chi tiết Phiếu 4 — miss 3 marker bên phải

Layout model **miss 3 marker**:
- `part1_right_top` (khoảng `x~1002, y~438`)
- `part1_right_bottom` (khoảng `x~1032, y~734`)  
- `part2_right_bottom` (khoảng `x~1053, y~936`)

Nhìn vào `marker_peaks`: model detect được peak tại `(1002.18, 438.58)` và `(1032.25, 734.44)` nhưng **không match được vào template** — vì homography bootstrap sai và tolerance bị vượt quá.

### Root cause phân tích

```
Layout model → detect peaks (heatmap) → match to template → global warp → local align
         ↑ vấn đề ở đây nếu thiếu marker
```

**Hai dạng failure quan sát thấy:**

1. **Miss peak** (phiếu 1 – `part1_left_top`): Model không generate heatmap peak cho marker đó. Nguyên nhân:
   - Marker bị phủ khuất bởi nội dung gần (chữ ký/chú thích)
   - Training data thiếu augmentation case này
   - Marker gần edge bị crop hoặc scale lạ

2. **Peak detected nhưng không match** (phiếu 4 – 3 marker phải): Peak detect được nhưng bị loại trong bước matching. Nguyên nhân:
   - Scan bị skew/perspective nặng khiến bootstrap matrix sai
   - Tolerance 45px không đủ sau khi dùng chỉ 4 corners
   - Marker nằm ở góc phần scan tối hơn hoặc có nhiễu

**Kết quả downstream**: Khi thiếu marker → `warp_from_markers` dùng ít điểm hơn → global warp kém chính xác → **local align fail** vì bubble grid bị lệch quá nhiều.

---

## Kế hoạch train lại model

### Mục tiêu
- Tăng recall của `marker_heatmap` lên ≥ 0.90 val dice
- Giảm false negative trên các marker bị che khuất một phần hoặc ở góc scan nghiêng

### Hiện trạng dataset

| Thuộc tính | Giá trị |
|-----------|---------|
| Tổng samples | 668 |
| Train / Val / Test | 480 / 100 / 88 |
| Augmentations/scan | 3 |
| Base scans | ~167 scans |
| Markers per sample | 10 |
| Occlusion probability | 35% |
| Markers bị occlude per sample | 1–3 random |

**Best model** (epoch 10): `marker_heatmap` val dice = **0.8436** — còn thấp.

### Nguyên nhân val dice thấp

1. Dataset có `augmentations_per_scan = 3` — ít, chủ yếu nhìn cùng một scan gốc
2. Occlusion chỉ 35% và tối đa 3 marker — model chưa học đủ case một marker bị g隐 bởi nội dung thực
3. Không có augmentation về skew nặng (±24° hiện tại), scan nghiêng nhiều hơn trong thực tế
4. base_channels = 32 nhưng image_width=1192, height=832 — model có thể cần input lớn hơn hoặc kiến trúc sâu hơn

### Proposed changes

#### 1. Tăng augmentation dataset

**[MODIFY]** `build_layout_dataset.py` — tăng `augmentations_per_scan` từ 3 lên **6-8**, thêm augmentations aggressive hơn:
- Tăng rotation range lên ±30°
- Tăng occlusion probability lên 50%, cho phép occlude đến 4 marker
- Thêm heavy JPEG compression (quality 30-50)
- Thêm motion blur simulation

**Command để rebuild dataset:**
```bash
python -m omr.commands.build_layout_dataset \
  --augmentations-per-scan 6 \
  --output-dir ../data_train/layout_v0_v2 \
  --overwrite
```

#### 2. Train lại với hyperparameters tốt hơn

**[MODIFY]** `train_layout_model.py` — tăng pos_weight cho marker channel:
- `marker_heatmap` pos_weight: 30 → **50** (tăng penalty cho false negative)
- Tăng epochs: 30 → **50**
- Tăng base_channels: 32 → **48** (model capacity lớn hơn)
- Thêm learning rate scheduler (cosine annealing)

**Command train:**
```bash
python -m omr.commands.train_layout_model \
  --dataset-dir data_train/layout_v0_v2 \
  --output-dir baseline/reports/layout_v0_runs/unet_v2 \
  --epochs 50 \
  --base-channels 48 \
  --pos-weights 3,50,10,6 \
  --overwrite
```

#### 3. Cải thiện inference matching

**[MODIFY]** `layout_inference.py` — sau khi train xong:
- Giảm `marker_threshold` từ 0.35 → **0.25** (để detect thêm weak peaks)
- Tăng `max_marker_peaks` từ 30 → **40**
- Xem xét tăng `marker_match_tolerance_px` từ 45 → **55** cho phiếu bị skew nhiều

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Hiện tại dataset gốc dùng 167 scans thực tế. Có thêm scans thực mới nào có thể thêm vào training không? Scan mới đặc biệt từ các phiếu bị fail (phiếu 1, 4) sẽ rất có giá trị.

> [!IMPORTANT]
> **Q2**: Bạn có muốn train trên GPU không? Script hiện tại hỗ trợ `--device cuda`. Nếu có GPU sẵn, thời gian train sẽ giảm từ ~vài giờ (CPU) xuống ~15-30 phút.

> [!WARNING]
> **Q3**: Rebuild dataset mới (`layout_v0_v2`) sẽ overwrite data cũ nếu dùng cùng output-dir. Bạn muốn giữ dataset cũ không?

## Verification Plan

### Automated Tests
```bash
# Eval model mới trên test set
python -m omr.commands.evaluate_layout_model \
  --checkpoint baseline/reports/layout_v0_runs/unet_v2/best_model.pt \
  --dataset-dir data_train/layout_v0_v2
```

### Manual Verification
- Chạy lại debug flow trên 4 phiếu test để xem marker detection cải thiện không
- So sánh `matched_marker_count` trước/sau
- Kiểm tra phiếu 1 có còn dùng fallback `rule_based` không
