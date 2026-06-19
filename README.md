# OMR Extraction

Du an gom hai phan chinh:

- `baseline/`: pipeline OpenCV baseline de warp phieu va trich xuat SBD, ma de, phan I, phan II, phan III.
- `web_demo/`: demo React hien thi anh phieu, overlay bubble, ket qua doc tu `web_demo/data/demo_data.json`, upload trich xuat va chinh sua ground truth.

## Chay baseline

Yeu cau: Python 3.10+ tren Windows PowerShell.

```powershell
cd baseline
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\run_end_to_end.ps1
```

Co the chay nhanh tap mau template bang:

```powershell
.\scripts\run_end_to_end.ps1 -Sample
```

Ket qua baseline duoc ghi vao:

- `baseline/data/processed/results/sheet_extraction_baseline.jsonl`
- `baseline/data/processed/warped/sheet_extraction/`
- `baseline/data/processed/crops/sheet_extraction/`
- `baseline/reports/test_runs/<run_id>/`

Buoc cuoi cua script se cap nhat lai `web_demo/data/demo_data.json`.

## Trich xuat mot phieu ra JSON

Chay full pipeline cho mot anh phieu thi raw:

```powershell
cd baseline
python scripts\extract_sheet_json.py phieu_thi\Ly01.jpg --output data\processed\results\Ly01_full.json
```

JSON dau ra gom `identity.sbd`, `identity.exam_code`, `part1`, `part2`, `part3` va thong tin warp. Co the them `--warped-output` hoac `--crop-output-dir` neu can anh debug.

## Chay web demo

Web demo co 2 trang:

- `Ket qua`: xem ket qua doc phieu tu `web_demo/data/demo_data.json` va chinh sua gia tri dung lam ground truth.
- `Upload trich xuat`: upload mot anh phieu va chay baseline extraction.

Build React frontend:

```powershell
cd web_demo
npm install
npm run build
cd ..
```

Chay server rieng cua demo de dung duoc ca trang upload va luu ground truth:

```powershell
.\baseline\.venv\Scripts\python.exe web_demo\server.py
```

Mo trinh duyet tai:

```text
http://127.0.0.1:8000/web_demo/
```

Neu cong `8000` dang ban, dung cong khac:

```powershell
.\baseline\.venv\Scripts\python.exe web_demo\server.py --port 8001
```

Trang ket qua van co the doc bang server static, nhung trang upload can `web_demo/server.py` vi trinh duyet khong the tu goi Python baseline.

Ground truth duoc luu tai `web_demo/data/ground_truth.json` khi chay qua `web_demo/server.py`. Neu chi chay frontend dev server, cac chinh sua van duoc giu trong localStorage va co the xuat JSON tu giao dien.

Che do frontend dev:

```powershell
cd web_demo
npm run dev
```

Neu can dung upload trong che do dev, hay de `web_demo/server.py` chay o cong `8000` de Vite proxy `/api` va `/baseline`.
