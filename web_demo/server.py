from __future__ import annotations

import argparse
import cgi
import json
import re
import shutil
import sys
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DEMO_ROOT = REPO_ROOT / "web_demo"
WEB_DEMO_DIST = WEB_DEMO_ROOT / "dist"
BASELINE_ROOT = REPO_ROOT / "baseline"
BASELINE_SRC = BASELINE_ROOT / "src"
GROUND_TRUTH_PATH = WEB_DEMO_ROOT / "data" / "ground_truth.json"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_baseline_path() -> None:
    path = str(BASELINE_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)


def json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def empty_ground_truth() -> dict:
    return {
        "version": 1,
        "updatedAt": None,
        "sheets": {},
    }


def normalize_ground_truth(payload: dict) -> dict:
    sheets = payload.get("sheets", {})
    if isinstance(sheets, list):
        sheets = {
            str(sheet.get("image_id") or sheet.get("imageId")): sheet
            for sheet in sheets
            if isinstance(sheet, dict) and (sheet.get("image_id") or sheet.get("imageId"))
        }
    if not isinstance(sheets, dict):
        sheets = {}

    return {
        "version": int(payload.get("version", 1) or 1),
        "updatedAt": payload.get("updatedAt") or datetime.now().isoformat(timespec="seconds"),
        "sheets": sheets,
    }


def read_ground_truth() -> dict:
    if not GROUND_TRUTH_PATH.exists():
        return empty_ground_truth()
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        return empty_ground_truth()
    return normalize_ground_truth(payload)


def write_ground_truth(payload: dict) -> dict:
    normalized = normalize_ground_truth(payload)
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = GROUND_TRUTH_PATH.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as output:
        json.dump(normalized, output, ensure_ascii=False, indent=2)
        output.write("\n")
    temp_path.replace(GROUND_TRUTH_PATH)
    return normalized


def safe_dist_path(relative: str) -> Path:
    parts: list[str] = []
    for part in PurePosixPath(relative).parts:
        if part in ("", "/", "."):
            continue
        if part == "..":
            return WEB_DEMO_DIST / "__invalid__"
        parts.append(part)
    return WEB_DEMO_DIST.joinpath(*parts)


def safe_upload_name(filename: str) -> str:
    name = Path(filename or "sheet.jpg").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".jpg"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("._")
    return f"{stem or 'sheet'}{suffix}"


def public_baseline_url(path_value: str | None) -> str | None:
    if not path_value:
        return None
    clean = str(path_value).replace("\\", "/")
    if clean.startswith("/"):
        return clean
    if clean.startswith("baseline/"):
        return f"/{clean}"
    return f"/baseline/{clean}"


def review_count(extracted: dict) -> int:
    total = len(extracted.get("part1", {}).get("review_items", []))
    for section in ("part2", "part3"):
        counts = extracted.get(section, {}).get("counts", {})
        total += int(counts.get("need_review", 0))
        total += int(counts.get("multi_mark", 0))

    identity = extracted.get("identity", {})
    for field in ("sbd", "exam_code"):
        status = identity.get(field, {}).get("status")
        if status and status != "accepted":
            total += 1
    return total


def extract_uploaded_sheet(input_path: Path, original_name: str) -> dict:
    ensure_baseline_path()

    from omr.sheet_pipeline import ExtractionThresholds, build_all_specs, extract_sheet
    from omr.template import canonical_size, load_template

    template_path = BASELINE_ROOT / "data" / "labels" / "template_tnthpt.json"
    template = load_template(template_path)

    warped_dir = BASELINE_ROOT / "data" / "uploads" / "warped"
    warped_dir.mkdir(parents=True, exist_ok=True)
    warped_path = warped_dir / f"{input_path.stem}_warped.jpg"

    extracted = extract_sheet(
        input_path,
        template,
        project_root=BASELINE_ROOT,
        thresholds=ExtractionThresholds(),
        warped_output_path=warped_path,
    )

    sbd = extracted.get("identity", {}).get("sbd", {})
    exam_code = extracted.get("identity", {}).get("exam_code", {})
    template_width, template_height = canonical_size(template)

    return {
        "status": "ok",
        "generatedAt": extracted.get("generated_at"),
        "fileName": original_name,
        "imageId": extracted.get("image_id"),
        "sourceImageUrl": public_baseline_url(extracted.get("source_path")),
        "warpedImageUrl": public_baseline_url(extracted.get("warp", {}).get("warped_path")),
        "summary": {
            "sbd": sbd.get("value"),
            "examCode": exam_code.get("value"),
            "needReview": review_count(extracted),
        },
        "overlay": {
            "templateSize": {
                "width": template_width,
                "height": template_height,
            },
            "bubbleSpecs": build_all_specs(template),
            "sheet": {
                "answers": extracted.get("part1", {}).get("answers", {}),
                "identity": extracted.get("identity", {}),
                "part2": extracted.get("part2", {}),
                "part3": extracted.get("part3", {}),
            },
        },
    }


class DemoRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def translate_path(self, path: str) -> str:
        route = urlparse(path).path
        if WEB_DEMO_DIST.is_dir():
            if route in {"/web_demo", "/web_demo/", "/web_demo/index.html"}:
                return str(WEB_DEMO_DIST / "index.html")
            if route == "/web_demo/upload.html":
                return str(WEB_DEMO_DIST / "upload.html")
            if route.startswith("/web_demo/assets/"):
                return str(safe_dist_path(route.removeprefix("/web_demo/")))
        return super().translate_path(path)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/ground-truth":
            self.handle_get_ground_truth()
            return
        super().do_GET()

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/extract":
            self.handle_extract()
            return
        if route == "/api/ground-truth":
            self.handle_save_ground_truth()
            return
        json_response(self, HTTPStatus.NOT_FOUND, {"status": "error", "error": "Unknown endpoint"})

    def handle_get_ground_truth(self) -> None:
        try:
            json_response(self, HTTPStatus.OK, {"status": "ok", "groundTruth": read_ground_truth()})
        except Exception as exc:  # noqa: BLE001 - surface API failures as JSON
            json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "error", "error": str(exc)})

    def handle_save_ground_truth(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            json_response(self, HTTPStatus.BAD_REQUEST, {"status": "error", "error": "Missing JSON body"})
            return
        if content_length > MAX_JSON_BYTES:
            json_response(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"status": "error", "error": "JSON too large"})
            return

        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object")
            ground_truth = payload.get("groundTruth", payload)
            if not isinstance(ground_truth, dict):
                raise ValueError("Expected ground truth object")
            saved = write_ground_truth(ground_truth)
            json_response(self, HTTPStatus.OK, {"status": "ok", "groundTruth": saved})
        except Exception as exc:  # noqa: BLE001 - surface API failures as JSON
            json_response(self, HTTPStatus.BAD_REQUEST, {"status": "error", "error": str(exc)})

    def handle_extract(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            json_response(self, HTTPStatus.BAD_REQUEST, {"status": "error", "error": "Missing upload body"})
            return
        if content_length > MAX_UPLOAD_BYTES:
            json_response(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"status": "error", "error": "File too large"})
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            json_response(self, HTTPStatus.BAD_REQUEST, {"status": "error", "error": "Expected multipart/form-data"})
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": str(content_length),
                },
            )
            field = form["file"] if "file" in form else None
            if field is None or not getattr(field, "filename", ""):
                json_response(self, HTTPStatus.BAD_REQUEST, {"status": "error", "error": "Missing file"})
                return

            upload_dir = BASELINE_ROOT / "data" / "uploads" / "raw"
            upload_dir.mkdir(parents=True, exist_ok=True)
            safe_name = safe_upload_name(field.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            input_path = upload_dir / f"{timestamp}_{safe_name}"
            with input_path.open("wb") as output:
                shutil.copyfileobj(field.file, output)

            payload = extract_uploaded_sheet(input_path, Path(field.filename).name)
            json_response(self, HTTPStatus.OK, payload)
        except Exception as exc:  # noqa: BLE001 - surface API failures as JSON
            json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "error", "error": str(exc)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the OMR web demo and upload API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    print(f"Web demo: http://{args.host}:{args.port}/web_demo/")
    if WEB_DEMO_DIST.is_dir():
        print(f"React build: {WEB_DEMO_DIST}")
    else:
        print("React build not found. Run: cd web_demo; npm install; npm run build")
    print("Stop: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
