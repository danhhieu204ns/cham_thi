import React, { useEffect, useMemo, useState } from "react";
import {
  Download,
  Eye,
  EyeOff,
  RefreshCcw,
  Save,
  Upload,
} from "lucide-react";
import {
  assetUrl,
  BASE_URL,
  buildAnswerRows,
  getOverlayBubbles,
  identityText,
  navUrl,
  readStatusLabel,
  reviewText,
  statusClass,
  withBase,
} from "./omrUtils.js";
import {
  countChangedRows,
  getGroundTruthValue,
  loadLocalGroundTruth,
  materializeGroundTruth,
  mergeLatestGroundTruth,
  resetSheetGroundTruth,
  saveLocalGroundTruth,
  setGroundTruthValue,
} from "./groundTruth.js";

export function App({ initialPage }) {
  const page = initialPage === "upload" ? "upload" : "results";
  return (
    <>
      <Topbar page={page} />
      {page === "upload" ? <UploadPage /> : <ResultsPage />}
    </>
  );
}

function Topbar({ page }) {
  return (
    <header className="topbar">
      <a className="brand" href={BASE_URL}>
        OMR Extraction
      </a>
      <nav className="main-nav" aria-label="Trang">
        <a className={page === "results" ? "active" : ""} href={navUrl("results")}>
          Ket qua
        </a>
        <a className={page === "upload" ? "active" : ""} href={navUrl("upload")}>
          Upload trich xuat
        </a>
      </nav>
    </header>
  );
}

function ResultsPage() {
  const [data, setData] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [overlay, setOverlay] = useState(true);
  const [groundTruth, setGroundTruth] = useState({ version: 1, updatedAt: null, sheets: {} });
  const [groundTruthReady, setGroundTruthReady] = useState(false);
  const [status, setStatus] = useState({ message: "Dang tai du lieu...", tone: "working" });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const response = await fetch(withBase("data/demo_data.json"), { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const demoData = await response.json();
        const serverGroundTruth = await fetchServerGroundTruth();
        const localGroundTruth = loadLocalGroundTruth();
        const mergedGroundTruth = mergeLatestGroundTruth(serverGroundTruth, localGroundTruth);

        if (cancelled) {
          return;
        }

        setData(demoData);
        setSelectedId(demoData.sheets?.[0]?.image_id ?? null);
        setGroundTruth(mergedGroundTruth);
        setGroundTruthReady(true);
        setStatus({ message: "San sang", tone: "ok" });
      } catch (error) {
        if (!cancelled) {
          setStatus({ message: `Khong doc duoc du lieu demo: ${error.message}`, tone: "error" });
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (groundTruthReady) {
      saveLocalGroundTruth(groundTruth);
    }
  }, [groundTruth, groundTruthReady]);

  const sheets = useMemo(
    () =>
      [...(data?.sheets ?? [])].sort((left, right) =>
        left.file_name.localeCompare(right.file_name)
      ),
    [data]
  );

  const selectedSheet = useMemo(() => {
    if (!selectedId) {
      return sheets[0] ?? null;
    }
    return sheets.find((sheet) => sheet.image_id === selectedId) ?? sheets[0] ?? null;
  }, [selectedId, sheets]);

  const answerRows = useMemo(() => buildAnswerRows(selectedSheet), [selectedSheet]);
  const changedCount = selectedSheet ? countChangedRows(groundTruth, selectedSheet, answerRows) : 0;

  function handleSelectSheet(imageId) {
    setSelectedId(imageId);
  }

  function handleGroundTruthChange(row, value) {
    setGroundTruth((current) => setGroundTruthValue(current, selectedSheet, row, value));
    setStatus({ message: "Da cap nhat ground truth trong trinh duyet", tone: "ok" });
  }

  function handleResetSheet() {
    if (!selectedSheet) {
      return;
    }
    setGroundTruth((current) => resetSheetGroundTruth(current, selectedSheet));
    setStatus({ message: "Da dua phieu ve gia tri doc ban dau", tone: "ok" });
  }

  async function handleSaveGroundTruth() {
    if (!data?.sheets?.length) {
      return;
    }

    const payload = materializeGroundTruth(data.sheets, groundTruth);
    saveLocalGroundTruth(payload);
    setStatus({ message: "Dang luu ground truth...", tone: "working" });

    try {
      const response = await fetch("/api/ground-truth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await response.json().catch(() => null);
      if (!response.ok || result?.status !== "ok") {
        throw new Error(result?.error ?? `HTTP ${response.status}`);
      }
      setGroundTruth(payload);
      setStatus({ message: "Da luu ground truth", tone: "ok" });
    } catch (error) {
      setGroundTruth(payload);
      setStatus({
        message: `Da giu local, chua luu duoc server: ${error.message}`,
        tone: "error",
      });
    }
  }

  function handleExportGroundTruth() {
    const payload = materializeGroundTruth(data?.sheets ?? [], groundTruth);
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "ground_truth.json";
    link.click();
    URL.revokeObjectURL(url);
    setStatus({ message: "Da tao file ground truth", tone: "ok" });
  }

  return (
    <main className="page">
      <section className="result-layout">
        <aside className="sheet-list-panel" aria-label="Danh sach phieu">
          <div className="list-title">
            <h2>Danh sach phieu</h2>
            <span className="list-count">{sheets.length}</span>
          </div>
          <div className="sheet-list">
            {sheets.map((sheet) => {
              const rows = buildAnswerRows(sheet);
              const sheetChangedCount = countChangedRows(groundTruth, sheet, rows);
              return (
                <SheetRow
                  key={sheet.image_id}
                  sheet={sheet}
                  active={sheet.image_id === selectedSheet?.image_id}
                  changedCount={sheetChangedCount}
                  onSelect={handleSelectSheet}
                />
              );
            })}
          </div>
        </aside>

        <section className="sheet-detail" aria-label="Chi tiet phieu">
          <div className="detail-head">
            <div className="detail-title">
              <h1>{selectedSheet?.file_name ?? "Ket qua"}</h1>
              <p className={`muted status-${status.tone}`}>
                {selectedSheet ? `${identityText(selectedSheet)} | Da sua ${changedCount}` : status.message}
              </p>
              {selectedSheet ? (
                <p className={`muted status-${status.tone}`}>{status.message}</p>
              ) : null}
            </div>
            <div className="detail-actions">
              <OverlayToggle checked={overlay} onChange={setOverlay} />
              <button className="secondary-button" type="button" onClick={handleResetSheet}>
                <RefreshCcw aria-hidden="true" />
                <span>Dat lai</span>
              </button>
              <button className="secondary-button" type="button" onClick={handleExportGroundTruth}>
                <Download aria-hidden="true" />
                <span>Xuat JSON</span>
              </button>
              <button className="primary-button compact" type="button" onClick={handleSaveGroundTruth}>
                <Save aria-hidden="true" />
                <span>Luu</span>
              </button>
            </div>
          </div>

          <div className="detail-grid">
            <SheetPreview
              sheet={selectedSheet}
              imageSrc={assetUrl(selectedSheet?.warped_path || selectedSheet?.source_path)}
              overlay={overlay}
              bubbleSpecs={data?.bubble_specs}
              templateSize={data?.template?.canonical_size}
            />

            <section className="answers-panel" aria-label="Ket qua doc">
              <h2>Ket qua doc phieu</h2>
              <AnswerTable
                rows={answerRows}
                sheet={selectedSheet}
                groundTruth={groundTruth}
                editable
                onChange={handleGroundTruthChange}
              />
            </section>
          </div>
        </section>
      </section>
    </main>
  );
}

function SheetRow({ sheet, active, changedCount, onSelect }) {
  const changed = changedCount > 0;
  return (
    <button
      type="button"
      className={`sheet-row${active ? " active" : ""}${changed ? " changed" : ""}`}
      onClick={() => onSelect(sheet.image_id)}
    >
      <span>
        <strong>{sheet.file_name}</strong>
        <span>{identityText(sheet)}</span>
      </span>
      <b className="status-pill">{changed ? `Sua ${changedCount}` : reviewText(sheet)}</b>
    </button>
  );
}

function UploadPage() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState({ message: "San sang", tone: "ok" });
  const [loading, setLoading] = useState(false);
  const [overlay, setOverlay] = useState(true);
  const [result, setResult] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) {
      setStatus({ message: "Chua chon anh phieu.", tone: "error" });
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    setStatus({ message: "Dang trich xuat...", tone: "working" });

    try {
      const response = await fetch("/api/extract", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => null);
      if (!payload) {
        throw new Error("API upload chua san sang. Chay web_demo/server.py de dung trang nay.");
      }
      if (!response.ok || payload.status !== "ok") {
        throw new Error(payload.error ?? `HTTP ${response.status}`);
      }
      setResult(payload);
      setStatus({ message: "Da trich xuat xong.", tone: "ok" });
    } catch (error) {
      setResult(null);
      setStatus({ message: `Khong trich xuat duoc: ${error.message}`, tone: "error" });
    } finally {
      setLoading(false);
    }
  }

  const overlaySheet = result?.overlay?.sheet ?? null;
  const answerRows = useMemo(() => buildAnswerRows(overlaySheet), [overlaySheet]);

  return (
    <main className="page">
      <section className="page-header">
        <div>
          <h1>Upload va trich xuat</h1>
          <p className={`muted status-${status.tone}`}>{status.message}</p>
        </div>
      </section>

      <section className="upload-panel">
        <form className="upload-form" onSubmit={handleSubmit}>
          <label className="file-picker">
            <span>Anh phieu</span>
            <input
              name="file"
              type="file"
              accept="image/*"
              required
              disabled={loading}
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
            <strong>{file?.name ?? "Chon file anh"}</strong>
          </label>
          <button className="primary-button" type="submit" disabled={loading}>
            <Upload aria-hidden="true" />
            <span>{loading ? "Dang trich xuat..." : "Trich xuat"}</span>
          </button>
        </form>
      </section>

      {result ? (
        <section className="sheet-detail" aria-label="Ket qua trich xuat">
          <div className="detail-head">
            <div>
              <h2>{result.fileName ?? "Ket qua"}</h2>
              <p className="muted">
                SBD {result.summary?.sbd ?? "-"} | Ma de {result.summary?.examCode ?? "-"} | Can xem{" "}
                {result.summary?.needReview ?? 0}
              </p>
            </div>
            <div className="detail-actions">
              <OverlayToggle checked={overlay} onChange={setOverlay} />
            </div>
          </div>

          <div className="detail-grid">
            <SheetPreview
              sheet={overlaySheet}
              imageSrc={result.warpedImageUrl || result.sourceImageUrl || ""}
              overlay={overlay}
              bubbleSpecs={result.overlay?.bubbleSpecs}
              templateSize={result.overlay?.templateSize}
              alt={result.fileName ?? "Anh phieu da xu ly"}
            />

            <section className="answers-panel" aria-label="Ket qua doc">
              <h2>Ket qua doc phieu</h2>
              <AnswerTable rows={answerRows} />
            </section>
          </div>
        </section>
      ) : null}
    </main>
  );
}

function OverlayToggle({ checked, onChange }) {
  const Icon = checked ? Eye : EyeOff;
  return (
    <label className="overlay-toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <Icon aria-hidden="true" />
      <span>Overlay</span>
    </label>
  );
}

function SheetPreview({ sheet, imageSrc, overlay, bubbleSpecs, templateSize, alt }) {
  const bubbles = useMemo(
    () => getOverlayBubbles(sheet, bubbleSpecs, templateSize),
    [sheet, bubbleSpecs, templateSize]
  );

  return (
    <figure className="sheet-preview">
      <div className="sheet-canvas">
        {imageSrc ? <img src={imageSrc} alt={alt ?? sheet?.file_name ?? "Anh phieu thi"} /> : null}
        <div className={`overlay-layer${overlay ? "" : " hidden"}`} aria-hidden="true">
          {bubbles.map((bubble) => (
            <span
              key={bubble.key}
              className={bubble.className}
              style={bubble.style}
              title={bubble.title}
            />
          ))}
        </div>
      </div>
    </figure>
  );
}

function AnswerTable({ rows, sheet, groundTruth, editable = false, onChange }) {
  if (!rows.length) {
    return (
      <div className="answer-table-wrap">
        <table className="answer-table">
          <tbody>
            <tr>
              <td className="empty-cell">Khong co du lieu doc phieu</td>
            </tr>
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div className="answer-table-wrap">
      <table className={`answer-table${editable ? " editable" : ""}`}>
        <thead>
          <tr>
            <th>Muc</th>
            <th>Gia tri doc</th>
            {editable ? <th>Ground truth</th> : null}
            <th>Trang thai doc</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) =>
            row.type === "section" ? (
              <tr key={row.key} className="section-row">
                <td colSpan={editable ? 4 : 3}>{row.title}</td>
              </tr>
            ) : (
              <AnswerRow
                key={row.key}
                row={row}
                sheet={sheet}
                groundTruth={groundTruth}
                editable={editable}
                onChange={onChange}
              />
            )
          )}
        </tbody>
      </table>
    </div>
  );
}

function AnswerRow({ row, sheet, groundTruth, editable, onChange }) {
  const groundTruthValue = editable ? getGroundTruthValue(groundTruth, sheet, row) : "";
  const changed = editable && String(groundTruthValue ?? "") !== String(row.canonicalValue ?? "");

  return (
    <tr className={changed ? "changed-row" : ""}>
      <td>{row.label}</td>
      <td>
        <span className={row.readValue ? "" : "empty-value"}>{row.readValue || "-"}</span>
      </td>
      {editable ? (
        <td className="ground-truth-cell">
          <GroundTruthInput row={row} value={groundTruthValue} onChange={(value) => onChange(row, value)} />
        </td>
      ) : null}
      <td>
        <StatusChip status={row.status} />
      </td>
    </tr>
  );
}

function GroundTruthInput({ row, value, onChange }) {
  if (row.editor?.type === "select") {
    return (
      <select
        className="ground-truth-input"
        value={value ?? ""}
        aria-label={`Ground truth ${row.label}`}
        onChange={(event) => onChange(event.target.value)}
      >
        {row.editor.options.map((option) => (
          <option key={option.value || "blank"} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    );
  }

  return (
    <input
      className="ground-truth-input"
      value={value ?? ""}
      inputMode={row.editor?.inputMode}
      aria-label={`Ground truth ${row.label}`}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

function StatusChip({ status }) {
  const label = readStatusLabel(status);
  if (!label) {
    return null;
  }
  return <span className={`chip ${statusClass(status)}`}>{label}</span>;
}

async function fetchServerGroundTruth() {
  try {
    const response = await fetch("/api/ground-truth", { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return response.json();
  } catch {
    return null;
  }
}
