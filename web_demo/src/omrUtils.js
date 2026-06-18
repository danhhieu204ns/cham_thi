export const OVERLAY_DIAMETER_PX = {
  identity: 30,
  part1: 30,
  part2: 30,
  part3: 28,
};

export const BASE_URL = import.meta.env.BASE_URL || "/web_demo/";

export function withBase(path = "") {
  const cleanPath = String(path).replace(/^\/+/, "");
  return `${BASE_URL}${cleanPath}`;
}

export function navUrl(page) {
  return page === "results" ? BASE_URL : `${BASE_URL}upload.html`;
}

export function assetUrl(path) {
  if (!path) {
    return "";
  }
  const cleanPath = String(path).replaceAll("\\", "/");
  if (
    cleanPath.startsWith("/") ||
    cleanPath.startsWith("http://") ||
    cleanPath.startsWith("https://")
  ) {
    return cleanPath;
  }
  const assetPath = cleanPath.startsWith("baseline/") ? cleanPath : `baseline/${cleanPath}`;
  return `/${assetPath}`;
}

export function buildAnswerRows(sheet) {
  if (!sheet) {
    return [];
  }

  return [
    sectionRow("identity", "Nhan dang"),
    ...identityRows(sheet),
    sectionRow("part1", "Phan I"),
    ...part1Rows(sheet),
    sectionRow("part2", "Phan II"),
    ...part2Rows(sheet),
    sectionRow("part3", "Phan III"),
    ...part3Rows(sheet),
  ];
}

function sectionRow(key, title) {
  return {
    type: "section",
    key: `section.${key}`,
    title,
  };
}

function fieldRow({ key, label, readValue, canonicalValue, status, field, editor }) {
  return {
    type: "field",
    key,
    label,
    readValue: readValue ?? "",
    canonicalValue: canonicalValue ?? "",
    status: status ?? "",
    field,
    editor,
  };
}

function identityRows(sheet) {
  const sbd = sheet.identity?.sbd ?? {};
  const examCode = sheet.identity?.exam_code ?? {};
  return [
    fieldRow({
      key: "identity.sbd",
      label: "SBD",
      readValue: sbd.value ?? "",
      canonicalValue: sbd.value ?? "",
      status: sbd.status ?? "",
      field: { section: "identity", key: "sbd" },
      editor: { type: "text", inputMode: "numeric" },
    }),
    fieldRow({
      key: "identity.exam_code",
      label: "Ma de",
      readValue: examCode.value ?? "",
      canonicalValue: examCode.value ?? "",
      status: examCode.status ?? "",
      field: { section: "identity", key: "exam_code" },
      editor: { type: "text", inputMode: "numeric" },
    }),
  ];
}

function part1Rows(sheet) {
  return part1QuestionIds(sheet.answers).map((questionId) => {
    const answer = sheet.answers?.[questionId] ?? {};
    const questionNumberText = String(answer.question_number ?? questionNumber(questionId)).padStart(2, "0");
    return fieldRow({
      key: `part1.${questionId}`,
      label: `I.${questionNumberText}`,
      readValue: answer.selected ?? "",
      canonicalValue: answer.selected ?? "",
      status: answer.status || answer.decode_status || "",
      field: { section: "part1", key: questionId },
      editor: {
        type: "select",
        options: [
          { value: "", label: "Trong" },
          { value: "A", label: "A" },
          { value: "B", label: "B" },
          { value: "C", label: "C" },
          { value: "D", label: "D" },
        ],
      },
    });
  });
}

export function part1QuestionIds(answers) {
  const ids = new Set(Object.keys(answers ?? {}));
  for (let number = 1; number <= 40; number += 1) {
    ids.add(`I_${String(number).padStart(3, "0")}`);
  }
  return [...ids].sort(questionSort);
}

function part2Rows(sheet) {
  const rows = [];
  for (let question = 1; question <= 8; question += 1) {
    for (const statement of ["a", "b", "c", "d"]) {
      const questionId = `II_${String(question).padStart(3, "0")}_${statement}`;
      const answer = sheet.part2?.answers?.[questionId] ?? {};
      rows.push(
        fieldRow({
          key: `part2.${questionId}`,
          label: `II.${question}${statement}`,
          readValue: part2Value(answer.selected),
          canonicalValue: answer.selected ?? "",
          status: answer.status ?? "",
          field: { section: "part2", key: questionId },
          editor: {
            type: "select",
            options: [
              { value: "", label: "Trong" },
              { value: "T", label: "Dung" },
              { value: "F", label: "Sai" },
            ],
          },
        })
      );
    }
  }
  return rows;
}

function part3Rows(sheet) {
  const rows = [];
  for (let question = 1; question <= 6; question += 1) {
    const questionId = `III_${String(question).padStart(3, "0")}`;
    const answer = sheet.part3?.answers?.[questionId] ?? {};
    rows.push(
      fieldRow({
        key: `part3.${questionId}`,
        label: `III.${question}`,
        readValue: answer.value ?? answer.raw_value ?? "",
        canonicalValue: answer.value ?? "",
        status: answer.status ?? "",
        field: { section: "part3", key: questionId },
        editor: { type: "text", inputMode: "decimal" },
      })
    );
  }
  return rows;
}

export function getOverlayBubbles(sheet, specs, templateSize) {
  if (!sheet || !Array.isArray(specs) || !templateSize) {
    return [];
  }

  const width = Number(templateSize.width ?? templateSize[0]);
  const height = Number(templateSize.height ?? templateSize[1]);
  if (!width || !height) {
    return [];
  }

  return specs.map((spec) => {
    const bbox = overlayBbox(spec);
    const info = getBubbleInfo(sheet, spec);
    return {
      key: spec.spec_id ?? `${spec.section}-${spec.question_id}-${spec.choice}-${spec.slot}`,
      className: overlayClass(spec, info),
      title: overlayTitle(spec, info),
      style: {
        left: `${(bbox[0] / width) * 100}%`,
        top: `${(bbox[1] / height) * 100}%`,
        width: `${((bbox[2] - bbox[0]) / width) * 100}%`,
        height: `${((bbox[3] - bbox[1]) / height) * 100}%`,
      },
    };
  });
}

function overlayBbox(spec) {
  const diameter = OVERLAY_DIAMETER_PX[spec.section] ?? 30;
  const [centerX, centerY] = spec.center ?? [null, null];
  if (centerX === null || centerY === null) {
    return spec.bbox;
  }
  const half = diameter / 2;
  return [centerX - half, centerY - half, centerX + half, centerY + half];
}

function getBubbleInfo(sheet, spec) {
  if (spec.section === "identity") {
    const group = (sheet.identity?.[spec.field]?.columns ?? [])[Number(spec.slot) - 1] ?? {};
    return {
      selected: group.selected,
      status: group.status,
      state: group.states?.[spec.choice] ?? {},
    };
  }

  if (spec.section === "part1") {
    const answer = sheet.answers?.[spec.question_id] ?? {};
    return {
      selected: answer.selected,
      status: answer.status || answer.decode_status,
      state: answer.states?.[spec.choice] ?? {},
    };
  }

  if (spec.section === "part2") {
    const answer = sheet.part2?.answers?.[spec.question_id] ?? {};
    return {
      selected: answer.selected,
      status: answer.status,
      state: answer.states?.[spec.choice] ?? {},
    };
  }

  if (spec.section === "part3") {
    const answer = sheet.part3?.answers?.[spec.question_id] ?? {};
    const group = part3Group(answer, spec);
    return {
      selected: group.selected,
      status: group.status,
      state: group.states?.[spec.choice] ?? {},
    };
  }

  return { selected: null, status: "blank", state: {} };
}

function part3Group(answer, spec) {
  if (spec.role === "sign") {
    return answer.sign ?? {};
  }
  if (spec.role === "comma") {
    return answer.comma ?? {};
  }
  return (answer.digits ?? [])[Number(spec.slot) - 1] ?? {};
}

function overlayClass(spec, info) {
  const classes = ["overlay-bubble", spec.section];
  if (spec.field) {
    classes.push(spec.field);
  }

  const selected = String(info.selected ?? "") === String(spec.choice);
  const prelabel = info.state?.prelabel;
  if (selected) {
    classes.push("selected", statusClass(info.status));
  } else if (prelabel === "filled") {
    classes.push("prelabel-filled");
  } else if (prelabel === "ambiguous" || prelabel === "invalid") {
    classes.push("prelabel-ambiguous");
  } else {
    classes.push("empty");
  }

  return classes.join(" ");
}

function overlayTitle(spec, info) {
  const selected = info.selected ?? "";
  const status = info.status ?? "";
  const label = spec.label ?? spec.choice;
  return `${spec.section} ${spec.question_id} ${label} | doc ${selected} | ${status}`;
}

export function identityText(sheet) {
  const sbd = sheet.identity?.sbd?.value ?? "-";
  const examCode = sheet.identity?.exam_code?.value ?? "-";
  return `SBD ${sbd} | Ma de ${examCode}`;
}

export function reviewText(sheet) {
  const total = countReview(sheet);
  return total > 0 ? `Can xem ${total}` : "OK";
}

export function countReview(sheet) {
  let total = sheet.review_items?.length ?? sheet.part1?.review_items?.length ?? 0;
  total += sheet.part2?.counts?.need_review ?? 0;
  total += sheet.part2?.counts?.multi_mark ?? 0;
  total += sheet.part3?.counts?.need_review ?? 0;
  total += sheet.part3?.counts?.multi_mark ?? 0;
  if (sheet.identity?.sbd?.status && sheet.identity.sbd.status !== "accepted") {
    total += 1;
  }
  if (sheet.identity?.exam_code?.status && sheet.identity.exam_code.status !== "accepted") {
    total += 1;
  }
  return total;
}

export function questionSort(left, right) {
  return questionNumber(left) - questionNumber(right);
}

export function questionNumber(questionId) {
  return Number(String(questionId).split("_")[1]) || 0;
}

export function part2Value(value) {
  if (value === "T") {
    return "Dung";
  }
  if (value === "F") {
    return "Sai";
  }
  return value ?? "";
}

export function readStatusLabel(status) {
  if (!status) {
    return "";
  }
  if (status === "accepted") {
    return "Da doc";
  }
  if (status === "blank") {
    return "Trong";
  }
  if (status === "need_review") {
    return "Can xem";
  }
  if (status === "multi_mark") {
    return "Nhieu chon";
  }
  if (status === "incomplete") {
    return "Thieu";
  }
  return String(status).replaceAll("_", " ");
}

export function statusClass(status) {
  return String(status ?? "blank").replaceAll("_", "-");
}
