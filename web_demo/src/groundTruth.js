import { part1QuestionIds } from "./omrUtils.js";

export const STORAGE_KEY = "omr_ground_truth_v1";

export function emptyGroundTruth() {
  return {
    version: 1,
    updatedAt: null,
    sheets: {},
  };
}

export function normalizeGroundTruth(input) {
  const payload = input?.groundTruth ?? input;
  if (!payload || typeof payload !== "object") {
    return emptyGroundTruth();
  }

  const normalized = {
    version: Number(payload.version ?? 1),
    updatedAt: payload.updatedAt ?? null,
    sheets: {},
  };

  if (Array.isArray(payload.sheets)) {
    for (const sheet of payload.sheets) {
      const imageId = sheet?.image_id ?? sheet?.imageId;
      if (imageId) {
        normalized.sheets[imageId] = normalizeSheetTruth(sheet);
      }
    }
    return normalized;
  }

  if (payload.sheets && typeof payload.sheets === "object") {
    for (const [imageId, sheet] of Object.entries(payload.sheets)) {
      normalized.sheets[imageId] = normalizeSheetTruth({
        image_id: imageId,
        ...sheet,
      });
    }
  }

  return normalized;
}

export function mergeGroundTruth(...items) {
  const merged = emptyGroundTruth();
  for (const item of items) {
    const normalized = normalizeGroundTruth(item);
    merged.updatedAt = normalized.updatedAt ?? merged.updatedAt;
    for (const [imageId, sheetTruth] of Object.entries(normalized.sheets)) {
      merged.sheets[imageId] = mergeSheetTruth(merged.sheets[imageId] ?? {}, sheetTruth);
    }
  }
  return merged;
}

export function mergeLatestGroundTruth(serverGroundTruth, localGroundTruth) {
  const server = normalizeGroundTruth(serverGroundTruth);
  const local = normalizeGroundTruth(localGroundTruth);
  const serverTime = Date.parse(server.updatedAt ?? "") || 0;
  const localTime = Date.parse(local.updatedAt ?? "") || 0;
  return localTime >= serverTime ? mergeGroundTruth(server, local) : mergeGroundTruth(local, server);
}

export function loadLocalGroundTruth() {
  try {
    return normalizeGroundTruth(JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "null"));
  } catch {
    return emptyGroundTruth();
  }
}

export function saveLocalGroundTruth(groundTruth) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalizeGroundTruth(groundTruth)));
}

export function buildGroundTruthSheet(sheet) {
  const answers = {};
  for (const questionId of part1QuestionIds(sheet.answers)) {
    answers[questionId] = sheet.answers?.[questionId]?.selected ?? "";
  }

  const part2Answers = {};
  for (let question = 1; question <= 8; question += 1) {
    for (const statement of ["a", "b", "c", "d"]) {
      const questionId = `II_${String(question).padStart(3, "0")}_${statement}`;
      part2Answers[questionId] = sheet.part2?.answers?.[questionId]?.selected ?? "";
    }
  }

  const part3Answers = {};
  for (let question = 1; question <= 6; question += 1) {
    const questionId = `III_${String(question).padStart(3, "0")}`;
    part3Answers[questionId] = sheet.part3?.answers?.[questionId]?.value ?? "";
  }

  return {
    image_id: sheet.image_id,
    file_name: sheet.file_name,
    identity: {
      sbd: sheet.identity?.sbd?.value ?? "",
      exam_code: sheet.identity?.exam_code?.value ?? "",
    },
    answers,
    part2: {
      answers: part2Answers,
    },
    part3: {
      answers: part3Answers,
    },
  };
}

export function materializeGroundTruth(sheets, groundTruth) {
  const normalized = normalizeGroundTruth(groundTruth);
  const next = {
    version: 1,
    updatedAt: new Date().toISOString(),
    sheets: {},
  };

  for (const sheet of sheets ?? []) {
    if (!sheet?.image_id) {
      continue;
    }
    const base = buildGroundTruthSheet(sheet);
    next.sheets[sheet.image_id] = mergeSheetTruth(base, normalized.sheets[sheet.image_id]);
  }

  return next;
}

export function getGroundTruthValue(groundTruth, sheet, row) {
  if (!sheet || row?.type !== "field") {
    return "";
  }
  const sheetTruth = normalizeGroundTruth(groundTruth).sheets[sheet.image_id];
  if (!sheetTruth) {
    return row.canonicalValue ?? "";
  }

  const section = row.field.section;
  const key = row.field.key;
  if (section === "identity") {
    return sheetTruth.identity?.[key] ?? row.canonicalValue ?? "";
  }
  if (section === "part1") {
    return sheetTruth.answers?.[key] ?? row.canonicalValue ?? "";
  }
  if (section === "part2") {
    return sheetTruth.part2?.answers?.[key] ?? row.canonicalValue ?? "";
  }
  if (section === "part3") {
    return sheetTruth.part3?.answers?.[key] ?? row.canonicalValue ?? "";
  }
  return row.canonicalValue ?? "";
}

export function setGroundTruthValue(groundTruth, sheet, row, value) {
  const normalized = normalizeGroundTruth(groundTruth);
  const sheetTruth = mergeSheetTruth(
    buildGroundTruthSheet(sheet),
    normalized.sheets[sheet.image_id]
  );
  const next = {
    ...normalized,
    updatedAt: new Date().toISOString(),
    sheets: {
      ...normalized.sheets,
      [sheet.image_id]: sheetTruth,
    },
  };

  const cleanValue = value ?? "";
  if (row.field.section === "identity") {
    next.sheets[sheet.image_id] = {
      ...sheetTruth,
      identity: {
        ...sheetTruth.identity,
        [row.field.key]: cleanValue,
      },
    };
  } else if (row.field.section === "part1") {
    next.sheets[sheet.image_id] = {
      ...sheetTruth,
      answers: {
        ...sheetTruth.answers,
        [row.field.key]: cleanValue,
      },
    };
  } else if (row.field.section === "part2") {
    next.sheets[sheet.image_id] = {
      ...sheetTruth,
      part2: {
        ...sheetTruth.part2,
        answers: {
          ...sheetTruth.part2?.answers,
          [row.field.key]: cleanValue,
        },
      },
    };
  } else if (row.field.section === "part3") {
    next.sheets[sheet.image_id] = {
      ...sheetTruth,
      part3: {
        ...sheetTruth.part3,
        answers: {
          ...sheetTruth.part3?.answers,
          [row.field.key]: cleanValue,
        },
      },
    };
  }

  return next;
}

export function resetSheetGroundTruth(groundTruth, sheet) {
  const normalized = normalizeGroundTruth(groundTruth);
  const { [sheet.image_id]: _removed, ...remainingSheets } = normalized.sheets;
  return {
    ...normalized,
    updatedAt: new Date().toISOString(),
    sheets: remainingSheets,
  };
}

export function countChangedRows(groundTruth, sheet, rows) {
  return rows.filter((row) => {
    if (row.type !== "field") {
      return false;
    }
    return String(getGroundTruthValue(groundTruth, sheet, row) ?? "") !== String(row.canonicalValue ?? "");
  }).length;
}

function normalizeSheetTruth(sheet) {
  return {
    image_id: sheet.image_id ?? sheet.imageId ?? "",
    file_name: sheet.file_name ?? sheet.fileName ?? "",
    identity: {
      sbd: sheet.identity?.sbd ?? "",
      exam_code: sheet.identity?.exam_code ?? sheet.identity?.examCode ?? "",
    },
    answers: { ...(sheet.answers ?? {}) },
    part2: {
      answers: { ...(sheet.part2?.answers ?? {}) },
    },
    part3: {
      answers: { ...(sheet.part3?.answers ?? {}) },
    },
  };
}

function mergeSheetTruth(base, override) {
  if (!override) {
    return normalizeSheetTruth(base);
  }
  const normalizedBase = normalizeSheetTruth(base);
  const normalizedOverride = normalizeSheetTruth(override);
  return {
    ...normalizedBase,
    ...normalizedOverride,
    identity: {
      ...normalizedBase.identity,
      ...normalizedOverride.identity,
    },
    answers: {
      ...normalizedBase.answers,
      ...normalizedOverride.answers,
    },
    part2: {
      answers: {
        ...normalizedBase.part2.answers,
        ...normalizedOverride.part2.answers,
      },
    },
    part3: {
      answers: {
        ...normalizedBase.part3.answers,
        ...normalizedOverride.part3.answers,
      },
    },
  };
}
