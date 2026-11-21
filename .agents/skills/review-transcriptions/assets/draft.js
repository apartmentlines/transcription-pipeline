export function randomId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, "0")).join("");
}

const clone = value => structuredClone(value);
const REVIEW_MARKER = "[REVIEW NEEDED]";
const equal = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const require = (condition, message) => { if (!condition) throw new Error(message); };
const index = (items, key) => {
  require(Array.isArray(items), "Expected a list of records.");
  const found = new Map();
  for (const item of items) {
    require(typeof item?.[key] === "string" && item[key], `Missing ${key}.`);
    require(!found.has(item[key]), `Duplicate ${key}: ${item[key]}`);
    found.set(item[key], item);
  }
  return found;
};
const fields = (object, expected) => require(
  object && typeof object === "object" && equal(Object.keys(object).sort(), [...expected].sort()),
  "Unexpected or missing feedback fields."
);

export function emptyDraft(corpus) {
  return {schema_version: 1, project: corpus.project, corpus_id: corpus.corpus_id,
    corpus_revision: corpus.revision, corpus_sha256: corpus.content_sha256,
    session_id: randomId(), revision: 0, reviewer: "",
    active_case: corpus.cases[0].id, cases: []};
}

export function proposal(base) {
  const passages = clone(base.passages);
  for (const passage of passages) {
    delete passage.speaker_change_before;
    delete passage.boundary_basis;
  }
  if (base.review.status === "pending") for (const passage of passages) {
    if (passage.uncertainty.trim() && !passage.uncertainty.includes(REVIEW_MARKER)) {
      passage.uncertainty = `${REVIEW_MARKER} ${passage.uncertainty}`;
    }
  }
  return {case_id: base.id, base_revision: base.revision, base_sha256: base.content_sha256,
    passages, notes: "", status: "draft", reviewed_at: null, reviewed_by: null};
}

export function hasUncertainty(passage) {
  return Boolean(passage.uncertainty.trim() ||
    passage.text.toLowerCase().includes("[unclear]"));
}

export function knownRole(passage) {
  const role = passage?.role_assignment?.role;
  return ["Operator", "Caller"].includes(role) ? role : null;
}

export function roleChange(previous, passage) {
  return Boolean(knownRole(previous) && knownRole(passage) && knownRole(previous) !== knownRole(passage));
}

export function reviewSnapshot(draft) {
  const snapshot = clone(draft);
  for (const edit of snapshot.cases) for (const passage of edit.passages) {
    delete passage.speaker_change_before;
    delete passage.boundary_basis;
  }
  return snapshot;
}

export function validateDraft(corpus, draft, requireRoles = true) {
  fields(draft, ["schema_version", "project", "corpus_id", "corpus_revision", "corpus_sha256",
    "session_id", "revision", "reviewer", "active_case", "cases"]);
  require(draft.schema_version === 1, "Unsupported draft format.");
  require(draft.project === corpus.project && draft.corpus_id === corpus.corpus_id &&
    draft.corpus_revision === corpus.revision && draft.corpus_sha256 === corpus.content_sha256,
    "This feedback belongs to a different reference revision. Your current work has been kept.");
  require(typeof draft.session_id === "string" && draft.session_id, "Missing review session.");
  require(Number.isInteger(draft.revision) && draft.revision >= 0, "Invalid draft revision.");
  require(typeof draft.reviewer === "string", "Invalid reviewer.");
  const bases = index(corpus.cases, "id");
  require(bases.has(draft.active_case), "Unknown active recording.");
  const issues = [];
  for (const [id, edit] of index(draft.cases, "case_id")) {
    fields(edit, ["case_id", "base_revision", "base_sha256", "passages", "notes", "status", "reviewed_at", "reviewed_by"]);
    const base = bases.get(id);
    require(base, `Unknown recording: ${id}`);
    require(edit.base_sha256 === base.content_sha256 && edit.base_revision === base.revision, `Stale recording: ${id}`);
    const original = index(base.passages, "id");
    const passages = index(edit.passages, "id");
    require(passages.size > 0, "The reference must contain at least one passage.");
    let position = 0;
    for (const [passageId, passage] of passages) {
      fields(passage, ["id", "text", "start", "end", "origin_ids", "uncertainty", "alternatives",
        ...["speaker_change_before", "boundary_basis"].filter(key => key in passage),
        ...("role_assignment" in passage ? ["role_assignment"] : [])]);
      if ("role_assignment" in passage) {
        const assignment = passage.role_assignment;
        try {
          fields(assignment, ["role", "reviewer", "assigned_at"]);
          require(["Operator", "Caller", "Unsure"].includes(assignment.role) &&
            typeof assignment.reviewer === "string" && assignment.reviewer.trim() &&
            typeof assignment.assigned_at === "string" && Number.isFinite(Date.parse(assignment.assigned_at)),
            "Invalid role assignment or attribution.");
        } catch {
          issues.push({caseId: id, passageId, position: position + 1, start: passage.start, field: "role_assignment",
            reason: "Role assignment is invalid. Select Operator, Caller or Unsure again to record a valid role and reviewer."});
        }
      } else if (requireRoles && edit.status !== "draft") {
        issues.push({caseId: id, passageId, position: position + 1, start: passage.start, field: "role_assignment",
          reason: "Role assignment is required. Select Operator, Caller or Unsure for this block."});
      }
      require(/^[\w:-]+$/u.test(passageId), "Invalid passage identity.");
      require(typeof passage.text === "string", "Invalid passage text.");
      if (!passage.text.trim() && edit.status !== "draft") issues.push({caseId: id, passageId, position: position + 1,
        start: passage.start, field: "text", reason: "Transcript is empty. Enter the spoken words or remove this passage."});
      require(typeof passage.uncertainty === "string", "Invalid uncertainty note.");
      if (edit.status !== "draft" && passage.uncertainty.includes(REVIEW_MARKER)) {
        issues.push({caseId: id, passageId, position: position + 1, start: passage.start, field: "uncertainty",
          reason: "The initial AI note still has [REVIEW NEEDED]. Check this block and replace the tagged note with your findings, or clear it if resolved. Your own notes are allowed."});
      }
      for (const key of ["origin_ids", "alternatives"]) {
        require(Array.isArray(passage[key]) && passage[key].every(v => typeof v === "string") &&
          new Set(passage[key]).size === passage[key].length, `Invalid ${key}.`);
      }
      require(Number.isFinite(passage.start) && Number.isFinite(passage.end) && passage.start >= 0 &&
        passage.start <= passage.end && passage.end <= base.audio.duration, "Invalid navigation position.");
      if (original.has(passageId)) require(equal(original.get(passageId).origin_ids, passage.origin_ids), "Changed passage lineage.");
      else require(passageId.startsWith("n-") && passage.origin_ids.every(p => original.has(p)), "Unknown new passage or split parent.");
      position++;
    }
    require(typeof edit.notes === "string", "Invalid recording notes.");
    require(["draft", "complete", "complete_with_uncertainty"].includes(edit.status), "Invalid review disposition.");
    if (edit.status === "draft") require(edit.reviewed_at === null && edit.reviewed_by === null, "Unfinished review cannot claim completion.");
    else {
      require(typeof edit.reviewed_by === "string" && edit.reviewed_by.trim() && typeof edit.reviewed_at === "string" && edit.reviewed_at, "Completed review needs a reviewer and date.");
    }
  }
  if (issues.length) {
    const error = new Error("Check the linked blocks below before completing the review. Your own uncertainty notes are allowed.");
    error.issues = issues;
    throw error;
  }
  return draft;
}

export function feedbackPackage(corpus, draft) {
  draft = reviewSnapshot(draft);
  validateDraft(corpus, draft, true);
  const bases = index(corpus.cases, "id");
  const lines = ["Transcription review", "", `Reviewer: ${draft.reviewer || "(not supplied)"}`,
    `Reference revision: ${draft.corpus_revision}`, `Reference hash: ${draft.corpus_sha256}`, "",
    "This document contains the complete portable draft. Import it in the review site to resume.", ""];
  for (const edit of draft.cases) {
    lines.push(edit.case_id, `Disposition: ${edit.status}`,
      `Completed by: ${edit.reviewed_by || "(not completed)"}`,
      `Completed at: ${edit.reviewed_at || "(not completed)"}`, "");
    const originals = index(bases.get(edit.case_id).passages, "id");
    for (const passage of edit.passages) {
      const old = originals.get(passage.id);
      if (equal(old, passage)) continue;
      lines.push(`${passage.id} · ${passage.start.toFixed(1)}s`,
        `Original: ${old?.text ?? "(added passage)"}`, `Proposed: ${passage.text}`,
        ...(passage.role_assignment || old?.role_assignment ? [
          `Role: ${old?.role_assignment?.role ?? "Unassigned"} → ${passage.role_assignment?.role ?? "Unassigned"}`,
          `Role assigned by: ${passage.role_assignment?.reviewer ?? "none"} · ${passage.role_assignment?.assigned_at ?? ""}`
        ] : []),
        `Uncertainty: ${passage.uncertainty.trim() || "none"}`,
        `Previous alternatives: ${JSON.stringify(old?.alternatives ?? [])}`,
        `Proposed alternatives: ${JSON.stringify(passage.alternatives)}`, "");
    }
    for (const old of originals.values()) if (!edit.passages.some(p => p.id === old.id)) lines.push(`Deleted ${old.id}: ${old.text}`, "");
    if (edit.notes) lines.push(`Notes: ${edit.notes}`, "");
  }
  return {format: "transcription-review-feedback", schema_version: 1, exported_at: new Date().toISOString(),
    review_markdown: lines.join("\n"), draft: clone(draft)};
}

export function feedbackHTML(package_) {
  const escape = text => text.replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transcription review feedback</title>
  <style>body{max-width:900px;margin:3rem auto;padding:1rem;color:#182b37;background:#fafaf7;font:16px/1.65 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}h1{font-size:1.7rem}</style>
  <h1>Transcription review feedback</h1><pre>${escape(package_.review_markdown)}</pre>
  <script type="application/json" id="review-feedback">${JSON.stringify(package_).replaceAll("<", "\\u003c")}</script></html>`;
}

export function parseFeedback(corpus, text) {
  let package_;
  if (text.trimStart().startsWith("<")) {
    const document_ = new DOMParser().parseFromString(text, "text/html");
    const payloads = document_.querySelectorAll("script#review-feedback");
    require(payloads.length === 1, "Missing or duplicate feedback data.");
    package_ = JSON.parse(payloads[0].textContent);
  } else package_ = JSON.parse(text);
  fields(package_, ["format", "schema_version", "exported_at", "review_markdown", "draft"]);
  require(package_.format === "transcription-review-feedback" && package_.schema_version === 1 &&
    typeof package_.review_markdown === "string", "Unsupported feedback file.");
  validateDraft(corpus, package_.draft, false);
  return package_.draft;
}

export class DraftStore {
  constructor(corpus) {
    this.scope = `${corpus.project}:${corpus.content_sha256}`;
    this.db = null;
  }
  async open() {
    this.db = await new Promise((resolve, reject) => {
      const request = indexedDB.open("transcription-reviews-v1", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("drafts", {keyPath: "key"});
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error("Browser draft storage is blocked."));
    });
    const token = randomId();
    await this.transaction("readwrite", store => {
      store.put({key: `probe:${this.scope}`, token});
      const check = store.get(`probe:${this.scope}`);
      check.onsuccess = () => { if (check.result?.token !== token) store.transaction.abort(); };
      store.delete(`probe:${this.scope}`);
    });
  }
  transaction(mode, action) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction("drafts", mode);
      let result;
      transaction.oncomplete = () => resolve(result);
      transaction.onerror = () => reject(transaction.error || new Error("Draft storage failed."));
      transaction.onabort = () => reject(transaction.error || new Error("Draft storage transaction was cancelled."));
      action(transaction.objectStore("drafts"), value => { result = value; });
    });
  }
  async load() {
    return this.transaction("readonly", (store, result) => {
      const pointer = store.get(`active:${this.scope}`);
      pointer.onsuccess = () => {
        if (!pointer.result) { result(null); return; }
        const request = store.get(pointer.result.draft_key);
        request.onsuccess = () => result(request.result || null);
      };
    });
  }
  async list() {
    return this.transaction("readonly", (store, result) => {
      const request = store.getAll();
      request.onsuccess = () => result(request.result.filter(row => row.key.startsWith(`${this.scope}:`) && row.draft));
    });
  }
  async read(session) {
    return this.transaction("readonly", (store, result) => {
      const request = store.get(`${this.scope}:${session}`);
      request.onsuccess = () => result(request.result || null);
    });
  }
  async save(draft, expectedStamp) {
    let conflict = false;
    const key = `${this.scope}:${draft.session_id}`;
    const stamp = randomId();
    await this.transaction("readwrite", store => {
      const request = store.get(key);
      request.onsuccess = () => {
        if ((request.result?.stamp ?? null) !== expectedStamp) { conflict = true; return; }
        store.put({key, stamp, draft: reviewSnapshot(draft)});
        store.put({key: `active:${this.scope}`, draft_key: key});
      };
    });
    if (conflict) throw new Error("Another tab changed this draft. Download your work or keep it as a separate draft; neither version has been overwritten.");
    return stamp;
  }
}
