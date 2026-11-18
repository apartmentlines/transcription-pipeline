export function randomId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, "0")).join("");
}

const clone = value => structuredClone(value);
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
  return {case_id: base.id, base_revision: base.revision, base_sha256: base.content_sha256,
    passages: clone(base.passages), notes: "", status: "draft", reviewed_at: null, reviewed_by: null};
}

export function validateDraft(corpus, draft) {
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
      fields(passage, ["id", "text", "start", "end", "origin_ids", "uncertainty", "alternatives", "speaker_change_before", "boundary_basis"]);
      require(/^[\w:-]+$/u.test(passageId), "Invalid passage identity.");
      require(typeof passage.text === "string" && (passage.text.trim() || edit.status === "draft"), "Remove empty passages using Remove before completing review.");
      require(typeof passage.uncertainty === "string", "Invalid uncertainty note.");
      for (const key of ["origin_ids", "alternatives"]) {
        require(Array.isArray(passage[key]) && passage[key].every(v => typeof v === "string") &&
          new Set(passage[key]).size === passage[key].length, `Invalid ${key}.`);
      }
      require(Number.isFinite(passage.start) && Number.isFinite(passage.end) && passage.start >= 0 &&
        passage.start <= passage.end && passage.end <= base.audio.duration, "Invalid navigation position.");
      if (original.has(passageId)) require(equal(original.get(passageId).origin_ids, passage.origin_ids), "Changed passage lineage.");
      else require(passageId.startsWith("n-") && passage.origin_ids.every(p => original.has(p)), "Unknown new passage or split parent.");
      require(typeof passage.speaker_change_before === "boolean", "Invalid speaker change.");
      if (passage.speaker_change_before) require(position > 0 &&
        ["human", "model", "assistant"].includes(passage.boundary_basis?.actor_type) && passage.boundary_basis?.note,
        "Speaker changes need supporting evidence.");
      else require(passage.boundary_basis === null, "Unexpected boundary evidence.");
      position++;
    }
    require(typeof edit.notes === "string", "Invalid recording notes.");
    require(["draft", "complete", "complete_with_uncertainty"].includes(edit.status), "Invalid review disposition.");
    if (edit.status === "draft") require(edit.reviewed_at === null && edit.reviewed_by === null, "Unfinished review cannot claim completion.");
    else {
      require(typeof edit.reviewed_by === "string" && edit.reviewed_by.trim() && typeof edit.reviewed_at === "string" && edit.reviewed_at, "Completed review needs a reviewer and date.");
      const unresolved = edit.passages.some(p => p.uncertainty.trim() || p.text.toLowerCase().includes("[unclear]"));
      require(edit.status !== "complete" || !unresolved, "Choose Complete with unresolved passages, or resolve the uncertainty.");
    }
  }
  return draft;
}

export function feedbackPackage(corpus, draft) {
  validateDraft(corpus, draft);
  const bases = index(corpus.cases, "id");
  const lines = ["Transcription review", "", `Reviewer: ${draft.reviewer || "(not supplied)"}`,
    `Reference revision: ${draft.corpus_revision}`, `Reference hash: ${draft.corpus_sha256}`, "",
    "This document contains the complete portable draft. Import it in the review site to resume.", ""];
  for (const edit of draft.cases) {
    lines.push(edit.case_id, `Disposition: ${edit.status}`, "");
    const originals = index(bases.get(edit.case_id).passages, "id");
    for (const passage of edit.passages) {
      const old = originals.get(passage.id);
      if (equal(old, passage)) continue;
      lines.push(`${passage.id} · ${passage.start.toFixed(1)}s`,
        `Original: ${old?.text ?? "(added passage)"}`, `Proposed: ${passage.text}`,
        `Speaker change before: ${passage.speaker_change_before}`,
        `Uncertainty: ${passage.uncertainty || "none"}`, "");
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
  validateDraft(corpus, package_.draft);
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
        store.put({key, stamp, draft: clone(draft)});
        store.put({key: `active:${this.scope}`, draft_key: key});
      };
    });
    if (conflict) throw new Error("Another tab changed this draft. Download your work or keep it as a separate draft; neither version has been overwritten.");
    return stamp;
  }
}
