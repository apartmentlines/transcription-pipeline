
const $ = id => document.getElementById(id);
const clone = value => structuredClone(value);
const time = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
let corpus, selection, bases, draft, store, stamp = null, timer, channel;
let saving = null, storageReady = false, blocked = false, savedRevision = -1;
let lastExport = null, evidenceLoaded = null;
let rangeAnchor = null, rangeEnd = null, lastRolePosition = -1;
let highlightedBlock = null;
const recordingMessages = new Map();

function message(text) {
  if (text) recordingMessages.set(draft?.active_case, {message: text});
  else recordingMessages.delete(draft?.active_case);
  renderMessage();
}
function renderMessage() {
  const error = recordingMessages.get(draft?.active_case);
  $("message").textContent = error?.message || "";
  $("message").hidden = !error;
  if (!error?.issues?.length) return;
  const list = document.createElement("ul");
  for (const issue of error.issues) {
    const row = document.createElement("li"), link = document.createElement("a");
    link.href = `#passage-${issue.passageId}`;
    link.textContent = `${issue.caseId} · Block ${issue.position} · ${time(issue.start)}`;
    link.addEventListener("click", event => {
      event.preventDefault();
      const at = selection.indexOf(issue.caseId);
      if (at < 0) { message(`Recording ${issue.caseId} is outside this site's selection. Open the full review site to edit it.`); return; }
      if (draft.active_case !== issue.caseId) navigate(at);
      const card = document.getElementById(`passage-${issue.passageId}`);
      if (!card) return;
      if (issue.field === "role_assignment") {
        selectRange(issue.passageId);
        $("assign-operator").focus({preventScroll: true});
        return;
      }
      if (issue.field === "uncertainty") card.querySelector("details").open = true;
      const target = issue.field === "uncertainty" ? card.querySelector(".uncertainty textarea") : card.querySelector("textarea");
      target.focus({preventScroll: true});
      target.scrollIntoView({block: "center"});
    });
    row.append(link, ` — ${issue.reason}`); list.append(row);
  }
  $("message").append(list);
}
function validationMessage(error) {
  if (!error.issues?.length) { message(error.message); return; }
  for (const id of new Set(error.issues.map(issue => issue.caseId))) {
    recordingMessages.set(id, {message: error.message, issues: error.issues.filter(issue => issue.caseId === id)});
  }
  if (!error.issues.some(issue => issue.caseId === draft.active_case)) {
    const id = error.issues[0].caseId, at = selection.indexOf(id);
    if (at < 0) { message(`Recording ${id} is outside this site's selection. Open the full review site to edit it.`); return; }
    navigate(at);
  }
  renderMessage();
  $("message").scrollIntoView({block: "center"});
}
function status(text) { $("save-status").textContent = text; }
function base() { return bases.get(draft.active_case); }
function current() { return draft.cases.find(c => c.case_id === draft.active_case) || proposal(base()); }
function mutable() {
  let edit = draft.cases.find(c => c.case_id === draft.active_case);
  if (!edit) { edit = proposal(base()); draft.cases.push(edit); }
  return edit;
}
function isComplete(edit) {
  return Boolean(edit?.status.startsWith("complete") && edit.passages.every(p => p.role_assignment));
}
function header() {
  $("reviewer").value = draft.reviewer;
  const complete = draft.cases.filter(c => selection.includes(c.case_id) && isComplete(c)).length;
  $("progress").textContent = `${complete} of ${selection.length} reviewed`;
  $("download").disabled = false;
  $("saved-drafts").disabled = !storageReady;
}
function disposition() {
  const edit = current();
  const needsRoles = edit.status !== "draft" && edit.passages.some(p => !p.role_assignment);
  const labels = {draft: "Unreviewed", complete: "Reviewed", complete_with_uncertainty: "Reviewed · uncertainty noted"};
  $("call-status").textContent = needsRoles ? "Roles required" : labels[edit.status];
  $("review-attribution").textContent = edit.reviewed_by
    ? `Completed by ${edit.reviewed_by} · ${edit.reviewed_at}` : "";
  $("reopen").hidden = edit.status === "draft";
  $("completion-message").hidden = edit.status === "draft" || needsRoles;
  $("listened").checked = edit.status !== "draft";
  header();
}
function change() {
  draft.revision++;
  clearTimeout(timer);
  if (blocked) status("Conflict · download or keep a separate draft");
  else if (storageReady) { status("Saving…"); timer = setTimeout(flush, 180); }
  else status("Only in this page · download to keep changes");
  header();
}
function editCase(action) {
  message("");
  const edit = mutable();
  action(edit);
  edit.status = "draft";
  edit.reviewed_at = null;
  edit.reviewed_by = null;
  change();
  disposition();
}
async function flush() {
  clearTimeout(timer);
  if (saving) await saving;
  if (!storageReady || blocked || savedRevision === draft.revision) return;
  const snapshot = clone(draft);
  saving = (async () => {
    try {
      stamp = await store.save(snapshot, stamp);
      savedRevision = snapshot.revision;
      channel?.postMessage({session: snapshot.session_id, stamp});
      if (draft.revision === snapshot.revision) status("Saved in this browser");
    } catch (error) {
      blocked = error.message.includes("Another tab");
      if (!blocked) storageReady = false;
      status(blocked ? "Conflict · download or keep a separate draft" : "Only in this page · download to keep changes");
      message(error.message + (blocked ? "" : " Your edits are still here and can be downloaded."));
    }
  })();
  await saving;
  saving = null;
  if (storageReady && !blocked && savedRevision !== draft.revision) timer = setTimeout(flush, 0);
  header();
}
function button(text, action, className = "") {
  const result = document.createElement("button"); result.textContent = text; result.className = className;
  result.addEventListener("click", action); return result;
}
function startPlayback() {
  $("audio").play().catch(error => {
    if (error.name !== "AbortError") message("Could not start playback. Try the audio player's Play button or reload this page.");
  });
}
function reviewShortcut(event) {
  if (event.isComposing || event.ctrlKey || event.metaKey || event.altKey) return;
  const target = event.composedPath()[0];
  const editing = target instanceof HTMLElement && (
    target instanceof HTMLTextAreaElement || target.isContentEditable || target.closest('[role="textbox"]') ||
    target instanceof HTMLInputElement && !["checkbox", "radio", "button", "submit", "reset", "range", "file", "color"].includes(target.type)
  );
  if (editing) {
    if (event.type === "keydown" && event.key === "Escape") {
      document.activeElement?.blur(); event.preventDefault(); event.stopPropagation();
    }
    return;
  }
  const key = event.key.toLowerCase();
  const navigationKey = ["j", "k", "arrowdown", "arrowup"].includes(key);
  const selectionKey = key === "s" || rangeAnchor !== null && ["o", "c", "u", "r", "x"].includes(key);
  const attestationKey = key === "enter" && target === $("listened");
  if ((navigationKey || selectionKey) && (target instanceof HTMLSelectElement || target.closest?.("dialog"))) return;
  if (!navigationKey && !selectionKey && !attestationKey && ![" ", "h", "l", "arrowleft", "arrowright"].includes(key)) return;
  event.preventDefault(); event.stopPropagation();
  if (event.type !== "keydown") return;
  if (navigationKey) {
    const direction = key === "j" || key === "arrowdown" ? 1 : -1;
    if (rangeAnchor !== null) moveRangeEnd(direction);
    else moveHighlightedBlock(direction);
    return;
  }
  if (selectionKey) {
    if (event.repeat) return;
    if (key === "s") {
      if (!rangeAnchor) {
        if (!highlightedBlock) focusReviewPosition();
        if (highlightedBlock) { selectRange(highlightedBlock); focusBlock(highlightedBlock); }
      }
    } else finishKeyboardSelection(key);
    return;
  }
  if (attestationKey) {
    if (!event.repeat) $("listened").click();
    return;
  }
  const audio = $("audio");
  if (key === " ") {
    if (!event.repeat) { if (audio.paused) startPlayback(); else audio.pause(); }
    return;
  }
  const step = event.shiftKey || event.key === "H" || event.key === "L" ? 10 : 2;
  const direction = key === "h" || key === "arrowleft" ? -1 : 1;
  const duration = Number.isFinite(audio.duration) ? audio.duration : base().audio.duration;
  audio.currentTime = Math.max(0, Math.min(duration, audio.currentTime + direction * step));
}
function play(start) {
  $("audio").currentTime = Math.max(0, start - 2);
  startPlayback();
}
function freshPassage(start, end, text = "", origins = []) {
  return {id: `n-${randomId()}`, text, start, end, origin_ids: origins,
    uncertainty: "", alternatives: []};
}
function selectedRange() {
  const passages = current().passages;
  const start = passages.findIndex(p => p.id === rangeAnchor), end = passages.findIndex(p => p.id === rangeEnd);
  return start < 0 || end < 0 ? [] : passages.slice(Math.min(start, end), Math.max(start, end) + 1);
}
function highlightBlock(id) {
  highlightedBlock = id;
  document.querySelectorAll(".passage").forEach(card => {
    const highlighted = card.dataset.id === id;
    card.classList.toggle("keyboard-highlighted", highlighted);
    if (highlighted) card.setAttribute("aria-current", "true");
    else card.removeAttribute("aria-current");
  });
}
function focusBlock(id) {
  highlightBlock(id);
  const card = document.getElementById(`passage-${id}`);
  card.querySelector(".select-block").focus({preventScroll: true});
  card.scrollIntoView({block: "center"});
}
function focusReviewPosition() {
  const first = current().passages.find(p => !p.role_assignment);
  if (first) focusBlock(first.id);
  else {
    highlightBlock(null);
    $("listened").focus({preventScroll: true});
    $("listened").scrollIntoView({block: "center"});
  }
}
function moveHighlightedBlock(direction) {
  const passages = current().passages, at = passages.findIndex(p => p.id === highlightedBlock);
  if (at < 0) { focusReviewPosition(); return; }
  const next = Math.max(0, Math.min(passages.length - 1, at + direction));
  focusBlock(passages[next].id);
}
function finishKeyboardSelection(key) {
  const selected = selectedRange();
  if (!selected.length) return;
  const passages = current().passages, last = selected.at(-1).id;
  const next = passages[passages.findIndex(p => p.id === last) + 1]?.id;
  if (key === "x") { resetRoleTools(); updateRoleTools(); }
  else if (!assignRole({o: "Operator", c: "Caller", u: "Unsure", r: null}[key])) return;
  if (next) focusBlock(next);
  else if (current().passages.every(p => p.role_assignment)) focusReviewPosition();
  else focusBlock(last);
}
function selectRange(id, extend = false) {
  const card = document.getElementById(`passage-${id}`), top = card.getBoundingClientRect().top;
  if (!extend || !rangeAnchor) rangeAnchor = id;
  rangeEnd = id;
  highlightBlock(id);
  updateRoleTools();
  window.scrollBy(0, card.getBoundingClientRect().top - top);
  document.querySelector(".role-tools").scrollIntoView({block: "nearest"});
}
function moveRangeEnd(direction) {
  const passages = current().passages, at = passages.findIndex(p => p.id === rangeEnd);
  const next = at + direction;
  if (at < 0 || next < 0 || next >= passages.length) return;
  const focused = document.activeElement;
  const keepFocus = document.querySelector(".role-tools").contains(focused);
  selectRange(passages[next].id, true);
  const card = document.getElementById(`passage-${rangeEnd}`);
  const target = keepFocus ? focused : card.querySelector(".select-block");
  target.focus({preventScroll: true});
  card.scrollIntoView({block: "center"});
}
function updateRoleTools() {
  const passages = current().passages, selected = new Set(selectedRange().map(p => p.id));
  const assigned = passages.filter(p => knownRole(p)).length;
  const unsure = passages.filter(p => p.role_assignment?.role === "Unsure").length;
  $("role-progress").textContent = `Roles: ${assigned}/${passages.length} labeled · ${unsure} unsure · ${passages.length - assigned - unsure} unassigned`;
  $("next-unlabeled").disabled = assigned === passages.length;
  $("role-selection").hidden = !selected.size;
  const toolbar = document.querySelector(".role-tools");
  if (selected.size) {
    document.getElementById(`passage-${rangeEnd}`).querySelector(".passage-head").after(toolbar);
    const positions = passages.map((p, i) => selected.has(p.id) ? i + 1 : null).filter(Boolean);
    $("range-description").textContent = `Blocks ${positions[0]}–${positions.at(-1)} selected · ${selected.size} ${selected.size === 1 ? "block" : "blocks"}. J/↓ or K/↑ adjusts the range. O: Operator · C: Caller · U: Unsure · R: Remove labels · X: Clear selection.`;
  } else $("role-tools-home").append(toolbar);
  document.querySelectorAll(".passage").forEach(card => {
    card.classList.toggle("range-selected", selected.has(card.dataset.id));
    card.querySelector(".select-block").setAttribute("aria-pressed", String(selected.has(card.dataset.id)));
    card.querySelector(".extend-range").hidden = !rangeAnchor || rangeAnchor === card.dataset.id;
  });
}
function resetRoleTools() {
  rangeAnchor = rangeEnd = null;
}
function assignRole(role) {
  const selected = selectedRange();
  if (!selected.length) return false;
  if (role && !draft.reviewer.trim()) {
    message("Enter your name so these role decisions can be attributed to you."); $("reviewer").focus(); return false;
  }
  const endpoint = rangeEnd;
  const ids = new Set(selected.map(p => p.id));
  const assignment = {role, reviewer: draft.reviewer.trim(), assigned_at: new Date().toISOString()};
  editCase(edit => {
    for (const passage of edit.passages) if (ids.has(passage.id)) {
      if (role) passage.role_assignment = clone(assignment);
      else delete passage.role_assignment;
    }
  });
  lastRolePosition = current().passages.findIndex(p => p.id === endpoint);
  rangeAnchor = rangeEnd = null;
  renderPassages();
  document.getElementById(`passage-${endpoint}`).querySelector(".select-block").focus({preventScroll: true});
  return true;
}
function nextUnlabeled() {
  const passages = current().passages;
  const positions = passages.map((_, i) => (i + lastRolePosition + 1) % passages.length);
  const at = positions.find(i => !knownRole(passages[i]));
  if (at === undefined) return;
  lastRolePosition = at;
  selectRange(passages[at].id);
  const card = document.getElementById(`passage-${passages[at].id}`);
  card.scrollIntoView({block: "center"});
  card.querySelector(".select-block").focus({preventScroll: true});
}
function insertAfter(id, atStart = false) {
  resetRoleTools();
  const playback = Math.min(base().audio.duration, $("audio").currentTime || 0);
  const added = freshPassage(playback, playback);
  editCase(edit => {
    const at = atStart ? 0 : id === null ? edit.passages.length : edit.passages.findIndex(p => p.id === id) + 1;
    edit.passages.splice(at, 0, added);
  });
  renderPassages(added.id);
}
function renderPassages(focusId = null) {
  const edit = current();
  $("role-tools-home").append(document.querySelector(".role-tools"));
  const parent = $("passages"); parent.replaceChildren();
  $("passage-count").textContent = `${edit.passages.length} passages`;
  edit.passages.forEach((item, position) => {
    const previous = edit.passages[position - 1], role = knownRole(item);
    const derivedChange = roleChange(previous, item);
    const card = document.createElement("article"); card.className = `passage${derivedChange ? " changed-speaker" : ""}`; card.dataset.id = item.id;
    if (role) card.classList.add(`role-${role.toLowerCase()}`);
    if (role && knownRole(previous) === role) card.classList.add("role-continuation");
    card.id = `passage-${item.id}`;
    card.addEventListener("focusin", () => highlightBlock(item.id));
    const head = document.createElement("div"); head.className = "passage-head";
    head.append(button(`▶ ${time(item.start)}`, () => play(item.start), "play-passage"));
    const roleBadge = document.createElement("span"); roleBadge.className = "role-badge";
    roleBadge.textContent = item.role_assignment?.role || "Unassigned";
    if (item.role_assignment) roleBadge.title = `Assigned by ${item.role_assignment.reviewer} · ${item.role_assignment.assigned_at}`;
    head.append(roleBadge);
    if (derivedChange) {
      const changeLabel = document.createElement("span"); changeLabel.className = "muted";
      changeLabel.textContent = "Speaker change · from roles"; head.append(changeLabel);
    }
    if (hasUncertainty(item)) { const tag = document.createElement("span"); tag.className = "uncertain-tag"; tag.textContent = "Needs a listen"; head.append(tag); }
    const select = button(`Select ${position + 1}`, event => selectRange(item.id, event.shiftKey), "select-block");
    select.setAttribute("aria-label", `Select block ${position + 1}`);
    const extend = button("Extend selection through here", () => selectRange(item.id, true), "extend-range");
    extend.setAttribute("aria-label", `Extend selection through block ${position + 1}`);
    head.append(select, extend);
    const editor = document.createElement("textarea"); editor.value = item.text; editor.rows = Math.max(2, Math.ceil(item.text.length / 95)); editor.setAttribute("aria-label", `Passage ${position + 1} text`);
    editor.addEventListener("input", () => editCase(edit => { edit.passages.find(p => p.id === item.id).text = editor.value; }));
    editor.addEventListener("focus", () => {
      const player = document.querySelector(".player");
      const ceiling = Math.max(document.querySelector(".review-header").getBoundingClientRect().bottom,
        player.classList.contains("inline-player") ? 0 : player.getBoundingClientRect().bottom) + 12;
      const floor = window.innerHeight - 12;
      const bounds = editor.getBoundingClientRect();
      if (bounds.top < ceiling) window.scrollBy(0, bounds.top - ceiling);
      else if (bounds.bottom > floor) window.scrollBy(0, Math.min(bounds.bottom - floor, bounds.top - ceiling));
    });
    const actions = document.createElement("div"); actions.className = "passage-actions";
    if (position === 0) actions.append(button("+ Missing speech before", () => insertAfter(null, true)));
    actions.append(button("Split at cursor", () => {
      const offset = editor.selectionStart;
      const before = editor.value.slice(0, offset).trim(), after = editor.value.slice(offset).trim();
      if (!before || !after) { message("Place the text cursor between the two speakers’ words, then choose Split at cursor."); editor.focus(); return; }
      resetRoleTools();
      let newId;
      editCase(edit => {
        const at = edit.passages.findIndex(p => p.id === item.id), existing = edit.passages[at];
        const parents = base().passages.some(p => p.id === existing.id) ? [existing.id] : [...existing.origin_ids];
        const added = freshPassage(existing.start, existing.end, after, parents);
        added.uncertainty = existing.uncertainty; added.alternatives = [...existing.alternatives];
        existing.text = before; edit.passages.splice(at + 1, 0, added); newId = added.id;
      });
      message("Split saved. Both passages keep the same approximate audio window; timing precision is not needed here."); renderPassages(newId);
    }));
    actions.append(button("+ Missing speech after", () => insertAfter(item.id)));
    actions.append(button("Uncertain wording", () => { details.open = true; uncertainty.focus(); }));
    const remove = button("Remove", () => {
      if (!window.confirm("Remove this passage from your proposed transcript? The original reference will be preserved.")) return;
      resetRoleTools();
      editCase(edit => {
        edit.passages = edit.passages.filter(p => p.id !== item.id);
      }); renderPassages();
    }, "danger"); remove.disabled = edit.passages.length === 1; actions.append(remove);
    const details = document.createElement("details"); details.className = "uncertainty"; details.open = hasUncertainty(item) || item.alternatives.length > 0;
    const summary = document.createElement("summary"); summary.textContent = "Uncertainty, alternatives and audio position"; details.append(summary);
    const uncertaintyLabel = document.createElement("label"); uncertaintyLabel.textContent = "Uncertainty notes (optional). Replace any [REVIEW NEEDED] AI note with your findings, or clear it if resolved. Your own notes can remain when you complete review.";
    const uncertainty = document.createElement("textarea"); uncertainty.rows = 2; uncertainty.value = item.uncertainty;
    uncertainty.setAttribute("aria-label", `Passage ${position + 1} uncertainty`);
    uncertainty.addEventListener("input", () => editCase(edit => { edit.passages.find(p => p.id === item.id).uncertainty = uncertainty.value; }));
    uncertaintyLabel.append(uncertainty); details.append(uncertaintyLabel);
    const alternativeLabel = document.createElement("label"); alternativeLabel.textContent = "Possible wording (one alternative per line). This does not change the transcript. Edit confirmed corrections above; describe any remaining doubt in Uncertainty notes.";
    const alternatives = document.createElement("textarea"); alternatives.rows = 2; alternatives.value = item.alternatives.join("\n");
    alternatives.addEventListener("input", () => editCase(edit => { edit.passages.find(p => p.id === item.id).alternatives = [...new Set(alternatives.value.split("\n").map(s => s.trim()).filter(Boolean))]; }));
    alternativeLabel.append(alternatives); details.append(alternativeLabel);
    const adjust = button("Use current audio position for replay", () => {
      const now = Math.min(base().audio.duration, $("audio").currentTime);
      editCase(edit => { const p = edit.passages.find(p => p.id === item.id); p.start = now; p.end = Math.max(now, p.end); }); renderPassages(item.id);
    }); details.append(adjust);
    card.append(head);
    card.append(editor, actions, details); parent.append(card);
  });
  updateRoleTools();
  highlightBlock(edit.passages.some(p => p.id === highlightedBlock) ? highlightedBlock : null);
  if (focusId) document.getElementById(`passage-${focusId}`).querySelector("textarea").focus();
}
function renderCase() {
  resetRoleTools(); lastRolePosition = -1; highlightedBlock = null;
  if (!selection.includes(draft.active_case)) draft.active_case = selection[0];
  renderMessage();
  const item = base(), edit = current(), at = selection.indexOf(item.id);
  $("recording").value = item.id;
  $("previous").disabled = at === 0; $("next").disabled = at === selection.length - 1;
  $("previous-bottom").disabled = at === 0; $("next-bottom").disabled = at === selection.length - 1;
  $("call-title").textContent = item.id.replace(/T(\d{2})_(\d{2})_(\d{2})_/, " · $1:$2:$3 · ");
  $("audio").src = `audio/${encodeURIComponent(item.id)}.wav`;
  $("audio").playbackRate = Number($("speed").value);
  $("notes").value = edit.notes;
  evidenceLoaded = null; $("evidence").open = false; $("evidence-content").replaceChildren();
  renderPassages(); disposition();
  focusReviewPosition();
}
function navigate(index_) {
  if (index_ < 0 || index_ >= selection.length) return;
  draft.active_case = selection[index_]; change(); renderCase();
}
async function complete(unresolved) {
  if (!draft.reviewer.trim()) { message("Enter your name before completing a review."); $("reviewer").focus(); return; }
  if (!$("listened").checked) { message("Confirm that you listened through the call and checked the words and speaker changes."); return; }
  const trial = clone(draft), edit = trial.cases.find(c => c.case_id === trial.active_case) || proposal(base());
  if (!trial.cases.includes(edit)) trial.cases.push(edit);
  edit.status = unresolved || edit.passages.some(hasUncertainty) ? "complete_with_uncertainty" : "complete";
  edit.reviewed_at = new Date().toISOString(); edit.reviewed_by = trial.reviewer;
  try { validateDraft(corpus, {...trial, cases: [edit]}, true); } catch (error) { validationMessage(error); return; }
  draft = trial; change(); disposition(); message("");
  await flush();
  if (!storageReady || blocked || savedRevision !== draft.revision ||
      draft.active_case !== edit.case_id || !isComplete(current())) return;
  const at = selection.indexOf(edit.case_id);
  for (let offset = 1; offset < selection.length; offset++) {
    const next = (at + offset) % selection.length;
    if (!isComplete(draft.cases.find(c => c.case_id === selection[next]))) {
      recordingMessages.delete(selection[next]);
      navigate(next);
      break;
    }
  }
}
function download() {
  try {
    const package_ = feedbackPackage(corpus, draft);
    const blob = new Blob([feedbackHTML(package_)], {type: "text/html;charset=utf-8"});
    const url = URL.createObjectURL(blob), anchor = document.createElement("a");
    anchor.href = url; anchor.download = `transcript-review-r${draft.corpus_revision}-${draft.session_id.slice(0, 8)}.html`;
    document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000);
    lastExport = `${draft.session_id}:${draft.revision}`;
    message("Feedback download started. Your browser draft has been kept; check that the file saved successfully.");
  } catch (error) { validationMessage(error); }
}
async function canSwitch() {
  await flush();
  if (savedRevision === draft.revision || (!draft.cases.length && !draft.reviewer)) return true;
  if (lastExport === `${draft.session_id}:${draft.revision}` && window.confirm("Current edits are not saved in this browser. Confirm that you saved the downloaded feedback before opening another draft.")) return true;
  message("Your current edits have not been saved. Download feedback first, then try opening the other draft again."); return false;
}
async function openDraft(incoming, incomingStamp = null) {
  validateDraft(corpus, incoming, false);
  if (!await canSwitch()) return;
  draft = clone(incoming); stamp = incomingStamp; savedRevision = -1; blocked = false;
  recordingMessages.clear();
  clearTimeout(timer); renderCase(); status("Saving…"); await flush();
}
async function resume(file) {
  try {
    const incoming = parseFeedback(corpus, await file.text());
    if (draft.cases.length && JSON.stringify(reviewSnapshot(draft)) !== JSON.stringify(reviewSnapshot(incoming)) && !window.confirm("Open the imported review? Your current draft will remain in Saved browser drafts.")) return;
    const existing = storageReady ? await store.read(incoming.session_id) : null;
    let incomingStamp = existing?.stamp || null;
    if (existing && JSON.stringify(reviewSnapshot(existing.draft)) !== JSON.stringify(reviewSnapshot(incoming))) {
      if (!window.confirm("This session already has different edits. Keep the imported version as a separate draft?")) return;
      incoming.session_id = randomId(); incomingStamp = null;
    }
    await openDraft(incoming, incomingStamp);
  } catch (error) { validationMessage(error); }
}
async function separate() {
  if (saving) await saving;
  clearTimeout(timer); draft.session_id = randomId(); stamp = null; savedRevision = -1; blocked = false;
  change(); await flush(); message(storageReady ? "Kept as a separate draft. The other session has been preserved." : "Separate draft is held in this page. Download it to keep a copy.");
}
async function savedDrafts() {
  try {
    const records = await store.list(); $("draft-choice").replaceChildren();
    for (const row of records) {
      const option = document.createElement("option"); option.value = row.draft.session_id;
      option.textContent = `${row.draft.reviewer || "Unnamed reviewer"} · ${row.draft.cases.length} recordings · ${row.draft.session_id.slice(0, 8)}`; $("draft-choice").append(option);
    }
    $("draft-picker").showModal();
  } catch (error) { message(error.message); }
}
async function showEvidence() {
  const identity = draft.active_case;
  if (!$("evidence").open || evidenceLoaded === identity) return;
  const target = $("evidence-content"); target.replaceChildren();
  const note = document.createElement("p"); note.textContent = base().preparation.notes; target.append(note);
  try {
    const evidence = window.reviewData.evidence[identity];
    if (draft.active_case !== identity) return;
    for (const item of evidence) {
      const details = document.createElement("details"), summary = document.createElement("summary"), text = document.createElement("pre");
      summary.textContent = `${item.id} · ${item.kind.replaceAll("_", " ")}`; text.textContent = item.contents;
      details.append(summary, text); target.append(details);
    }
    evidenceLoaded = identity;
  } catch (error) { message(error.message); }
}
async function main() {
  ({corpus, selection} = window.reviewData);
  bases = new Map(corpus.cases.map(c => [c.id, c])); draft = emptyDraft(corpus);
  store = new DraftStore(corpus);
  try {
    await store.open(); storageReady = true;
    const saved = await store.load();
    if (saved) { validateDraft(corpus, saved.draft, false); draft = saved.draft; stamp = saved.stamp; savedRevision = draft.revision; status("Saved in this browser"); }
  } catch (error) { storageReady = false; status("Only in this page · download to keep changes"); message(`Browser drafts are unavailable: ${error.message} Editing and downloading still work.`); }
  if ("BroadcastChannel" in window) {
    channel = new BroadcastChannel(`transcription-review:${corpus.content_sha256}`);
    channel.onmessage = event => {
      if (event.data.session === draft.session_id && event.data.stamp !== stamp && !saving) {
        blocked = true; status("Another tab changed this draft");
        message("Another tab changed this draft. Your page has kept its version. Download it or choose Files → Keep as separate draft before continuing.");
      }
    };
  }
  for (const [at, id] of selection.entries()) { const option = document.createElement("option"); option.value = id; option.textContent = `${at + 1}. ${id}`; $("recording").append(option); }
  $("reviewer").addEventListener("input", () => { draft.reviewer = $("reviewer").value; change(); });
  $("recording").addEventListener("change", () => navigate(selection.indexOf($("recording").value)));
  $("previous").onclick = () => navigate(selection.indexOf(draft.active_case) - 1);
  $("next").onclick = () => navigate(selection.indexOf(draft.active_case) + 1);
  $("previous-bottom").onclick = () => navigate(selection.indexOf(draft.active_case) - 1);
  $("next-bottom").onclick = () => navigate(selection.indexOf(draft.active_case) + 1);
  $("speed").onchange = () => { $("audio").playbackRate = Number($("speed").value); };
  $("replay").onclick = () => { const a = $("audio"); a.currentTime = Math.max(0, a.currentTime - 5); startPlayback(); };
  $("assign-operator").onclick = () => assignRole("Operator");
  $("assign-caller").onclick = () => assignRole("Caller");
  $("assign-unsure").onclick = () => assignRole("Unsure");
  $("clear-roles").onclick = () => assignRole(null);
  $("clear-selection").onclick = () => { rangeAnchor = rangeEnd = null; updateRoleTools(); };
  $("next-unlabeled").onclick = nextUnlabeled;
  $("audio").ontimeupdate = () => { $("audio-position").textContent = `${time($("audio").currentTime)} / ${time(base().audio.duration)}`; };
  $("audio").onerror = () => message("Audio could not be loaded. Check that the site's audio folder was included.");
  $("notes").oninput = () => editCase(edit => { edit.notes = $("notes").value; });
  $("add-end").onclick = () => insertAfter(null);
  $("complete").onclick = () => complete(false); $("complete-unresolved").onclick = () => complete(true);
  $("reopen").onclick = () => editCase(() => {});
  $("download").onclick = download;
  $("resume").onclick = () => { $("files").open = false; $("file-input").click(); };
  $("file-input").onchange = () => { const file = $("file-input").files[0]; if (file) resume(file); $("file-input").value = ""; };
  $("separate").onclick = () => { $("files").open = false; separate(); };
  $("saved-drafts").onclick = () => { $("files").open = false; savedDrafts(); };
  $("cancel-picker").onclick = () => $("draft-picker").close();
  $("open-draft").onclick = async () => { const row = await store.read($("draft-choice").value); $("draft-picker").close(); if (row) await openDraft(row.draft, row.stamp); };
  $("evidence").ontoggle = showEvidence;
  document.addEventListener("keydown", reviewShortcut, true);
  document.addEventListener("keyup", reviewShortcut, true);
  document.addEventListener("keydown", event => { if (event.key === "Escape" && $("files").open) { $("files").open = false; $("files").querySelector("summary").focus(); } });
  document.addEventListener("click", event => { if (!$("files").contains(event.target)) $("files").open = false; });
  document.addEventListener("focusin", event => { if (!$("files").contains(event.target)) $("files").open = false; });
  window.addEventListener("beforeunload", event => { if (savedRevision !== draft.revision) { event.preventDefault(); event.returnValue = ""; } });
  const measurePinned = () => {
    const headerHeight = document.querySelector(".review-header").getBoundingClientRect().height;
    const player = document.querySelector(".player"), playerHeight = player.getBoundingClientRect().height;
    const pinPlayer = headerHeight + playerHeight < window.innerHeight * .55;
    player.classList.toggle("inline-player", !pinPlayer);
    document.documentElement.style.setProperty("--header-height", `${headerHeight}px`);
    document.documentElement.style.setProperty("--player-height", `${pinPlayer ? playerHeight : 0}px`);
  };
  const measure = new ResizeObserver(measurePinned);
  measure.observe(document.querySelector(".review-header")); measure.observe(document.querySelector(".player"));
  window.addEventListener("resize", measurePinned);
  renderCase(); await flush(); header();
  $("running-version").textContent = `Running application version ${window.reviewAppVersion}`;
}
main().catch(error => { message(error.message); status("Could not open review"); });
