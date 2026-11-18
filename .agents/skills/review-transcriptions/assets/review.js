
const $ = id => document.getElementById(id);
const clone = value => structuredClone(value);
const time = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
let corpus, selection, bases, draft, store, stamp = null, timer, channel;
let saving = null, storageReady = false, blocked = false, savedRevision = -1;
let lastExport = null, evidenceLoaded = null;

function message(text) { $("message").textContent = text; $("message").hidden = !text; }
function status(text) { $("save-status").textContent = text; }
function base() { return bases.get(draft.active_case); }
function current() { return draft.cases.find(c => c.case_id === draft.active_case) || proposal(base()); }
function mutable() {
  let edit = draft.cases.find(c => c.case_id === draft.active_case);
  if (!edit) { edit = proposal(base()); draft.cases.push(edit); }
  return edit;
}
function header() {
  $("reviewer").value = draft.reviewer;
  const complete = draft.cases.filter(c => selection.includes(c.case_id) && c.status.startsWith("complete")).length;
  $("progress").textContent = `${complete} of ${selection.length} reviewed`;
  $("download").disabled = false;
  $("saved-drafts").disabled = !storageReady;
}
function disposition() {
  const edit = current();
  const labels = {draft: "Unreviewed", complete: "Reviewed", complete_with_uncertainty: "Reviewed · unresolved"};
  $("call-status").textContent = labels[edit.status];
  $("reopen").hidden = edit.status === "draft";
  $("completion-message").hidden = edit.status === "draft";
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
function play(start) {
  $("audio").currentTime = Math.max(0, start - 2);
  startPlayback();
}
function freshPassage(start, end, text = "", origins = []) {
  return {id: `n-${randomId()}`, text, start, end, origin_ids: origins,
    uncertainty: "", alternatives: [], speaker_change_before: false, boundary_basis: null};
}
function speaker(passage, checked) {
  passage.speaker_change_before = checked;
  passage.boundary_basis = checked ? {actor_type: "human", note: "Marked while listening to the original recording."} : null;
}
function insertAfter(id, atStart = false) {
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
  const parent = $("passages"); parent.replaceChildren();
  $("passage-count").textContent = `${edit.passages.length} passages`;
  edit.passages.forEach((item, position) => {
    const card = document.createElement("article"); card.className = `passage${item.speaker_change_before ? " changed-speaker" : ""}`; card.dataset.id = item.id;
    const head = document.createElement("div"); head.className = "passage-head";
    head.append(button(`▶ ${time(item.start)}`, () => play(item.start), "play-passage"));
    const boundaryLabel = document.createElement("label");
    const boundary = document.createElement("input"); boundary.type = "checkbox"; boundary.checked = item.speaker_change_before; boundary.disabled = position === 0;
    boundary.addEventListener("change", () => { editCase(edit => speaker(edit.passages.find(p => p.id === item.id), boundary.checked)); card.classList.toggle("changed-speaker", boundary.checked); });
    boundaryLabel.append(boundary, "Speaker changes here"); head.append(boundaryLabel);
    if (item.uncertainty) { const tag = document.createElement("span"); tag.className = "uncertain-tag"; tag.textContent = "Needs a listen"; head.append(tag); }
    const number = document.createElement("span"); number.className = "passage-number"; number.textContent = position + 1; head.append(number);
    const editor = document.createElement("textarea"); editor.value = item.text; editor.rows = Math.max(2, Math.ceil(item.text.length / 95)); editor.setAttribute("aria-label", `Passage ${position + 1} text`);
    editor.addEventListener("input", () => editCase(edit => { edit.passages.find(p => p.id === item.id).text = editor.value; }));
    editor.addEventListener("focus", () => {
      parent.querySelectorAll(".active").forEach(n => n.classList.remove("active")); card.classList.add("active");
      const player = document.querySelector(".player");
      const ceiling = Math.max(document.querySelector(".review-header").getBoundingClientRect().bottom,
        player.classList.contains("inline-player") ? 0 : player.getBoundingClientRect().bottom) + 12;
      if (editor.getBoundingClientRect().top < ceiling) window.scrollBy(0, editor.getBoundingClientRect().top - ceiling);
    });
    const actions = document.createElement("div"); actions.className = "passage-actions";
    if (position === 0) actions.append(button("+ Missing speech before", () => insertAfter(null, true)));
    actions.append(button("Split at cursor", () => {
      const offset = editor.selectionStart;
      const before = editor.value.slice(0, offset).trim(), after = editor.value.slice(offset).trim();
      if (!before || !after) { message("Place the text cursor between the two speakers’ words, then choose Split at cursor."); editor.focus(); return; }
      let newId;
      editCase(edit => {
        const at = edit.passages.findIndex(p => p.id === item.id), existing = edit.passages[at];
        const parents = base().passages.some(p => p.id === existing.id) ? [existing.id] : [...existing.origin_ids];
        const added = freshPassage(existing.start, existing.end, after, parents);
        added.uncertainty = existing.uncertainty; added.alternatives = [...existing.alternatives]; speaker(added, true);
        existing.text = before; edit.passages.splice(at + 1, 0, added); newId = added.id;
      });
      message("Split saved. Both passages keep the same approximate audio window; timing precision is not needed here."); renderPassages(newId);
    }));
    actions.append(button("+ Missing speech after", () => insertAfter(item.id)));
    actions.append(button("Uncertain wording", () => { details.open = true; uncertainty.focus(); }));
    const remove = button("Remove", () => {
      if (!window.confirm("Remove this passage from your proposed transcript? The original reference will be preserved.")) return;
      editCase(edit => {
        edit.passages = edit.passages.filter(p => p.id !== item.id);
        if (position === 0) speaker(edit.passages[0], false);
      }); renderPassages();
    }, "danger"); remove.disabled = edit.passages.length === 1; actions.append(remove);
    const details = document.createElement("details"); details.className = "uncertainty"; details.open = Boolean(item.uncertainty);
    const summary = document.createElement("summary"); summary.textContent = "Uncertainty, alternatives and audio position"; details.append(summary);
    const uncertaintyLabel = document.createElement("label"); uncertaintyLabel.textContent = "What is unclear? Leave empty when resolved.";
    const uncertainty = document.createElement("textarea"); uncertainty.rows = 2; uncertainty.value = item.uncertainty;
    uncertainty.setAttribute("aria-label", `Passage ${position + 1} uncertainty`);
    uncertainty.addEventListener("input", () => editCase(edit => { edit.passages.find(p => p.id === item.id).uncertainty = uncertainty.value; }));
    uncertaintyLabel.append(uncertainty); details.append(uncertaintyLabel);
    const alternativeLabel = document.createElement("label"); alternativeLabel.textContent = "Possible wording (one alternative per line)";
    const alternatives = document.createElement("textarea"); alternatives.rows = 2; alternatives.value = item.alternatives.join("\n");
    alternatives.addEventListener("input", () => editCase(edit => { edit.passages.find(p => p.id === item.id).alternatives = [...new Set(alternatives.value.split("\n").map(s => s.trim()).filter(Boolean))]; }));
    alternativeLabel.append(alternatives); details.append(alternativeLabel);
    const adjust = button("Use current audio position for replay", () => {
      const now = Math.min(base().audio.duration, $("audio").currentTime);
      editCase(edit => { const p = edit.passages.find(p => p.id === item.id); p.start = now; p.end = Math.max(now, p.end); }); renderPassages(item.id);
    }); details.append(adjust);
    card.append(head, editor, actions, details); parent.append(card);
    if (item.id === focusId) requestAnimationFrame(() => editor.focus());
  });
}
function renderCase() {
  if (!selection.includes(draft.active_case)) draft.active_case = selection[0];
  const item = base(), edit = current(), at = selection.indexOf(item.id);
  $("recording").value = item.id;
  $("previous").disabled = at === 0; $("next").disabled = at === selection.length - 1;
  $("call-title").textContent = item.id.replace(/T(\d{2})_(\d{2})_(\d{2})_/, " · $1:$2:$3 · ");
  $("audio").src = `audio/${encodeURIComponent(item.id)}.wav`;
  $("audio").playbackRate = Number($("speed").value);
  $("notes").value = edit.notes;
  evidenceLoaded = null; $("evidence").open = false; $("evidence-content").replaceChildren();
  renderPassages(); disposition();
}
function navigate(index_) {
  if (index_ < 0 || index_ >= selection.length) return;
  draft.active_case = selection[index_]; change(); renderCase();
  $("recording").scrollIntoView({block: "center"});
}
function complete(unresolved) {
  if (!draft.reviewer.trim()) { message("Enter your name before completing a review."); $("reviewer").focus(); return; }
  if (!$("listened").checked) { message("Confirm that you listened through the call and checked the words and speaker changes."); return; }
  const trial = clone(draft), edit = trial.cases.find(c => c.case_id === trial.active_case) || proposal(base());
  if (!trial.cases.includes(edit)) trial.cases.push(edit);
  edit.status = unresolved ? "complete_with_uncertainty" : "complete";
  edit.reviewed_at = new Date().toISOString(); edit.reviewed_by = trial.reviewer;
  try { validateDraft(corpus, trial); } catch (error) { message(error.message); return; }
  draft = trial; change(); disposition(); message("");
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
  } catch (error) { message(error.message); }
}
async function canSwitch() {
  await flush();
  if (savedRevision === draft.revision || (!draft.cases.length && !draft.reviewer)) return true;
  if (lastExport === `${draft.session_id}:${draft.revision}` && window.confirm("Current edits are not saved in this browser. Confirm that you saved the downloaded feedback before opening another draft.")) return true;
  message("Your current edits have not been saved. Download feedback first, then try opening the other draft again."); return false;
}
async function openDraft(incoming, incomingStamp = null) {
  validateDraft(corpus, incoming);
  if (!await canSwitch()) return;
  draft = clone(incoming); stamp = incomingStamp; savedRevision = -1; blocked = false;
  clearTimeout(timer); renderCase(); status("Saving…"); await flush();
}
async function resume(file) {
  try {
    const incoming = parseFeedback(corpus, await file.text());
    if (draft.cases.length && JSON.stringify(draft) !== JSON.stringify(incoming) && !window.confirm("Open the imported review? Your current draft will remain in Saved browser drafts.")) return;
    const existing = storageReady ? await store.read(incoming.session_id) : null;
    let incomingStamp = existing?.stamp || null;
    if (existing && JSON.stringify(existing.draft) !== JSON.stringify(incoming)) {
      if (!window.confirm("This session already has different edits. Keep the imported version as a separate draft?")) return;
      incoming.session_id = randomId(); incomingStamp = null;
    }
    await openDraft(incoming, incomingStamp);
  } catch (error) { message(error.message); }
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
    if (saved) { validateDraft(corpus, saved.draft); draft = saved.draft; stamp = saved.stamp; savedRevision = draft.revision; status("Saved in this browser"); }
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
  $("speed").onchange = () => { $("audio").playbackRate = Number($("speed").value); };
  $("replay").onclick = () => { const a = $("audio"); a.currentTime = Math.max(0, a.currentTime - 5); startPlayback(); };
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
}
main().catch(error => { message(error.message); status("Could not open review"); });
