"""Protect reference integrity, honest scoring, and the human feedback round trip."""

from __future__ import annotations

from base64 import b64encode
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib
from pathlib import Path
import shutil
import sys
from threading import Thread
from typing import Any, override
import wave

import pytest

TOOLS = Path(__file__).resolve().parents[1] / ".agents/skills/review-transcriptions"
sys.path.insert(0, str(TOOLS / "scripts"))
core = importlib.import_module("reference_core")
feedback = importlib.import_module("reference_feedback")
evaluation = importlib.import_module("reference_evaluate")
preparation = importlib.import_module("reference_prepare")
runs = importlib.import_module("reference_run")


class ReferenceFixture:
    """Small synthetic calls with known words and a known speaker change."""

    def __init__(self, root: Path) -> None:
        self.root: Path = root
        self.corpus: Any = None

    @staticmethod
    def assign_roles(edit: Any) -> None:
        """Supply explicit role decisions for tests of completed review behavior."""
        for passage in edit["passages"]:
            passage["role_assignment"] = {
                "role": "Unsure",
                "reviewer": "Fixture reviewer",
                "assigned_at": "2026-11-20T12:00:00Z",
            }

    @staticmethod
    def label_recording(page: Any) -> None:
        """Assign all fixture blocks before testing unrelated completion rules."""
        count = page.locator(".passage").count()
        page.get_by_role("button", name="Select block 1", exact=True).click()
        if count > 1:
            page.get_by_role(
                "button", name=f"Extend selection through block {count}", exact=True
            ).click()
        page.locator("#assign-unsure").click()

    @staticmethod
    def complete_and_return(page: Any, button: str = "#complete") -> None:
        """Return to a completed fixture after its successful automatic advance."""
        identity = page.locator("#recording").input_value()
        page.locator(button).click()
        page.wait_for_function(
            "id => document.querySelector('#recording').value !== id", arg=identity
        )
        page.locator("#recording").select_option(identity)

    def build(self) -> Any:
        entries = []
        for name in ("call-one", "call-two", "call-three"):
            with wave.open(str(self.root / f"{name}.wav"), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(8000)
                audio.writeframes(b"\x00\x00" * 24000)
            (self.root / f"{name}.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHello there.\n\n"
                + "2\n00:00:01,000 --> 00:00:02,000\nGood evening.\n\n",
                encoding="utf-8",
            )
            (self.root / f"{name}.md").write_text("Synthetic audio assessment.")
            entries.append(
                {
                    "id": name,
                    "audio": f"{name}.wav",
                    "seed_srt": f"{name}.srt",
                    "evidence": [
                        {
                            "id": "audio-review",
                            "path": f"{name}.md",
                            "kind": "native_audio_assessment",
                        }
                    ],
                }
            )
        self.corpus = preparation.Preparation.seed(
            self.root,
            {"corpus_id": "synthetic-calls", "cases": entries},
            self.root / "corpus.json",
        )
        return self.corpus


class CachedReviewHandler(SimpleHTTPRequestHandler):
    """Model a static host that caches JavaScript across application upgrades."""

    @override
    def end_headers(self) -> None:
        """Cache scripts while allowing updated HTML to be loaded immediately."""
        policy = (
            "public, max-age=31536000, immutable"
            if self.path.endswith(".js")
            else "no-store"
        )
        self.send_header("Cache-Control", policy)
        super().end_headers()


@pytest.fixture
def reference(tmp_path: Path) -> Any:
    return ReferenceFixture(tmp_path).build()


def test_reconciliation_preserves_originals_and_requires_explicit_decisions(
    reference: Any,
    tmp_path: Path,
) -> None:
    original = deepcopy(reference.data)
    draft = feedback.Feedback.empty(reference)
    draft["reviewer"] = "Fixture reviewer"
    for base in reference.cases.values():
        edit = feedback.Feedback.proposal(base)
        ReferenceFixture.assign_roles(edit)
        edit["passages"][0]["text"] = "Hello friend."
        edit["status"] = "complete"
        edit["reviewed_by"] = "Fixture reviewer"
        edit["reviewed_at"] = "2026-11-18T12:00:00Z"
        edit["notes"] = "Maybe delete everything? This note is not a patch."
        draft["cases"].append(edit)
    package = feedback.Feedback.package(reference, draft)
    path = tmp_path / "feedback.json"
    core.Records.write(path, package)
    decisions = [
        {"case_id": key, "decision": decision, "rationale": "Fixture adjudication"}
        for key, decision in zip(reference.cases, ("accept", "reject", "defer"))
    ]
    with pytest.raises(ValueError, match="every proposal"):
        feedback.Feedback.reconcile(
            reference, path, decisions[:-1], "Adjudicator", tmp_path / "invalid"
        )
    assert not (tmp_path / "invalid").exists()
    revised = feedback.Feedback.reconcile(
        reference, path, decisions, "Adjudicator", tmp_path / "r2"
    )
    assert revised.data["revision"] == 2
    assert reference.data == original
    assert revised.cases["call-one"]["passages"][0]["id"] == "p0001"
    assert revised.cases["call-one"]["passages"][0]["text"] == "Hello friend."
    assert revised.cases["call-one"]["review"]["status"] == "complete"
    assert revised.cases["call-two"] == reference.cases["call-two"]
    assert revised.cases["call-three"] == reference.cases["call-three"]
    assert (tmp_path / "r2/feedback.json").read_bytes() == path.read_bytes()
    with pytest.raises(ValueError, match="Stale"):
        feedback.Feedback.load_package(revised, path)


@pytest.mark.parametrize(
    "damage",
    [
        "stale",
        "unknown",
        "duplicate",
        "lineage",
        "blank-completion",
        "role",
        "missing-role",
    ],
)
def test_invalid_proposals_do_not_modify_the_reference(
    reference: Any, damage: str
) -> None:
    original = deepcopy(reference.data)
    draft = feedback.Feedback.empty(reference)
    edit = feedback.Feedback.proposal(reference.cases["call-one"])
    draft["cases"].append(edit)
    if damage == "stale":
        draft["corpus_revision"] += 1
    elif damage == "unknown":
        edit["case_id"] = "invented-call"
    elif damage == "duplicate":
        draft["cases"].append(deepcopy(edit))
    elif damage == "lineage":
        edit["passages"][0]["origin_ids"] = ["invented-parent"]
    elif damage == "missing-role":
        edit.update(
            status="complete",
            reviewed_by="Reviewer",
            reviewed_at="2026-11-20T12:00:00Z",
        )
    elif damage == "role":
        for passage in edit["passages"]:
            passage["role_assignment"] = {
                "role": "Invented role",
                "reviewer": "Reviewer",
                "assigned_at": "2026-11-20T12:00:00Z",
            }
        edit["passages"][1]["speaker_change_before"] = True
        edit["passages"][1]["boundary_basis"] = {
            "actor_type": "human",
            "note": "Change",
        }
        edit.update(
            status="complete",
            reviewed_by="Reviewer",
            reviewed_at="2026-11-20T12:00:00Z",
        )
    else:
        edit["status"] = "complete"
        edit["passages"][0]["text"] = ""
    with pytest.raises(ValueError):
        feedback.Feedback.validate(reference, draft)
    assert reference.data == original


@pytest.mark.parametrize("boundary_source", ["change-mark", "roles"])
def test_scoring_ignores_cue_layout_except_evidenced_speaker_changes(
    reference: Any,
    boundary_source: str,
) -> None:
    case = deepcopy(reference.cases["call-one"])
    case["passages"][1]["speaker_change_before"] = True
    case["passages"][1]["boundary_basis"] = {
        "actor_type": "human",
        "note": "Synthetic speaker change",
    }
    if boundary_source == "roles":
        case["passages"][1].update(speaker_change_before=False, boundary_basis=None)
        for passage, role in zip(case["passages"], ("Operator", "Caller")):
            passage["role_assignment"] = {
                "role": role,
                "reviewer": "Reviewer",
                "assigned_at": "2026-11-20T12:00:00Z",
            }
    separate = (
        "1\n00:00:00,000 --> 00:00:02,000\nHELLO, there!\n\n"
        "2\n00:00:01,000 --> 00:00:03,000\nGood evening.\n\n"
    )
    merged = "1\n00:00:00,000 --> 00:00:03,000\nHello there. Good evening.\n\n"
    split = evaluation.Evaluator.compare(case, separate)
    mixed = evaluation.Evaluator.compare(case, merged)
    assert split["provisional_word_difference_rate"] == 0
    assert mixed["provisional_word_difference_rate"] == 0
    assert split["word_error_rate"] is None
    assert split["speaker_boundaries"][0]["status"] == "separate"
    assert mixed["speaker_boundaries"][0]["status"] == "mixed"
    if boundary_source == "roles":
        second = case["passages"][1]
        second["speaker_change_before"] = True
        second["boundary_basis"] = {"actor_type": "human", "note": "Old change mark"}
        second["role_assignment"]["role"] = "Operator"
        assert evaluation.Evaluator.compare(case, merged)["speaker_boundaries"] == []
        second["role_assignment"]["role"] = "Caller"
    missing_boundary_word = merged.replace("there.", "")
    result = evaluation.Evaluator.compare(case, missing_boundary_word)
    assert result["speaker_boundaries"][0]["status"] == "unscorable"
    assert result["mechanical"]["text"]["deletions"] == 1
    assert evaluation.Evaluator.words("Sí, sí. Don't translate 45.") == [
        "sí",
        "sí",
        "don't",
        "translate",
        "45",
    ]
    case["passages"][0]["text"] = "Yes yes."
    case["passages"][1]["text"] = "Yes."
    repeated = "1\n00:00:00,000 --> 00:00:03,000\nYes yes.\n\n"
    result = evaluation.Evaluator.compare(case, repeated)
    boundary = result["speaker_boundaries"][0]
    assert boundary["status"] == "unscorable"
    assert boundary["unscorable_reason"] == "ambiguous word correspondence"
    assert result["mechanical"]["text"]["deletions"] == 1


def test_rescore_uses_old_outputs_and_reports_missing_and_unreviewed_cases(
    reference: Any,
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    subset = tmp_path / "subset"
    subset.mkdir()
    (subset / "call-one.srt").write_bytes((tmp_path / "call-one.srt").read_bytes())
    runs.Runs.retain(
        reference,
        subset,
        output,
        "whisperx",
        {"kind": "fixture"},
        ["call-one", "call-two"],
    )
    first = evaluation.Evaluator.evaluate(
        reference, output / "run.json", tmp_path / "before.json"
    )
    assert first["summary"]["official_scored_cases"] == 0
    assert first["summary"]["statuses"] == {
        "compared": 1,
        "missing": 1,
        "not_selected": 1,
    }
    assert first["summary"]["official_exclusions"] == {"unreviewed reference": 1}
    draft = feedback.Feedback.empty(reference)
    edit = feedback.Feedback.proposal(reference.cases["call-one"])
    edit.update(
        status="complete",
        reviewed_at="2026-11-18T12:00:00Z",
        reviewed_by="Fixture reviewer",
    )
    edit["passages"][0]["text"] = "Hello friend."
    ReferenceFixture.assign_roles(edit)
    draft["cases"] = [edit]
    path = tmp_path / "feedback.json"
    core.Records.write(path, feedback.Feedback.package(reference, draft))
    revised = feedback.Feedback.reconcile(
        reference,
        path,
        [
            {
                "case_id": "call-one",
                "decision": "accept",
                "rationale": "Fixture correction",
            }
        ],
        "Fixture adjudicator",
        tmp_path / "r2",
    )
    second = evaluation.Evaluator.evaluate(
        revised, output / "run.json", tmp_path / "after.json"
    )
    assert second["summary"]["official_scored_cases"] == 1
    assert second["summary"]["official_word_error_rate"] == 0.25
    assert second["run_sha256"] == first["run_sha256"]
    (output / "outputs/call-one.srt").write_text("Changed output")
    with pytest.raises(ValueError, match="Output changed"):
        evaluation.Evaluator.evaluate(
            revised, output / "run.json", tmp_path / "bad.json"
        )


@pytest.mark.parametrize("backend", ["whisperx", "qwen3"])
def test_collection_options_reach_the_actual_transcriber_parser(
    reference: Any,
    tmp_path: Path,
    backend: str,
) -> None:
    from transcription_pipeline.transcriber import TranscriptionCommand

    batch, chunk = (16, 45) if backend == "qwen3" else (None, None)
    command = runs.Runs.command(
        reference,
        tmp_path,
        tmp_path / "out",
        backend,
        None,
        None,
        2,
        batch,
        chunk,
        ["call-one"],
    )
    args = TranscriptionCommand.parser().parse_args(command[1:])
    assert args.audio == [tmp_path / "call-one.wav"]
    assert args.backend == backend
    assert args.processing_limit == 2
    assert args.qwen_batch_size == batch
    assert args.qwen_chunk_seconds == chunk
    assert args.debug_output is True


@pytest.fixture(params=["http", "file"])
def review_site(reference: Any, tmp_path: Path, request: Any) -> Any:
    assets = TOOLS / "assets"
    preparation.Portal.build(reference, tmp_path, assets, tmp_path / "site")
    if request.param == "file":
        yield (tmp_path / "site/index.html").as_uri()
        return
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path / "site"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://review.test:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_site_version_tracks_application_changes_independently_of_corpus(
    reference: Any, tmp_path: Path
) -> None:
    """Distinguish UI builds even when they use exactly the same references."""
    assets = tmp_path / "assets"
    shutil.copytree(TOOLS / "assets", assets)
    first = tmp_path / "first"
    preparation.Portal.build(reference, tmp_path, assets, first)
    before = core.Records.load(first / "build.json")
    script = assets / "review.js"
    script.write_text(script.read_text() + "\nvoid 0;\n")
    second = tmp_path / "second"
    preparation.Portal.build(reference, tmp_path, assets, second)
    after = core.Records.load(second / "build.json")
    assert before["site_version"] != after["site_version"]
    assert before["corpus_sha256"] == after["corpus_sha256"]
    assert before["built_at"] != after["built_at"]
    assert after["site_version"] in (second / "index.html").read_text()
    assert "__SITE_BUILD__" not in (second / "index.html").read_text()
    assert after["assets"]["app.js"] == f"app.{after['site_version']}.js"
    assert after["assets"]["data.js"] == f"data.{after['site_version']}.js"
    generated_assets = {
        path.name for path in second.iterdir() if path.suffix in (".js", ".css")
    }
    assert generated_assets == set(after["assets"].values())


def test_cached_javascript_cannot_survive_a_site_upgrade(
    browser: Any, reference: Any, tmp_path: Path
) -> None:
    """Load new validation code even when the old unscoped script is cached."""
    served = tmp_path / "served"
    served.mkdir()
    (served / "index.html").write_text('<script src="app.js"></script>')
    (served / "app.js").write_text("window.oldReviewApp = true;")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(CachedReviewHandler, directory=str(served))
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = browser.new_context()
    try:
        page = context.new_page()
        page.goto(f"http://review.test:{server.server_port}")
        assert page.evaluate("window.oldReviewApp")
        fresh = tmp_path / "fresh"
        preparation.Portal.build(reference, tmp_path, TOOLS / "assets", fresh)
        metadata = core.Records.load(fresh / "build.json")
        shutil.copytree(fresh, served, dirs_exist_ok=True)
        cached = page.evaluate("async () => (await fetch('app.js')).text()")
        assert cached == "window.oldReviewApp = true;"
        page.reload()
        page.wait_for_function(
            "document.querySelector('#running-version').textContent.startsWith('Running application version')"
        )
        assert page.evaluate("window.reviewAppVersion") == metadata["site_version"]
        assert page.evaluate("typeof window.oldReviewApp") == "undefined"
        page.locator("#reviewer").fill("Cache test reviewer")
        ReferenceFixture.label_recording(page)
        page.get_by_role("button", name="Uncertain wording").first.click()
        page.get_by_label("Passage 1 uncertainty", exact=True).fill(
            "Word is hard to hear."
        )
        page.locator("#listened").check()
        ReferenceFixture.complete_and_return(page)
        assert (
            page.locator("#call-status").inner_text() == "Reviewed · uncertainty noted"
        )
        page.get_by_label("Passage 1 text", exact=True).fill("")
        page.locator("#listened").check()
        page.locator("#complete").click()
        assert page.locator("#message a").count() == 1
    finally:
        context.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_relocated_sources_preserve_hashes_and_reach_docker(
    reference: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    """Protect migrated provenance and external-audio container mounts."""
    audio = tmp_path / "external-audio"
    audio.mkdir()
    monkeypatch.setattr(core.Records, "audio_dir", audio)
    original = tmp_path / "call-one.wav"
    digest = core.Records.file_digest(original)
    original.rename(audio / original.name)
    reference.cases["call-one"]["audio"]["path"] = "audio-files/call-one.wav"
    resolved = core.Records.within(tmp_path, "audio-files/call-one.wav")
    assert resolved == audio / "call-one.wav"
    assert core.Records.file_digest(resolved) == digest
    assert core.Records.within(tmp_path, "reference-audio/call-one.wav") == resolved
    with pytest.raises(ValueError):
        core.Records.within(tmp_path, "reference-audio/../secret")
    command = runs.Runs.command(
        reference,
        tmp_path,
        tmp_path / "out",
        "qwen3",
        "fixture-image",
        None,
        2,
        16,
        45,
        ["call-one"],
    )
    assert f"{audio}:/reference-audio:ro" in command
    assert command[-1] == "/reference-audio/call-one.wav"


@pytest.fixture
def browser() -> Any:
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as manager:
        instance = manager.chromium.launch(
            args=[
                "--host-resolver-rules=MAP review.test 127.0.0.1",
                "--no-proxy-server",
            ]
        )
        try:
            yield instance
        finally:
            instance.close()


@pytest.mark.parametrize(
    "uncertainty,text",
    [
        (" ", "Hello there."),
        ("Name is hard to hear.", "Hello there."),
        ("", "[unclear]"),
    ],
)
def test_legacy_feedback_import_preserves_alternatives_and_attribution(
    browser: Any,
    review_site: str,
    reference: Any,
    tmp_path: Path,
    uncertainty: str,
    text: str,
) -> None:
    """Keep completed reviews importable with uncertainty, alternatives and authors."""
    draft = feedback.Feedback.empty(reference)
    draft["reviewer"] = "Current reviewer"
    edit = feedback.Feedback.proposal(reference.cases["call-one"])
    edit.update(
        status="complete",
        reviewed_by="Original reviewer",
        reviewed_at="2026-11-18T12:00:00Z",
    )
    edit["passages"][0]["uncertainty"] = uncertainty
    edit["passages"][0]["text"] = text
    edit["passages"][0]["alternatives"] = ["Hello there!"]
    ReferenceFixture.assign_roles(edit)
    draft["cases"] = [edit]
    package = feedback.Feedback.package(reference, draft)
    path = tmp_path / "legacy.json"
    core.Records.write(path, package)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto(review_site)
    page.locator("#download").wait_for(state="visible")
    page.locator("#file-input").set_input_files(path)
    page.wait_for_function(
        "document.querySelector('#reviewer').value === 'Current reviewer'"
    )
    assert "Original reviewer" in page.locator("#review-attribution").inner_text()
    with page.expect_download() as downloaded:
        page.locator("#download").click()
    exported = tmp_path / "restored.html"
    downloaded.value.save_as(exported)
    restored = feedback.Feedback.load_package(reference, exported)
    assert restored["draft"]["cases"] == draft["cases"]
    assert 'Proposed alternatives: ["Hello there!"]' in restored["review_markdown"]
    page.locator("#reopen").click()
    page.locator("#listened").check()
    ReferenceFixture.complete_and_return(page)
    unresolved = bool(uncertainty.strip() or "[unclear]" in text)
    expected = "Reviewed · uncertainty noted" if unresolved else "Reviewed"
    assert page.locator("#call-status").inner_text() == expected
    reviewed = deepcopy(reference.cases["call-one"])
    reviewed["passages"] = edit["passages"]
    reviewed["review"]["status"] = "complete"
    score = evaluation.Evaluator.compare(
        reviewed, (tmp_path / "call-one.srt").read_text()
    )
    assert score["word_error_rate"] == (None if unresolved else 0)
    context.close()


def test_tagged_ai_note_requires_review_but_replacement_notes_are_allowed(
    browser: Any, reference: Any, tmp_path: Path
) -> None:
    """Require attention to tagged prompts without blocking a human's findings."""
    data = deepcopy(reference.data)
    data["cases"][0]["passages"][0]["uncertainty"] = "Check the greeting."
    data["cases"][0] = core.Records.seal(data["cases"][0])
    tagged_reference = core.Corpus(core.Records.seal(data))
    site = tmp_path / "tagged-site"
    preparation.Portal.build(tagged_reference, tmp_path, TOOLS / "assets", site)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto((site / "index.html").as_uri())
    note = page.get_by_label("Passage 1 uncertainty", exact=True)
    assert note.input_value() == "[REVIEW NEEDED] Check the greeting."
    page.locator("#reviewer").fill("Fixture reviewer")
    ReferenceFixture.label_recording(page)
    page.locator("#listened").check()
    page.locator("#complete").click()
    link = page.locator("#message a")
    assert link.count() == 1
    link.click()
    assert note.evaluate("node => node === document.activeElement")
    assert page.locator("#call-status").inner_text() == "Unreviewed"
    note.fill("I hear a greeting, but cannot distinguish hello from hi.")
    page.locator("#listened").check()
    ReferenceFixture.complete_and_return(page)
    assert page.locator("#call-status").inner_text() == "Reviewed · uncertainty noted"
    with page.expect_download() as downloaded:
        page.locator("#download").click()
    path = tmp_path / "tagged-feedback.html"
    downloaded.value.save_as(path)
    package = feedback.Feedback.load_package(tagged_reference, path)
    assert (
        package["draft"]["cases"][0]["passages"][0]["uncertainty"] == note.input_value()
    )
    draft = feedback.Feedback.empty(tagged_reference)
    edit = feedback.Feedback.proposal(tagged_reference.cases["call-one"])
    ReferenceFixture.assign_roles(edit)
    edit.update(
        status="complete_with_uncertainty",
        reviewed_by="Reviewer",
        reviewed_at="2026-11-19T00:00:00Z",
    )
    draft["cases"] = [edit]
    with pytest.raises(ValueError, match="Review tagged AI notes"):
        feedback.Feedback.validate(tagged_reference, draft)
    context.close()


def test_completion_errors_link_to_every_empty_passage(
    browser: Any, review_site: str
) -> None:
    """Locate all blockers without losing edits or implying review completion."""
    context = browser.new_context()
    page = context.new_page()
    page.goto(review_site)
    page.locator("#reviewer").fill("Fixture reviewer")
    page.get_by_label("Passage 1 text", exact=True).fill(" ")
    page.get_by_label("Passage 2 text", exact=True).fill("")
    page.locator("#listened").check()
    page.locator("#complete").click()
    links = (
        page.locator("#message li").filter(has_text="Transcript is empty").locator("a")
    )
    assert links.count() == 2
    assert "Block 1" in links.nth(0).inner_text()
    assert "Block 2" in links.nth(1).inner_text()
    links.nth(1).click()
    assert page.get_by_label("Passage 2 text", exact=True).evaluate(
        "node => node === document.activeElement"
    )
    assert page.locator("#call-status").inner_text() == "Unreviewed"
    assert page.get_by_label("Passage 1 text", exact=True).input_value() == " "
    page.locator("#next").click()
    assert page.locator("#message").is_hidden()
    page.locator("#previous").click()
    assert links.count() == 2
    page.get_by_label("Passage 1 text", exact=True).fill("Hello there.")
    assert page.locator("#message").is_hidden()
    page.locator("#next-bottom").click()
    assert page.locator("#message").is_hidden()
    page.locator("#complete").click()
    assert "Confirm that you listened" in page.locator("#message").inner_text()
    page.locator("#previous-bottom").click()
    assert page.locator("#message").is_hidden()
    page.locator("#next-bottom").click()
    assert "Confirm that you listened" in page.locator("#message").inner_text()
    page.locator("#previous-bottom").click()
    page.get_by_label("Passage 2 text", exact=True).fill("Good evening.")
    page.locator("#listened").check()
    page.locator("#complete").click()
    role_links = (
        page.locator("#message li")
        .filter(has_text="Role assignment is required")
        .locator("a")
    )
    assert role_links.count() == 2
    role_links.first.click()
    assert page.locator(".passage").first.locator(".role-tools").is_visible()
    assert page.locator("#assign-operator").evaluate(
        "node => node === document.activeElement"
    )
    ReferenceFixture.label_recording(page)
    page.locator("#listened").check()
    ReferenceFixture.complete_and_return(page)
    assert page.locator("#call-status").inner_text() == "Reviewed"
    context.close()


def test_playback_shortcuts_respect_text_entry_and_override_control_defaults(
    browser: Any, review_site: str, tmp_path: Path
) -> None:
    """Keep transport keys consistent without editing text or activating buttons."""
    with wave.open(str(tmp_path / "site/audio/keyboard.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 8000 * 120)
    context = browser.new_context()
    page = context.new_page()
    page.goto(review_site)
    page.locator("#reviewer").fill("Keyboard reviewer")
    audio_data = b64encode((tmp_path / "site/audio/keyboard.wav").read_bytes()).decode()
    page.locator("#audio").evaluate(
        "(node, data) => { node.src = 'data:audio/wav;base64,' + data; node.load(); }",
        audio_data,
    )
    page.wait_for_function("""() => {
        const audio = document.querySelector('#audio');
        return audio.duration === 120 && audio.readyState === 4 &&
            audio.seekable.length && audio.seekable.end(0) === 120;
    }""")
    page.locator("#complete").focus()
    page.keyboard.press("Space")
    page.wait_for_function("!document.querySelector('#audio').paused")
    assert page.locator("#call-status").inner_text() == "Unreviewed"
    assert page.locator("#message").is_hidden()
    page.keyboard.press("Space")
    page.wait_for_function("document.querySelector('#audio').paused")
    for control in ("#audio", "#recording", "#speed", "#listened"):
        page.locator(control).focus()
        for key, expected in (
            ("h", 58),
            ("ArrowLeft", 58),
            ("l", 62),
            ("ArrowRight", 62),
            ("Shift+h", 50),
            ("Shift+ArrowLeft", 50),
            ("Shift+l", 70),
            ("Shift+ArrowRight", 70),
        ):
            page.locator("#audio").evaluate("node => { node.currentTime = 60; }")
            page.keyboard.press(key)
            assert page.locator("#audio").evaluate(
                "node => node.currentTime"
            ) == pytest.approx(expected), (control, key)
            assert page.locator("#audio").evaluate("node => node.paused")
    assert page.locator("#recording").input_value() == "call-one"
    assert page.locator("#speed").input_value() == "1"
    assert not page.locator("#listened").is_checked()
    page.locator("#audio").evaluate("node => { node.currentTime = 1; }")
    page.keyboard.press("h")
    assert page.locator("#audio").evaluate("node => node.currentTime") == 0
    page.locator("#audio").evaluate("node => { node.currentTime = 119; }")
    page.keyboard.press("Shift+l")
    assert page.locator("#audio").evaluate("node => node.currentTime") == 120
    for field in ("#reviewer", "#notes", ".passage textarea >> nth=0"):
        page.locator("#audio").evaluate(
            "node => { node.pause(); node.currentTime = 60; }"
        )
        editor = page.locator(field)
        editor.fill("")
        editor.press_sequentially("hjklsocurx ")
        editor.press("ArrowLeft")
        editor.press("Shift+ArrowRight")
        assert editor.input_value() == "hjklsocurx "
        assert page.locator("#audio").evaluate("node => node.currentTime") == 60
        assert page.locator("#audio").evaluate("node => node.paused")
        editor.press("Escape")
        assert editor.evaluate("node => node !== document.activeElement")
        page.keyboard.press("Space")
        page.wait_for_function("!document.querySelector('#audio').paused")
        page.keyboard.press("Space")
    page.locator("#audio").focus()
    page.keyboard.down("Space")
    page.keyboard.down("Space")
    page.keyboard.up("Space")
    page.wait_for_function("!document.querySelector('#audio').paused")
    page.keyboard.press("Space")
    page.wait_for_function("document.querySelector('#audio').paused")
    page.locator("#audio").evaluate("node => { node.currentTime = 60; }")
    page.keyboard.press("j")
    page.keyboard.press("k")
    assert page.locator("#audio").evaluate("node => node.currentTime") == 60
    assert page.locator("#audio").evaluate("node => node.paused")
    context.close()


def test_keyboard_block_navigation_and_role_assignment(
    browser: Any, review_site: str, reference: Any, tmp_path: Path
) -> None:
    """Label ranges by keyboard without losing focus, edits or native controls."""
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto(review_site)
    first = page.locator(".passage").nth(0)
    first.locator(".select-block").wait_for()
    assert first.get_attribute("aria-current") == "true"
    page.keyboard.press("s")
    page.keyboard.press("o")
    assert page.locator("#reviewer").evaluate("node => node === document.activeElement")
    assert page.locator(".range-selected").count() == 1
    assert page.locator(".role-badge").first.inner_text() == "Unassigned"
    page.locator("#reviewer").fill("Keyboard reviewer")
    page.keyboard.press("Escape")
    page.keyboard.press("o")
    assert page.locator(".role-badge").first.inner_text() == "Operator"
    assert page.locator(".keyboard-highlighted").get_attribute("data-id") == "p0002"
    assert page.locator(".range-selected").count() == 0
    page.locator("#add-end").click()
    page.get_by_label("Passage 3 text", exact=True).fill("Goodbye.")
    page.keyboard.press("Escape")
    for key, position in (
        ("k", 1),
        ("ArrowUp", 0),
        ("k", 0),
        ("j", 1),
        ("ArrowDown", 2),
        ("j", 2),
    ):
        page.keyboard.press(key)
        card = page.locator(".passage").nth(position)
        assert card.get_attribute("aria-current") == "true"
        assert page.locator(".keyboard-highlighted").count() == 1
        assert page.locator(".range-selected").count() == 0
        assert card.locator(".select-block").evaluate(
            "node => node === document.activeElement"
        )
        center, expected_center = card.evaluate("""node => {
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            const center = (innerHeight + parseFloat(style.scrollMarginTop)) / 2;
            return [(rect.top + rect.bottom) / 2, center];
        }""")
        assert center == pytest.approx(expected_center, abs=4)
    page.keyboard.press("k")
    page.keyboard.press("s")
    page.keyboard.press("j")
    page.keyboard.press("c")
    assert page.locator(".role-badge").all_text_contents() == [
        "Operator",
        "Caller",
        "Caller",
    ]
    assert page.locator("#listened").evaluate("node => node === document.activeElement")
    assert page.locator(".keyboard-highlighted").count() == 0
    page.keyboard.press("Enter")
    assert page.locator("#listened").is_checked()
    page.keyboard.press("Enter")
    assert not page.locator("#listened").is_checked()
    page.get_by_role("button", name="Select block 2", exact=True).focus()
    page.keyboard.press("s")
    page.keyboard.press("k")
    page.keyboard.press("r")
    assert page.locator(".role-badge").all_text_contents() == [
        "Unassigned",
        "Unassigned",
        "Caller",
    ]
    assert page.locator(".passage").nth(2).get_attribute("aria-current") == "true"
    page.keyboard.press("k")
    page.keyboard.press("s")
    page.keyboard.press("u")
    assert page.locator(".role-badge").nth(1).inner_text() == "Unsure"
    page.keyboard.press("k")
    page.keyboard.press("k")
    page.keyboard.press("s")
    page.keyboard.press("x")
    assert page.locator(".range-selected").count() == 0
    assert page.locator(".role-badge").first.inner_text() == "Unassigned"
    assert page.locator(".passage").nth(1).get_attribute("aria-current") == "true"
    page.locator("#next").click()
    page.locator("#previous").click()
    assert first.get_attribute("aria-current") == "true"
    page.keyboard.press("s")
    page.keyboard.press("o")
    page.keyboard.press("j")
    page.keyboard.press("s")
    page.keyboard.press("x")
    assert page.locator("#listened").evaluate("node => node === document.activeElement")
    page.keyboard.press("j")
    assert page.locator("#listened").evaluate("node => node === document.activeElement")
    page.locator("#speed").focus()
    page.keyboard.press("ArrowDown")
    assert page.locator("#speed").input_value() == "1.25"
    assert page.locator(".keyboard-highlighted").count() == 0
    with page.expect_download() as download:
        page.locator("#download").click()
    output = tmp_path / "keyboard-review.html"
    download.value.save_as(output)
    saved = feedback.Feedback.load_package(reference, output)["draft"]["cases"][0]
    assert [p["role_assignment"]["role"] for p in saved["passages"]] == [
        "Operator",
        "Unsure",
        "Caller",
    ]
    assert all(
        p["role_assignment"]["reviewer"] == "Keyboard reviewer"
        for p in saved["passages"]
    )
    context.close()


def test_completion_guidance_follows_recording_status(
    browser: Any, review_site: str
) -> None:
    """Keep completion guidance tied to the active recording across navigation."""
    context = browser.new_context()
    page = context.new_page()
    page.goto(review_site)
    page.locator("#reviewer").fill("Fixture reviewer")
    guidance = page.get_by_text(
        "Review completed. You can continue to the next recording or download your feedback.",
        exact=True,
    )
    for completion in ("#complete", "#complete-unresolved"):
        ReferenceFixture.label_recording(page)
        assert page.locator("#previous-bottom").is_disabled()
        assert guidance.is_hidden()
        page.locator("#listened").check()
        ReferenceFixture.complete_and_return(page, completion)
        assert guidance.is_visible()
        page.locator("#next-bottom").click()
        assert guidance.is_hidden()
        page.locator("#previous-bottom").click()
        assert guidance.is_visible()
        page.wait_for_function(
            "document.querySelector('#save-status').textContent === 'Saved in this browser'"
        )
        page.reload()
        guidance.wait_for(state="visible")
        page.locator("#reopen").click()
        assert guidance.is_hidden()
    context.close()


def test_completion_advances_to_edited_drafts_without_validating_other_recordings(
    browser: Any, review_site: str, reference: Any, tmp_path: Path
) -> None:
    """Prevent cross-recording validation jumps and skipping unfinished edits."""
    draft = feedback.Feedback.empty(reference)
    draft["reviewer"] = "Fixture reviewer"
    draft["cases"] = [
        feedback.Feedback.proposal(case) for case in reference.cases.values()
    ]
    draft["cases"][1]["passages"][0]["text"] = "Keep this edited greeting."
    draft["cases"][1]["notes"] = "Work already in progress."
    package = feedback.Feedback.package(reference, draft)
    package["draft"]["cases"][2].update(
        status="complete", reviewed_by="Reviewer", reviewed_at="2026-11-20T12:00:00Z"
    )
    path = tmp_path / "mixed-progress.json"
    core.Records.write(path, package)
    context = browser.new_context()
    page = context.new_page()
    page.goto(review_site)
    page.locator("#download").wait_for()
    page.locator("#file-input").set_input_files(path)
    page.wait_for_function(
        "document.querySelector('#reviewer').value === 'Fixture reviewer'"
    )
    for identity in ("call-two", "call-three"):
        ReferenceFixture.label_recording(page)
        page.locator("#listened").check()
        page.locator("#complete").click()
        page.wait_for_function(
            "id => document.querySelector('#recording').value === id", arg=identity
        )
        assert page.locator("#message").is_hidden()
        assert page.locator("#completion-message").is_hidden()
        if identity == "call-two":
            assert (
                page.get_by_label("Passage 1 text", exact=True).input_value()
                == "Keep this edited greeting."
            )
            assert page.locator("#notes").input_value() == "Work already in progress."
            page.wait_for_function(
                "document.querySelector('#save-status').textContent === 'Saved in this browser'"
            )
            page.reload()
            page.locator("#previous").click()
            assert page.locator("#call-status").inner_text() == "Reviewed"
            page.locator("#next").click()
            assert (
                page.get_by_label("Passage 1 text", exact=True).input_value()
                == "Keep this edited greeting."
            )
        else:
            assert page.locator("#call-status").inner_text() == "Roles required"
    ReferenceFixture.label_recording(page)
    page.locator("#listened").check()
    page.locator("#complete-unresolved").click()
    page.wait_for_function(
        "document.querySelector('#save-status').textContent === 'Saved in this browser'"
    )
    assert page.locator("#recording").input_value() == "call-three"
    assert page.locator("#completion-message").is_visible()
    assert page.locator("#progress").inner_text() == "3 of 3 reviewed"
    page.locator("#reopen").click()
    page.locator("#recording").select_option("call-one")
    page.locator("#reopen").click()
    page.locator("#listened").check()
    page.locator("#complete").click()
    page.wait_for_function(
        "document.querySelector('#recording').value === 'call-three'"
    )
    assert page.locator("#message").is_hidden()
    page.locator("#recording").select_option("call-two")
    page.locator("#reopen").click()
    page.locator("#recording").select_option("call-three")
    page.locator("#listened").check()
    page.locator("#complete").click()
    page.wait_for_function("document.querySelector('#recording').value === 'call-two'")
    assert page.locator("#message").is_hidden()
    context.close()


def test_browser_review_round_trip_keeps_pending_edits_and_reconciles(
    browser: Any,
    review_site: str,
    reference: Any,
    tmp_path: Path,
) -> None:
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto(review_site)
    page.locator("#reviewer").fill("Fixture reviewer")
    editor = page.get_by_label("Passage 1 text", exact=True)
    editor.fill("Hello friend. Yes please.")
    editor.evaluate("node => node.setSelectionRange(14, 14)")
    page.get_by_role("button", name="Split at cursor").first.click()
    page.get_by_role("button", name="Uncertain wording").nth(1).click()
    page.get_by_label("Passage 2 uncertainty", exact=True).fill(
        "Second phrase needs a human listen."
    )
    page.locator(".passage").nth(1).get_by_label("Possible wording", exact=False).fill(
        "Yes, thanks."
    )
    page.locator("#notes").fill("Keep <script>alert('x')</script> as literal notes.")
    for position, role in enumerate(("operator", "caller", "operator"), 1):
        page.get_by_role("button", name=f"Select block {position}", exact=True).click()
        page.locator(f"#assign-{role}").click()
    page.locator("#listened").check()
    ReferenceFixture.complete_and_return(page)
    with page.expect_download() as downloaded:
        page.locator("#download").click()
    path = tmp_path / "feedback.html"
    downloaded.value.save_as(path)
    package = feedback.Feedback.load_package(reference, path)
    edit = package["draft"]["cases"][0]
    assert edit["status"] == "complete_with_uncertainty"
    assert 'Proposed alternatives: ["Yes, thanks."]' in package["review_markdown"]
    assert "Completed by: Fixture reviewer" in package["review_markdown"]
    assert [p["text"] for p in edit["passages"]] == [
        "Hello friend.",
        "Yes please.",
        "Good evening.",
    ]
    assert edit["passages"][1]["origin_ids"] == ["p0001"]
    assert [p["role_assignment"]["role"] for p in edit["passages"]] == [
        "Operator",
        "Caller",
        "Operator",
    ]
    assert all(
        p["role_assignment"]["reviewer"] == "Fixture reviewer" for p in edit["passages"]
    )
    assert "Role: Unassigned → Caller" in package["review_markdown"]
    page.locator("#next").click()
    page.locator("#previous").click()
    assert (
        page.get_by_label("Passage 1 text", exact=True).input_value() == "Hello friend."
    )
    page.wait_for_function(
        "document.querySelector('#save-status').textContent === 'Saved in this browser'"
    )
    page.reload()
    assert page.locator(".role-badge").all_text_contents() == [
        "Operator",
        "Caller",
        "Operator",
    ]
    assert (
        page.get_by_label("Passage 1 text", exact=True).input_value() == "Hello friend."
    )
    fresh_context = browser.new_context(accept_downloads=True)
    fresh = fresh_context.new_page()
    fresh.goto(review_site)
    fresh.locator("#download").wait_for(state="visible")
    fresh.locator("#file-input").set_input_files(path)
    fresh.wait_for_function(
        "document.querySelector('#reviewer').value === 'Fixture reviewer'"
    )
    assert (
        fresh.get_by_label("Passage 2 text", exact=True).input_value() == "Yes please."
    )
    with fresh.expect_download() as round_trip:
        fresh.locator("#download").click()
    restored = tmp_path / "restored.html"
    round_trip.value.save_as(restored)
    assert (
        feedback.Feedback.load_package(reference, restored)["draft"]["cases"]
        == package["draft"]["cases"]
    )
    revised = feedback.Feedback.reconcile(
        reference,
        path,
        [
            {
                "case_id": "call-one",
                "decision": "accept",
                "rationale": "Synthetic browser round-trip acceptance",
            }
        ],
        "Fixture adjudicator",
        tmp_path / "accepted",
    )
    assert revised.cases["call-one"]["review"]["status"] == "complete_with_uncertainty"
    assert revised.cases["call-one"]["passages"] == edit["passages"]
    fresh.set_viewport_size({"width": 390, "height": 844})
    fresh.locator("#notes").scroll_into_view_if_needed()
    assert fresh.locator("#download").is_visible()
    assert fresh.evaluate(
        "document.documentElement.scrollWidth <= window.innerWidth + 1"
    )
    fresh.evaluate("document.documentElement.style.fontSize = '200%'")
    fresh.wait_for_function(
        "document.querySelector('.player').classList.contains('inline-player')"
    )
    assert fresh.evaluate(
        "document.documentElement.scrollWidth <= window.innerWidth + 1"
    )
    fresh.get_by_label("Passage 1 text", exact=True).focus()
    assert fresh.get_by_label("Passage 1 text", exact=True).evaluate(
        "node => node.getBoundingClientRect().top >= "
        + "document.querySelector('.review-header').getBoundingClientRect().bottom"
        + " && node.getBoundingClientRect().top < window.innerHeight - 40"
    )
    context.close()
    fresh_context.close()


def test_role_ranges_are_bounded_and_replace_legacy_speaker_decisions(
    browser: Any, review_site: str, reference: Any, tmp_path: Path
) -> None:
    """Keep range edits bounded and let roles supersede imported change marks."""
    draft = feedback.Feedback.empty(reference)
    draft["reviewer"] = "Role reviewer"
    proposal = feedback.Feedback.proposal(reference.cases["call-one"])
    proposal["passages"][1].update(
        speaker_change_before=True,
        boundary_basis={"actor_type": "human", "note": "Legacy change mark"},
    )
    draft["cases"] = [proposal]
    source = tmp_path / "old-boundary.json"
    package = feedback.Feedback.package(reference, draft)
    package["draft"] = draft
    core.Records.write(source, package)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto(review_site)
    page.locator("#download").wait_for(state="visible")
    page.locator("#file-input").set_input_files(source)
    page.wait_for_function(
        "document.querySelector('#reviewer').value === 'Role reviewer'"
    )
    page.locator("#add-end").click()
    page.get_by_label("Passage 3 text", exact=True).fill("Goodbye.")
    page.get_by_role("button", name="Select block 2", exact=True).click()
    assert page.locator(".passage").nth(1).locator(".role-tools").is_visible()
    assert page.locator(".role-tools").evaluate(
        "node => getComputedStyle(node).position === 'static' && "
        + "node.previousElementSibling.classList.contains('passage-head') && "
        + "node.nextElementSibling.tagName === 'TEXTAREA'"
    )
    page.get_by_role(
        "button", name="Extend selection through block 1", exact=True
    ).click()
    assert (
        "Blocks 1–2 selected · 2 blocks"
        in page.locator("#range-description").inner_text()
    )
    assert page.locator(".range-selected").count() == 2
    assert page.locator(".passage").nth(0).locator(".role-tools").is_visible()
    assert page.locator(".role-tools").count() == 1
    page.locator("#assign-caller").focus()
    for key, endpoint, count in (
        ("ArrowUp", 0, 2),
        ("j", 1, 1),
        ("ArrowDown", 2, 2),
        ("ArrowDown", 2, 2),
        ("k", 1, 1),
        ("ArrowUp", 0, 2),
    ):
        page.keyboard.press(key)
        assert page.locator(".range-selected").count() == count
        assert (
            page.locator(".passage").nth(endpoint).locator(".role-tools").is_visible()
        )
        assert page.locator("#assign-caller").evaluate(
            "node => node === document.activeElement"
        )
    editor = page.get_by_label("Passage 1 text", exact=True)
    editor.press("ArrowDown")
    assert page.locator(".passage").nth(0).locator(".role-tools").is_visible()
    editor.press("Escape")
    page.keyboard.press("j")
    assert page.locator(".range-selected").count() == 1
    page.keyboard.press("k")
    assert page.locator(".range-selected").count() == 2
    assert page.locator("#audio").evaluate("node => node.paused")
    page.locator("#assign-operator").click()
    assert page.locator(".role-badge").all_text_contents() == [
        "Operator",
        "Operator",
        "Unassigned",
    ]
    assert page.locator("#role-selection").is_hidden()
    page.locator("#listened").check()
    page.locator("#complete").click()
    assert page.locator("#call-status").inner_text() == "Unreviewed"
    assert "Block 3" in page.locator("#message a").inner_text()
    page.locator("#message a").click()
    page.locator("#assign-unsure").click()
    page.locator("#listened").check()
    ReferenceFixture.complete_and_return(page)
    assert page.locator("#call-status").inner_text() == "Reviewed"
    assert page.locator("#message").is_hidden()
    assert page.locator(".role-continuation").count() == 1
    page.get_by_label("Passage 1 text", exact=True).fill("Hello, corrected.")
    page.get_by_role("button", name="Select block 2", exact=True).click()
    page.locator("#assign-caller").click()
    page.get_by_role("button", name="Select block 2", exact=True).click()
    page.locator("#assign-operator").click()
    assert (
        page.get_by_label("Passage 1 text", exact=True).input_value()
        == "Hello, corrected."
    )
    assert page.locator(".role-continuation").count() == 1
    page.get_by_role("button", name="Select block 2", exact=True).click()
    page.get_by_role("button", name="Select block 3", exact=True).click(
        modifiers=["Shift"]
    )
    assert page.locator(".passage").nth(2).locator(".role-tools").is_visible()
    page.locator("#assign-caller").click()
    assert page.locator(".role-badge").all_text_contents() == [
        "Operator",
        "Caller",
        "Caller",
    ]
    assert page.get_by_text("Speaker change · from roles", exact=True).count() == 1
    editor = page.get_by_label("Passage 2 text", exact=True)
    editor.evaluate("node => node.setSelectionRange(5, 5)")
    page.get_by_role("button", name="Split at cursor").nth(1).click()
    assert page.locator(".role-badge").all_text_contents() == [
        "Operator",
        "Caller",
        "Unassigned",
        "Caller",
    ]
    page.locator("#next-unlabeled").click()
    assert "Blocks 3–3 selected" in page.locator("#range-description").inner_text()
    page.locator("#assign-unsure").click()
    assert "1 unsure · 0 unassigned" in page.locator("#role-progress").inner_text()
    page.get_by_role("button", name="Select block 4", exact=True).click()
    page.locator("#clear-roles").click()
    assert "1 unsure · 1 unassigned" in page.locator("#role-progress").inner_text()
    page.locator("#listened").check()
    page.locator("#complete").click()
    assert "Block 4" in page.locator("#message a").inner_text()
    page.locator("#message a").click()
    page.locator("#assign-unsure").click()
    page.locator("#listened").check()
    ReferenceFixture.complete_and_return(page)
    assert page.locator("#call-status").inner_text() == "Reviewed"
    with page.expect_download() as downloaded:
        page.locator("#download").click()
    path = tmp_path / "role-ranges.html"
    downloaded.value.save_as(path)
    edit = feedback.Feedback.load_package(reference, path)["draft"]["cases"][0]
    assert all(
        "speaker_change_before" not in p and "boundary_basis" not in p
        for p in edit["passages"]
    )
    page.wait_for_function(
        "document.querySelector('#save-status').textContent === 'Saved in this browser'"
    )
    stored = page.evaluate("""async () => {
        const db = await new Promise(resolve => {
            const request = indexedDB.open('transcription-reviews-v1', 1);
            request.onsuccess = () => resolve(request.result);
        });
        const rows = await new Promise(resolve => {
            const request = db.transaction('drafts').objectStore('drafts').getAll();
            request.onsuccess = () => resolve(request.result);
        });
        db.close();
        return rows.filter(row => row.draft).flatMap(row => row.draft.cases);
    }""")
    assert stored == [edit]
    assert [p.get("role_assignment", {}).get("role") for p in edit["passages"]] == [
        "Operator",
        "Caller",
        "Unsure",
        "Unsure",
    ]
    page.get_by_role("button", name="Select block 1", exact=True).click()
    page.locator("#next").click()
    assert page.locator("#role-selection").is_hidden()
    assert page.locator(".role-badge").all_text_contents() == [
        "Unassigned",
        "Unassigned",
    ]
    context.close()


def test_browser_storage_failure_still_exports_latest_incomplete_text(
    browser: Any,
    review_site: str,
    reference: Any,
    tmp_path: Path,
) -> None:
    context = browser.new_context(accept_downloads=True)
    context.add_init_script(
        "Object.defineProperty(window, 'indexedDB', "
        + "{get() {throw new Error('Synthetic storage unavailable')}})"
    )
    page = context.new_page()
    page.goto(review_site)
    page.get_by_label("Passage 1 text", exact=True).fill("")
    with page.expect_download() as downloaded:
        page.locator("#download").click()
    path = tmp_path / "unfinished.html"
    downloaded.value.save_as(path)
    package = feedback.Feedback.load_package(reference, path)
    assert package["draft"]["cases"][0]["passages"][0]["text"] == ""
    assert "Only in this page" in page.locator("#save-status").inner_text()
    assert page.locator("#call-status").inner_text() == "Unreviewed"
    page.locator("#reviewer").fill("Fixture reviewer")
    page.get_by_label("Passage 1 text", exact=True).fill("Hello there.")
    ReferenceFixture.label_recording(page)
    page.locator("#listened").check()
    page.locator("#complete").click()
    assert page.locator("#call-status").inner_text() == "Reviewed"
    assert page.locator("#recording").input_value() == "call-one"
    assert "Only in this page" in page.locator("#save-status").inner_text()
    context.close()


def test_browser_competing_tabs_keep_both_versions(
    browser: Any,
    review_site: str,
    tmp_path: Path,
    reference: Any,
) -> None:
    context = browser.new_context(accept_downloads=True)
    first = context.new_page()
    first.goto(review_site)
    first.wait_for_function(
        "document.querySelector('#save-status').textContent === 'Saved in this browser'"
    )
    second = context.new_page()
    second.goto(review_site)
    second.get_by_label("Passage 1 text", exact=True).wait_for()
    first.get_by_label("Passage 1 text", exact=True).fill("First reviewer text.")
    first.wait_for_function(
        "document.querySelector('#save-status').textContent === 'Saved in this browser'"
    )
    second.wait_for_function(
        "document.querySelector('#save-status').textContent.includes('Another tab')"
    )
    second.get_by_label("Passage 1 text", exact=True).fill("Second reviewer text.")
    second.locator("#files summary").click()
    second.locator("#separate").click()
    second.wait_for_function(
        "document.querySelector('#save-status').textContent === 'Saved in this browser'"
    )
    for number, page in enumerate((first, second)):
        with page.expect_download() as downloaded:
            page.locator("#download").click()
        path = tmp_path / f"tab-{number}.html"
        downloaded.value.save_as(path)
    one = feedback.Feedback.load_package(reference, tmp_path / "tab-0.html")["draft"]
    two = feedback.Feedback.load_package(reference, tmp_path / "tab-1.html")["draft"]
    assert one["session_id"] != two["session_id"]
    assert one["cases"][0]["passages"][0]["text"] == "First reviewer text."
    assert two["cases"][0]["passages"][0]["text"] == "Second reviewer text."
    context.close()


def test_browser_later_save_failure_and_stale_import_preserve_current_edits(
    browser: Any,
    review_site: str,
    reference: Any,
    tmp_path: Path,
) -> None:
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto(review_site)
    page.wait_for_function(
        "document.querySelector('#save-status').textContent === 'Saved in this browser'"
    )
    page.evaluate(
        "() => { IDBDatabase.prototype.transaction = function() "
        + "{ throw new Error('Synthetic quota failure'); }; }"
    )
    page.get_by_label("Passage 1 text", exact=True).fill("Keep my latest words.")
    page.wait_for_function(
        "document.querySelector('#save-status').textContent"
        + ".includes('Only in this page')"
    )
    stale = feedback.Feedback.package(reference, feedback.Feedback.empty(reference))
    stale["draft"]["corpus_revision"] += 1
    core.Records.write(tmp_path / "stale.json", stale)
    page.locator("#file-input").set_input_files(tmp_path / "stale.json")
    page.wait_for_function(
        "document.querySelector('#message').textContent.includes('different reference')"
    )
    assert page.get_by_label("Passage 1 text", exact=True).input_value() == (
        "Keep my latest words."
    )
    with page.expect_download() as downloaded:
        page.locator("#download").click()
    path = tmp_path / "after-failure.html"
    downloaded.value.save_as(path)
    restored = feedback.Feedback.load_package(reference, path)["draft"]
    assert restored["cases"][0]["passages"][0]["text"] == "Keep my latest words."
    context.close()
