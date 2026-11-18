"""Protect reference integrity, honest scoring, and the human feedback round trip."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib
from pathlib import Path
import sys
from threading import Thread
from typing import Any
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
    "damage", ["stale", "unknown", "duplicate", "lineage", "blank-completion"]
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
    else:
        edit["status"] = "complete"
        edit["passages"][0]["text"] = ""
    with pytest.raises(ValueError):
        feedback.Feedback.validate(reference, draft)
    assert reference.data == original


def test_scoring_ignores_cue_layout_except_evidenced_speaker_changes(
    reference: Any,
) -> None:
    case = deepcopy(reference.cases["call-one"])
    case["passages"][1]["speaker_change_before"] = True
    case["passages"][1]["boundary_basis"] = {
        "actor_type": "human",
        "note": "Synthetic speaker change",
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
        assert guidance.is_hidden()
        page.locator("#listened").check()
        page.locator(completion).click()
        assert guidance.is_visible()
        page.locator("#next").click()
        assert guidance.is_hidden()
        page.locator("#previous").click()
        assert guidance.is_visible()
        page.wait_for_function(
            "document.querySelector('#save-status').textContent === 'Saved in this browser'"
        )
        page.reload()
        guidance.wait_for(state="visible")
        page.locator("#reopen").click()
        assert guidance.is_hidden()
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
    page.locator("#notes").fill("Keep <script>alert('x')</script> as literal notes.")
    page.locator("#listened").check()
    page.locator("#complete-unresolved").click()
    with page.expect_download() as downloaded:
        page.locator("#download").click()
    path = tmp_path / "feedback.html"
    downloaded.value.save_as(path)
    package = feedback.Feedback.load_package(reference, path)
    edit = package["draft"]["cases"][0]
    assert edit["status"] == "complete_with_uncertainty"
    assert [p["text"] for p in edit["passages"]] == [
        "Hello friend.",
        "Yes please.",
        "Good evening.",
    ]
    assert edit["passages"][1]["speaker_change_before"]
    assert edit["passages"][1]["origin_ids"] == ["p0001"]
    page.locator("#next").click()
    page.locator("#previous").click()
    assert (
        page.get_by_label("Passage 1 text", exact=True).input_value() == "Hello friend."
    )
    page.wait_for_function(
        "document.querySelector('#save-status').textContent === 'Saved in this browser'"
    )
    page.reload()
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
