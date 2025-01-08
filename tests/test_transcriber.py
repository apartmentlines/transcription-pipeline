import pytest
import numpy as np
from unittest.mock import Mock, patch
from transcription_pipeline.transcriber import TranscriptionError, Transcriber
from transcription_pipeline.constants import DEFAULT_WHISPER_MODEL, DEFAULT_BATCH_SIZE


# Mock cuda availability for all tests
@pytest.fixture(autouse=True)
def mock_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_transcription_result():
    return {
        "segments": [
            {
                "text": "Hello world",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.5, "end": 1.0},
                ],
            }
        ],
        "language": "en",
    }


@pytest.fixture
def mock_alignment_result():
    return {
        "segments": [
            {
                "text": "Hello world",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.9},
                    {"word": "world", "start": 0.5, "end": 1.0, "score": 0.95},
                ],
            }
        ],
        "language": "en",
    }


@pytest.fixture
def mock_diarization_result():
    return {"segments": [{"speaker": "SPEAKER_0", "start": 0.0, "end": 1.0}]}


def test_transcription_error_message():
    original_exception = ValueError("Invalid input format")
    with pytest.raises(TranscriptionError) as exc_info:
        raise TranscriptionError(original_exception)
    assert str(exc_info.value) == "Transcription error: Invalid input format"


def test_init_defaults():
    transcriber = Transcriber()
    assert transcriber.whisper_model_name == DEFAULT_WHISPER_MODEL
    assert transcriber.device in ["cuda", "cpu"]
    assert transcriber.compute_type in ["float16", "int8"]


def test_init_custom_values():
    mock_whisperx = Mock()
    transcriber = Transcriber(
        whisper_model_name="custom-model",
        device="cpu",
        compute_type="int8",
        whisperx_module=mock_whisperx,
    )
    assert transcriber.whisper_model_name == "custom-model"
    assert transcriber.device == "cpu"
    assert transcriber.compute_type == "int8"


def test_initialize_whisper_model():
    mock_whisperx = Mock()
    transcriber = Transcriber(whisperx_module=mock_whisperx)
    mock_whisperx.load_model.assert_called_once_with(
        DEFAULT_WHISPER_MODEL, transcriber.device, compute_type=transcriber.compute_type
    )


def test_initialize_diarization_model_with_model():
    mock_whisperx = Mock()
    Transcriber(
        diarization_model_name="pyannote/speaker-diarization",
        whisperx_module=mock_whisperx,
        auth_token="test-token",
    )
    mock_whisperx.DiarizationPipeline.assert_called_once_with(
        model_name="pyannote/speaker-diarization",
        use_auth_token="test-token",
        device="cpu",
    )


def test_initialize_diarization_model_without_model():
    mock_whisperx = Mock()
    transcriber = Transcriber(whisperx_module=mock_whisperx)
    assert transcriber.diarization_model is None


def test_initialize_diarization_model_error():
    mock_whisperx = Mock()
    mock_whisperx.DiarizationPipeline.side_effect = Exception(
        "Pipeline initialization failed"
    )

    with pytest.raises(Exception, match="Pipeline initialization failed"):
        Transcriber(diarization_model_name="test-model", whisperx_module=mock_whisperx)


def test_validate_input_file(tmp_path):
    test_file = tmp_path / "test.wav"
    test_file.touch()

    transcriber = Transcriber()
    assert transcriber._validate_input_file(test_file) == test_file

    with pytest.raises(FileNotFoundError):
        transcriber._validate_input_file("nonexistent.wav")


def test_load_audio():
    mock_whisperx = Mock()
    mock_whisperx.load_audio.return_value = np.array([1.0, 2.0, 3.0])

    transcriber = Transcriber(whisperx_module=mock_whisperx)
    result = transcriber._load_audio("test.wav")

    assert isinstance(result, np.ndarray)
    mock_whisperx.load_audio.assert_called_once_with("test.wav")


def test_load_audio_error():
    mock_whisperx = Mock()
    mock_whisperx.load_audio.side_effect = Exception("Load failed")

    transcriber = Transcriber(whisperx_module=mock_whisperx)
    with pytest.raises(TranscriptionError):
        transcriber._load_audio("test.wav")


def test_perform_base_transcription():
    mock_whisperx = Mock()
    mock_model = Mock()
    mock_model.transcribe.return_value = {"segments": [{"text": "test"}]}

    transcriber = Transcriber(whisperx_module=mock_whisperx)
    transcriber.model = mock_model

    test_audio = np.array([1.0, 2.0])
    result = transcriber._perform_base_transcription(test_audio)
    assert "segments" in result

    # Get the actual call arguments
    call_args = mock_model.transcribe.call_args
    assert call_args is not None
    args, kwargs = call_args

    # Verify the arguments
    assert np.array_equal(args[0], test_audio)
    assert kwargs.get("batch_size") == DEFAULT_BATCH_SIZE


def test_perform_base_transcription_error():
    mock_whisperx = Mock()
    mock_model = Mock()
    mock_model.transcribe.side_effect = Exception("Transcription failed")

    transcriber = Transcriber(whisperx_module=mock_whisperx)
    transcriber.model = mock_model

    with pytest.raises(TranscriptionError):
        transcriber._perform_base_transcription(np.array([1.0, 2.0]))


def test_align_transcription():
    mock_whisperx = Mock()
    mock_whisperx.load_align_model.return_value = (Mock(), Mock())
    mock_whisperx.align.return_value = {"segments": [{"text": "aligned"}]}

    transcriber = Transcriber(whisperx_module=mock_whisperx)
    test_audio = np.array([1.0, 2.0])
    result = transcriber._align_transcription([{"text": "test"}], test_audio, "en")

    assert "segments" in result
    mock_whisperx.load_align_model.assert_called_once()

    # Get the actual call arguments
    call_args = mock_whisperx.align.call_args
    assert call_args is not None
    args, kwargs = call_args

    # Verify the arguments
    assert args[0] == [{"text": "test"}]
    assert args[1] == mock_whisperx.load_align_model.return_value[0]
    assert args[2] == mock_whisperx.load_align_model.return_value[1]
    assert np.array_equal(args[3], test_audio)
    assert args[4] == "cpu"
    assert kwargs.get("return_char_alignments") is False


def test_align_transcription_load_model_error():
    mock_whisperx = Mock()
    mock_whisperx.load_align_model.side_effect = Exception("Model load failed")

    transcriber = Transcriber(whisperx_module=mock_whisperx)
    with pytest.raises(Exception, match="Model load failed"):
        transcriber._align_transcription([{"text": "test"}], np.array([1.0, 2.0]), "en")


def test_align_transcription_align_error():
    mock_whisperx = Mock()
    mock_whisperx.load_align_model.return_value = (Mock(), Mock())
    mock_whisperx.align.side_effect = Exception("Alignment failed")

    transcriber = Transcriber(whisperx_module=mock_whisperx)
    with pytest.raises(TranscriptionError):
        transcriber._align_transcription([{"text": "test"}], np.array([1.0, 2.0]), "en")


def test_perform_diarization():
    mock_diarization_model = Mock()
    mock_diarization_model.return_value = {"segments": [{"speaker": "SPEAKER_1"}]}

    transcriber = Transcriber()
    transcriber.diarization_model = mock_diarization_model

    result = transcriber._perform_diarization(np.array([1.0, 2.0]), 2)
    assert "segments" in result
    mock_diarization_model.assert_called_once()


def test_perform_diarization_error():
    mock_diarization_model = Mock()
    mock_diarization_model.side_effect = Exception("Diarization failed")

    transcriber = Transcriber()
    transcriber.diarization_model = mock_diarization_model

    with pytest.raises(TranscriptionError):
        transcriber._perform_diarization(np.array([1.0, 2.0]), 2)


def test_assign_speakers():
    mock_whisperx = Mock()
    mock_whisperx.assign_word_speakers.return_value = {
        "segments": [{"speaker": "SPEAKER_1"}]
    }

    transcriber = Transcriber(whisperx_module=mock_whisperx)
    result = transcriber._assign_speakers(
        {"segments": [{"speaker": "SPEAKER_1"}]},
        {"segments": [{"text": "test"}], "language": "en"},
    )

    assert "segments" in result
    assert "language" in result
    mock_whisperx.assign_word_speakers.assert_called_once()


def test_assign_speakers_error():
    mock_whisperx = Mock()
    mock_whisperx.assign_word_speakers.side_effect = Exception(
        "Speaker assignment failed"
    )

    transcriber = Transcriber(whisperx_module=mock_whisperx)
    with pytest.raises(TranscriptionError):
        transcriber._assign_speakers(
            {"segments": [{"speaker": "SPEAKER_1"}]},
            {"segments": [{"text": "test"}], "language": "en"},
        )


def test_handle_diarization_with_model():
    mock_whisperx = Mock()
    transcriber = Transcriber(
        whisperx_module=mock_whisperx, diarization_model_name="test-model"
    )

    transcriber._perform_diarization = Mock(return_value={"segments": []})
    transcriber._assign_speakers = Mock(return_value={"segments": [], "language": "en"})

    result = transcriber._handle_diarization(
        np.array([1.0, 2.0]), {"segments": [], "language": "en"}, 2
    )

    assert "segments" in result
    transcriber._perform_diarization.assert_called_once()
    transcriber._assign_speakers.assert_called_once()


def test_handle_diarization_without_model():
    transcriber = Transcriber()  # No diarization model
    aligned_result = {"segments": [], "language": "en"}
    result = transcriber._handle_diarization(np.array([1.0, 2.0]), aligned_result, 2)
    assert result == aligned_result


def test_handle_diarization_empty_segments():
    mock_whisperx = Mock()
    transcriber = Transcriber(
        whisperx_module=mock_whisperx, diarization_model_name="test-model"
    )

    transcriber._perform_diarization = Mock(return_value={"segments": []})
    transcriber._assign_speakers = Mock(return_value={"segments": [], "language": "en"})

    result = transcriber._handle_diarization(
        np.array([1.0, 2.0]),
        {"segments": [], "language": "en"},
        0,  # Edge case: no speakers
    )

    assert result["segments"] == []


def test_save_output(tmp_path):
    mock_writer = Mock()
    with patch(
        "transcription_pipeline.transcriber.get_writer", return_value=mock_writer
    ):
        transcriber = Transcriber()
        result = {"segments": []}

        transcriber._save_output(result, "test.wav", tmp_path, "srt")

        mock_writer.assert_called_once_with(
            {"segments": []},
            "test.wav",
            {
                "max_line_width": None,
                "max_line_count": None,
                "highlight_words": False,
            },
        )


def test_save_output_no_directory():
    transcriber = Transcriber()
    result = {"segments": []}

    transcriber._save_output(result, "test.wav", None, "srt")


def test_save_output_writer_error(tmp_path):
    mock_writer = Mock()
    mock_writer.side_effect = Exception("Write failed")

    with patch(
        "transcription_pipeline.transcriber.get_writer", return_value=mock_writer
    ):
        transcriber = Transcriber()
        result = {"segments": []}

        with pytest.raises(Exception, match="Write failed"):
            transcriber._save_output(result, "test.wav", tmp_path, "srt")


def test_transcribe_integration_error_propagation(tmp_path, mock_transcription_result):
    test_file = tmp_path / "test.wav"
    test_file.touch()

    mock_whisperx = Mock()
    mock_whisperx.load_audio.return_value = np.array([1.0, 2.0])
    mock_whisperx.load_align_model.return_value = (Mock(), Mock())
    mock_whisperx.align.side_effect = Exception("Alignment failed")

    mock_model = Mock()
    mock_model.transcribe.return_value = mock_transcription_result

    transcriber = Transcriber(
        whisperx_module=mock_whisperx, diarization_model_name="test_model"
    )
    transcriber.model = mock_model

    with pytest.raises(TranscriptionError) as exc:
        transcriber.transcribe(test_file)
    assert "Alignment failed" in str(exc.value)


def test_transcribe_integration(
    tmp_path, mock_transcription_result, mock_alignment_result, mock_diarization_result
):
    # Create test file
    test_file = tmp_path / "test.wav"
    test_file.touch()

    # Setup mocks
    mock_whisperx = Mock()
    mock_whisperx.load_audio.return_value = np.array([1.0, 2.0])
    mock_whisperx.load_align_model.return_value = (Mock(), Mock())

    # Configure mock returns
    mock_model = Mock()
    mock_model.transcribe.return_value = mock_transcription_result
    mock_whisperx.align.return_value = mock_alignment_result

    # Setup final result structure
    final_result = {
        "segments": [
            {
                "text": "Hello world",
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_0",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.9},
                    {"word": "world", "start": 0.5, "end": 1.0, "score": 0.95},
                ],
            }
        ],
        "language": "en",
    }
    mock_whisperx.assign_word_speakers.return_value = final_result

    # Create and configure transcriber
    transcriber = Transcriber(
        whisperx_module=mock_whisperx, diarization_model_name="test_model"
    )
    transcriber.model = mock_model
    transcriber.diarization_model = Mock(return_value=mock_diarization_result)

    # Perform transcription
    result = transcriber.transcribe(test_file)

    # Verify pipeline flow
    mock_whisperx.load_audio.assert_called_once_with(test_file)
    mock_model.transcribe.assert_called_once()
    mock_whisperx.load_align_model.assert_called_once()
    mock_whisperx.align.assert_called_once()
    transcriber.diarization_model.assert_called_once()
    mock_whisperx.assign_word_speakers.assert_called_once()

    # Verify content flow
    assert result["segments"][0]["text"] == "Hello world"
    assert result["segments"][0]["speaker"] == "SPEAKER_0"

    # Verify timestamp consistency
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == 1.0

    # Verify word-level details maintained
    words = result["segments"][0]["words"]
    assert len(words) == 2
    assert words[0]["word"] == "Hello"
    assert words[1]["word"] == "world"
    assert all(isinstance(word["score"], float) for word in words)
