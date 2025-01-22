import pytest
import numpy as np
from unittest.mock import Mock, patch
from transcription_pipeline.transcriber import TranscriptionError, Transcriber
from transcription_pipeline.constants import DEFAULT_WHISPER_MODEL, DEFAULT_BATCH_SIZE


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


def test_transcription_error_is_gpu_error():
    # Test GPU errors
    error1 = TranscriptionError("CUDA failed with error out of memory")
    assert error1.is_gpu_error() is True

    error2 = TranscriptionError("cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED")
    assert error2.is_gpu_error() is True

    # Test non-GPU errors
    error3 = TranscriptionError("File not found")
    assert error3.is_gpu_error() is False


def test_init_defaults():
    transcriber = Transcriber()
    assert transcriber.whisper_model_name == DEFAULT_WHISPER_MODEL
    assert transcriber.device == "cpu"  # We know it's CPU because cuda is mocked False
    assert (
        transcriber.compute_type == "int8"
    )  # We know it's int8 because cuda is mocked False


def test_init_custom_values():
    transcriber = Transcriber(
        whisper_model_name="custom-model",
        device="cpu",
        compute_type="int8",
    )
    assert transcriber.whisper_model_name == "custom-model"
    assert transcriber.device == "cpu"
    assert transcriber.compute_type == "int8"


def test_initialize_whisper_model(mock_dependencies):
    _, mock_whisperx, _ = mock_dependencies.return_value
    transcriber = Transcriber()
    mock_whisperx.load_model.assert_called_once_with(
        DEFAULT_WHISPER_MODEL, transcriber.device, compute_type=transcriber.compute_type
    )


def test_initialize_diarization_model_with_model(mock_dependencies):
    _, mock_whisperx, _ = mock_dependencies.return_value
    Transcriber(
        diarization_model_name="pyannote/speaker-diarization",
        auth_token="test-token",
    )
    mock_whisperx.DiarizationPipeline.assert_called_once_with(
        model_name="pyannote/speaker-diarization",
        use_auth_token="test-token",
        device="cpu",
    )


def test_initialize_diarization_model_without_model():
    transcriber = Transcriber()
    assert transcriber.diarization_model is None


def test_initialize_diarization_model_error(mock_dependencies):
    _, mock_whisperx, _ = mock_dependencies.return_value
    mock_whisperx.DiarizationPipeline.side_effect = Exception(
        "Pipeline initialization failed"
    )

    with pytest.raises(Exception, match="Pipeline initialization failed"):
        Transcriber(diarization_model_name="test-model")


def test_validate_input_file(tmp_path):
    test_file = tmp_path / "test.wav"
    test_file.touch()

    transcriber = Transcriber()
    assert transcriber._validate_input_file(test_file) == test_file

    with pytest.raises(FileNotFoundError):
        transcriber._validate_input_file("nonexistent.wav")


def test_load_audio(mock_dependencies):
    _, mock_whisperx, _ = mock_dependencies.return_value
    mock_array = Mock()
    mock_whisperx.load_audio.return_value = mock_array

    transcriber = Transcriber()
    result = transcriber._load_audio("test.wav")

    assert result == mock_array
    mock_whisperx.load_audio.assert_called_once_with("test.wav")


def test_load_audio_error(mock_dependencies):
    _, mock_whisperx, _ = mock_dependencies.return_value
    mock_whisperx.load_audio.side_effect = Exception("Load failed")

    transcriber = Transcriber()
    with pytest.raises(TranscriptionError):
        transcriber._load_audio("test.wav")


def test_perform_base_transcription(initial_prompt):
    mock_model = Mock()
    mock_model.transcribe.return_value = {"segments": [{"text": "test"}]}

    transcriber = Transcriber()
    transcriber.model = mock_model

    test_audio = np.array([1.0, 2.0])
    result = transcriber._perform_base_transcription(test_audio, initial_prompt)
    assert "segments" in result

    # Get the actual call arguments
    call_args = mock_model.transcribe.call_args
    assert call_args is not None
    args, kwargs = call_args

    # Verify the arguments
    assert np.array_equal(args[0], test_audio)
    assert kwargs.get("batch_size") == DEFAULT_BATCH_SIZE


def test_perform_base_transcription_error(initial_prompt):
    mock_model = Mock()
    mock_model.transcribe.side_effect = Exception("Transcription failed")

    transcriber = Transcriber()
    transcriber.model = mock_model

    with pytest.raises(TranscriptionError):
        transcriber._perform_base_transcription(np.array([1.0, 2.0]), initial_prompt)


def test_align_transcription(mock_dependencies):
    _, mock_whisperx, _ = mock_dependencies.return_value
    mock_whisperx.load_align_model.return_value = (Mock(), Mock())
    mock_whisperx.align.return_value = {"segments": [{"text": "aligned"}]}

    transcriber = Transcriber()
    test_audio = Mock()
    result = transcriber._align_transcription([{"text": "test"}], test_audio, "en")

    assert "segments" in result
    mock_whisperx.load_align_model.assert_called_once()

    # Verify the call arguments
    call_args = mock_whisperx.align.call_args
    assert call_args is not None
    _, kwargs = call_args

    # Verify the arguments we care about
    assert kwargs.get("return_char_alignments") is True


def test_align_transcription_load_model_error(mock_dependencies):
    _, mock_whisperx, _ = mock_dependencies.return_value
    mock_whisperx.load_align_model.side_effect = Exception("Model load failed")

    transcriber = Transcriber()
    with pytest.raises(Exception, match="Model load failed"):
        transcriber._align_transcription([{"text": "test"}], np.array([1.0, 2.0]), "en")


def test_align_transcription_align_error(mock_dependencies):
    _, mock_whisperx, _ = mock_dependencies.return_value
    mock_whisperx.load_align_model.return_value = (Mock(), Mock())
    mock_whisperx.align.side_effect = Exception("Alignment failed")

    transcriber = Transcriber()
    with pytest.raises(TranscriptionError):
        transcriber._align_transcription([{"text": "test"}], np.array([1.0, 2.0]), "en")


def test_perform_diarization():
    mock_diarization_model = Mock()
    mock_diarization_model.return_value = {"segments": [{"speaker": "SPEAKER_1"}]}

    transcriber = Transcriber()
    transcriber.diarization_model = mock_diarization_model

    result = transcriber._perform_diarization(np.array([1.0, 2.0]), 2)
    assert result and "segments" in result
    mock_diarization_model.assert_called_once()


def test_perform_diarization_error():
    mock_diarization_model = Mock()
    mock_diarization_model.side_effect = Exception("Diarization failed")

    transcriber = Transcriber()
    transcriber.diarization_model = mock_diarization_model

    with pytest.raises(TranscriptionError):
        transcriber._perform_diarization(np.array([1.0, 2.0]), 2)


def test_assign_speakers(mock_dependencies):
    _, mock_whisperx, _ = mock_dependencies.return_value
    mock_whisperx.assign_word_speakers.return_value = {
        "segments": [{"speaker": "SPEAKER_1"}]
    }

    transcriber = Transcriber()
    result = transcriber._assign_speakers(
        {"segments": [{"speaker": "SPEAKER_1"}]},
        {"segments": [{"text": "test"}], "language": "en"},
    )

    assert "segments" in result
    assert "language" in result
    mock_whisperx.assign_word_speakers.assert_called_once()


def test_assign_speakers_error(mock_dependencies):
    _, mock_whisperx, _ = mock_dependencies.return_value
    mock_whisperx.assign_word_speakers.side_effect = Exception(
        "Speaker assignment failed"
    )

    transcriber = Transcriber()
    with pytest.raises(TranscriptionError):
        transcriber._assign_speakers(
            {"segments": [{"speaker": "SPEAKER_1"}]},
            {"segments": [{"text": "test"}], "language": "en"},
        )


def test_handle_diarization_with_model():
    transcriber = Transcriber(diarization_model_name="test-model")

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
    transcriber = Transcriber(diarization_model_name="test-model")

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


def test_extract_transcription_metadata():
    transcriber = Transcriber()
    final_result = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "words": [{"word": "hello"}, {"word": "world"}],
            },
        ],
        "word_segments": [
            {"word": "hello", "score": 0.8},
            {"word": "world", "score": 0.9},
        ],
    }
    transcriber._extract_transcription_metadata(final_result)

    assert final_result["total_words"] == 2
    assert final_result["total_duration"] == 1.0
    assert final_result["speaking_duration"] == 1.0
    assert final_result["average_word_confidence"] == 0.8500  # (0.8 + 0.9) / 2


def test_extract_transcription_metadata_empty_segments():
    transcriber = Transcriber()
    final_result = {"segments": [], "word_segments": []}
    transcriber._extract_transcription_metadata(final_result)

    assert final_result["total_words"] == 0
    assert final_result["total_duration"] == 0
    assert final_result["speaking_duration"] == 0
    assert (
        "average_word_confidence" not in final_result
    )  # Should not be present with empty word_segments


def test_validate_language_supported():
    transcriber = Transcriber()
    # Should not raise any exception
    transcriber._validate_language("en")
    transcriber._validate_language("es")


def test_validate_language_unsupported():
    transcriber = Transcriber()
    with pytest.raises(TranscriptionError) as exc:
        transcriber._validate_language("fr")
    assert "Language 'fr' is not supported" in str(exc.value)


def test_transcribe_unsupported_language(tmp_path, mock_dependencies, initial_prompt):
    test_file = tmp_path / "test.wav"
    test_file.touch()

    _, mock_whisperx, _ = mock_dependencies.return_value
    mock_whisperx.load_audio.return_value = np.array([1.0, 2.0])

    mock_model = Mock()
    mock_model.transcribe.return_value = {
        "segments": [],
        "language": "fr",  # Unsupported language
    }

    transcriber = Transcriber()
    transcriber.model = mock_model

    with pytest.raises(TranscriptionError) as exc:
        transcriber.transcribe(test_file, initial_prompt)
    assert "Language 'fr' is not supported" in str(exc.value)


def test_extract_transcription_metadata_missing_fields():
    transcriber = Transcriber()
    final_result = {
        "segments": [
            {"start": 0.0, "end": 1.0},  # No words
            {"words": [{"word": "test"}]},  # No start/end
            {},  # Empty segment
        ],
        "word_segments": [
            {"word": "test", "score": 0.75},  # Normal segment
            {"word": "missing"},  # Missing score
            {},  # Empty segment
        ],
    }
    transcriber._extract_transcription_metadata(final_result)

    assert final_result["total_words"] == 1
    assert final_result["total_duration"] == 1.0  # Uses last valid end time
    assert final_result["speaking_duration"] == 1.0  # Only from first segment
    assert (
        final_result["average_word_confidence"] == 0.7500
    )  # Only uses segments with valid scores


def test_transcribe_integration_error_propagation(
    tmp_path, mock_transcription_result, mock_dependencies, initial_prompt
):
    test_file = tmp_path / "test.wav"
    test_file.touch()

    _, mock_whisperx, _ = mock_dependencies.return_value
    mock_whisperx.load_audio.return_value = np.array([1.0, 2.0])
    mock_whisperx.load_align_model.return_value = (Mock(), Mock())
    mock_whisperx.align.side_effect = Exception("Alignment failed")

    mock_model = Mock()
    mock_model.transcribe.return_value = mock_transcription_result

    transcriber = Transcriber(diarization_model_name="test_model")
    transcriber.model = mock_model

    with pytest.raises(TranscriptionError) as exc:
        transcriber.transcribe(test_file, initial_prompt)
    assert "Alignment failed" in str(exc.value)


def test_transcribe_integration(
    tmp_path,
    mock_transcription_result,
    mock_alignment_result,
    mock_diarization_result,
    mock_dependencies,
    initial_prompt,
):
    test_file = tmp_path / "test.wav"
    test_file.touch()

    # Setup mocks
    _, mock_whisperx, _ = mock_dependencies.return_value
    mock_whisperx.load_audio.return_value = Mock()
    mock_whisperx.load_align_model.return_value = (Mock(), Mock())

    # Configure mock returns with metadata
    mock_model = Mock()
    mock_model.transcribe.return_value = {
        **mock_transcription_result,
    }
    mock_whisperx.align.return_value = mock_alignment_result

    # Setup final result with metadata
    final_result = {
        **mock_alignment_result,
        "total_words": 2,
        "total_duration": 1.0,
        "speaking_duration": 1.0,
        "word_segments": [
            {"word": "Hello", "score": 0.9},
            {"word": "world", "score": 0.95},
        ],
    }
    mock_whisperx.assign_word_speakers.return_value = final_result

    # Create and configure transcriber
    transcriber = Transcriber(diarization_model_name="test_model")
    transcriber.model = mock_model
    transcriber.diarization_model = Mock(return_value=mock_diarization_result)

    # Perform transcription
    result = transcriber.transcribe(test_file, initial_prompt)

    # Verify pipeline flow
    mock_whisperx.load_audio.assert_called_once_with(test_file)
    mock_model.transcribe.assert_called_once()
    mock_whisperx.load_align_model.assert_called_once()
    mock_whisperx.align.assert_called_once()
    transcriber.diarization_model.assert_called_once()
    mock_whisperx.assign_word_speakers.assert_called_once()

    # Verify metadata in result
    assert "total_words" in result
    assert "total_duration" in result
    assert "speaking_duration" in result
    assert "average_word_confidence" in result
    assert result["average_word_confidence"] == 0.9250
