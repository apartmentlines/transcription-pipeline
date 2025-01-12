from unittest.mock import Mock, patch
import pytest
from transcription_pipeline.audio_file_validator import (
    AudioFileValidator,
    AudioFileLengthError,
    parse_arguments,
    main,
)
from transcription_pipeline.constants import MIN_AUDIO_DURATION, MAX_AUDIO_DURATION


@pytest.fixture
def mock_wave_module():
    mock_wave = Mock()
    mock_wave.Error = Exception
    return mock_wave


@pytest.fixture
def mock_wav_file():
    mock_file = Mock()
    mock_file.getnframes.return_value = 48000
    mock_file.getframerate.return_value = 16000
    mock_file.__enter__ = Mock(return_value=mock_file)
    mock_file.__exit__ = Mock(return_value=None)
    return mock_file


def test_init():
    validator = AudioFileValidator()
    assert validator.min_duration == 5
    assert validator.max_duration == 600


def test_validate_file_path_success(tmp_path):
    test_file = tmp_path / "test.wav"
    test_file.touch()
    validator = AudioFileValidator(test_file)
    assert validator._validate_file_path(test_file) == test_file


def test_validate_file_path_not_found():
    validator = AudioFileValidator()
    with pytest.raises(FileNotFoundError):
        validator.validate("nonexistent.wav")


def test_open_wave_file(tmp_path, mock_wave_module, mock_wav_file):
    test_file = tmp_path / "test.wav"
    test_file.touch()
    mock_wave_module.open.return_value = mock_wav_file
    validator = AudioFileValidator(wave_module=mock_wave_module)
    with validator._open_wave_file(test_file) as wav:
        assert wav == mock_wav_file

    mock_wave_module.open.assert_called_once_with(str(test_file), "r")


def test_open_wave_file_error(tmp_path, mock_wave_module):
    test_file = tmp_path / "test.wav"
    test_file.touch()
    mock_wave_module.open.side_effect = mock_wave_module.Error("Test error")

    validator = AudioFileValidator(wave_module=mock_wave_module)
    with pytest.raises(AudioFileLengthError):
        validator._open_wave_file(test_file)


def test_calculate_duration(tmp_path, mock_wav_file):
    test_file = tmp_path / "test.wav"
    test_file.touch()
    validator = AudioFileValidator(test_file)

    duration = validator._calculate_duration(mock_wav_file)
    assert duration == 3.0  # 48000 frames / 16000 Hz = 3 seconds


def test_get_duration(tmp_path, mock_wave_module, mock_wav_file):
    test_file = tmp_path / "test.wav"
    test_file.touch()
    mock_wave_module.open.return_value = mock_wav_file
    validator = AudioFileValidator(wave_module=mock_wave_module)
    duration = validator.get_duration(test_file)
    assert duration == 3.0


def test_validate_success(tmp_path, mock_wave_module, mock_wav_file):
    test_file = tmp_path / "test.wav"
    test_file.touch()
    mock_wave_module.open.return_value = mock_wav_file

    validator = AudioFileValidator(
        min_duration=2.0, max_duration=4.0, wave_module=mock_wave_module
    )
    validator.validate(test_file)  # Should not raise any exceptions


def test_validate_too_short(tmp_path, mock_wave_module, mock_wav_file):
    test_file = tmp_path / "test.wav"
    test_file.touch()
    mock_wave_module.open.return_value = mock_wav_file

    validator = AudioFileValidator(
        min_duration=4.0, max_duration=5.0, wave_module=mock_wave_module
    )
    with pytest.raises(AudioFileLengthError, match="too short"):
        validator.validate(test_file)


def test_validate_too_long(tmp_path, mock_wave_module, mock_wav_file):
    test_file = tmp_path / "test.wav"
    test_file.touch()
    mock_wave_module.open.return_value = mock_wav_file

    validator = AudioFileValidator(
        min_duration=1.0, max_duration=2.0, wave_module=mock_wave_module
    )
    with pytest.raises(AudioFileLengthError, match="too long"):
        validator.validate(test_file)


def test_parse_arguments():
    test_args = ["--min-duration", "10", "--max-duration", "200", "--debug", "test.wav"]
    with patch("sys.argv", ["script.py"] + test_args):
        args = parse_arguments()
        assert args.min_duration == 10.0
        assert args.max_duration == 200.0
        assert args.debug is True
        assert args.file == "test.wav"


def test_parse_arguments_defaults():
    with patch("sys.argv", ["script.py", "test.wav"]):
        args = parse_arguments()
        assert args.min_duration == MIN_AUDIO_DURATION
        assert args.max_duration == MAX_AUDIO_DURATION
        assert args.debug is False
        assert args.file == "test.wav"


def test_main_success(tmp_path, capsys, mock_wave_module, mock_wav_file):
    """Test successful validation using mocked wave module."""
    test_file = tmp_path / "test.wav"
    test_file.touch()
    mock_wave_module.open.return_value = mock_wav_file
    with patch("sys.argv", ["script.py", str(test_file)]):
        with patch(
            "transcription_pipeline.audio_file_validator.AudioFileValidator"
        ) as mock_validator_class:
            mock_validator_instance = Mock()
            mock_validator_class.return_value = mock_validator_instance
            main()
            mock_validator_class.assert_called_once_with(
                min_duration=MIN_AUDIO_DURATION,
                max_duration=MAX_AUDIO_DURATION,
                debug=False,
            )
            mock_validator_instance.validate.assert_called_once_with(str(test_file))
            captured = capsys.readouterr()
            assert captured.err == ""


def test_main_failure(tmp_path, capsys):
    with patch("sys.argv", ["script.py", "nonexistent.wav"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.out
