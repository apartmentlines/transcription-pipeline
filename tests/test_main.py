import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from download_pipeline_processor.processing_pipeline import ProcessingPipeline
from transcription_pipeline.processors.transcription_processor import (
    TranscriptionProcessor,
)
from transcription_pipeline.processors.transcription_post_processor import (
    TranscriptionPostProcessor,
)
from transcription_pipeline.main import (
    parse_arguments,
    main,
    TranscriptionPipeline,
)
from transcription_pipeline.constants import (
    DEFAULT_PROCESSING_LIMIT,
    DEFAULT_DOWNLOAD_QUEUE_SIZE,
    DEFAULT_DOWNLOAD_CACHE,
)


class TestTranscriptionPipeline:
    @pytest.fixture
    def pipeline(self, mock_transcriber):
        return TranscriptionPipeline(api_key="test_key", domain="test_domain")

    def test_initialization(self, mock_transcriber):
        # Test with all arguments specified
        pipeline = TranscriptionPipeline(
            api_key="test_key",
            domain="test_domain",
            limit=5,
            processing_limit=10,
            download_queue_size=20,
            download_cache=Path("/tmp/test"),
            simulate_downloads=True,
            debug=True,
        )
        assert pipeline.api_key == "test_key"
        assert pipeline.domain == "test_domain"
        assert pipeline.limit == 5
        assert pipeline.processing_limit == 10
        assert pipeline.download_queue_size == 20
        assert pipeline.download_cache == Path("/tmp/test")
        assert pipeline.simulate_downloads is True
        assert pipeline.debug is True

        # Test with defaults
        pipeline = TranscriptionPipeline(api_key="test_key", domain="test_domain")
        assert pipeline.api_key == "test_key"
        assert pipeline.domain == "test_domain"
        assert pipeline.limit is None
        assert pipeline.processing_limit == DEFAULT_PROCESSING_LIMIT
        assert pipeline.download_queue_size == DEFAULT_DOWNLOAD_QUEUE_SIZE
        assert pipeline.download_cache == DEFAULT_DOWNLOAD_CACHE
        assert pipeline.simulate_downloads is False
        assert pipeline.debug is False

    @patch("transcription_pipeline.main.get_request")
    def test_retrieve_file_data_success(self, mock_get_request, pipeline):
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "files": [{"id": "1", "url": "url1"}],
        }
        mock_get_request.return_value = mock_response

        files = pipeline.retrieve_file_data()
        assert files == [{"id": "1", "url": "url1"}]

    @patch("transcription_pipeline.main.get_request")
    def test_retrieve_file_data_failure(self, mock_get_request, pipeline):
        mock_response = Mock()
        mock_response.json.return_value = {"success": False}
        mock_get_request.return_value = mock_response

        with pytest.raises(SystemExit):
            pipeline.retrieve_file_data()

    def test_prepare_file_data(self, pipeline):
        files = [
            {"url": "http://example.com/file1"},
            {"url": "http://example.com/file2"},
        ]
        updated_files = pipeline.prepare_file_data(files)
        assert len(updated_files) == 2
        assert updated_files[0]["url"] == "http://example.com/file1?api_key=test_key"
        assert updated_files[1]["url"] == "http://example.com/file2?api_key=test_key"

    def test_prepare_file_data_with_limit(self, pipeline):
        pipeline.limit = 1
        files = [
            {"url": "http://example.com/file1"},
            {"url": "http://example.com/file2"},
        ]
        updated_files = pipeline.prepare_file_data(files)
        assert len(updated_files) == 1

    def test_setup_configuration(self, pipeline):
        with patch(
            "transcription_pipeline.main.set_environment_variables"
        ) as mock_set_env:
            pipeline.setup_configuration()
            mock_set_env.assert_called_once_with("test_key", "test_domain")

    def test_pipeline_initialization(self, mock_transcriber):
        # Test that ProcessingPipeline is initialized with correct values
        pipeline = TranscriptionPipeline(
            api_key="test_key",
            domain="test_domain",
            processing_limit=5,
            download_queue_size=15,
            download_cache=Path("/tmp/test"),
            simulate_downloads=True,
            debug=True,
        )

        assert pipeline.pipeline is not None
        assert isinstance(pipeline.pipeline, ProcessingPipeline)
        assert pipeline.pipeline.processing_limit == 5
        assert pipeline.pipeline.download_queue_size == 15
        assert pipeline.pipeline.download_cache == Path("/tmp/test")
        assert pipeline.pipeline.simulate_downloads is True
        assert pipeline.pipeline.debug is True
        assert pipeline.pipeline.processor_class == TranscriptionProcessor
        assert pipeline.pipeline.post_processor_class == TranscriptionPostProcessor

    def test_pipeline_initialization_defaults(self, mock_transcriber):
        # Test that ProcessingPipeline is initialized with correct default values
        pipeline = TranscriptionPipeline(api_key="test_key", domain="test_domain")

        assert pipeline.pipeline is not None
        assert isinstance(pipeline.pipeline, ProcessingPipeline)
        assert pipeline.pipeline.processing_limit == DEFAULT_PROCESSING_LIMIT
        assert pipeline.pipeline.download_queue_size == DEFAULT_DOWNLOAD_QUEUE_SIZE
        assert pipeline.pipeline.download_cache == DEFAULT_DOWNLOAD_CACHE
        assert pipeline.pipeline.simulate_downloads is False
        assert pipeline.pipeline.debug is False
        assert pipeline.pipeline.processor_class == TranscriptionProcessor
        assert pipeline.pipeline.post_processor_class == TranscriptionPostProcessor

    def test_build_retrieve_request_url(self, pipeline):
        """Test that the retrieve request URL is correctly constructed."""
        expected_url = (
            "https://test_domain/al/transcriptions/retrieve/operator-recordings/ready"
        )
        assert pipeline.build_retrieve_request_url() == expected_url

    def test_build_retrieve_request_params(self, pipeline):
        """Test that the retrieve request parameters are correctly constructed."""
        expected_params = {"api_key": "test_key"}
        assert pipeline.build_retrieve_request_params() == expected_params

    def test_build_retrieve_request_params_with_filters(self):
        """Test request params with min_id, max_id and from_s3 filters."""
        pipeline = TranscriptionPipeline(
            api_key="test_key",
            domain="test_domain",
            min_id=100,
            max_id=200,
            from_s3=True
        )
        params = pipeline.build_retrieve_request_params()
        assert params == {
            "api_key": "test_key",
            "min_id": 100,
            "max_id": 200,
            "from_s3": "1"
        }

    def test_build_retrieve_request_params_partial_filters(self):
        """Test request params with only some filters set."""
        pipeline = TranscriptionPipeline(
            api_key="test_key",
            domain="test_domain",
            min_id=100,
            from_s3=True
        )
        params = pipeline.build_retrieve_request_params()
        assert params == {
            "api_key": "test_key",
            "min_id": 100,
            "from_s3": "1"
        }

    @patch("transcription_pipeline.main.get_request")
    def test_retrieve_file_data_with_filters(self, mock_get_request):
        """Test that retrieve_file_data passes filters correctly."""
        pipeline = TranscriptionPipeline(
            api_key="test_key",
            domain="test_domain",
            min_id=100,
            max_id=200,
            from_s3=True
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "files": [{"id": "1", "url": "url1"}],
        }
        mock_get_request.return_value = mock_response

        pipeline.retrieve_file_data()
        
        mock_get_request.assert_called_once_with(
            pipeline.build_retrieve_request_url(),
            {
                "api_key": "test_key",
                "min_id": 100,
                "max_id": 200,
                "from_s3": "1"
            }
        )

    @patch("download_pipeline_processor.processing_pipeline.ProcessingPipeline.run")
    @patch.object(TranscriptionPipeline, "prepare_file_data")
    @patch.object(TranscriptionPipeline, "retrieve_file_data")
    @patch.object(TranscriptionPipeline, "setup_configuration")
    def test_run_successful_execution(
        self, mock_setup, mock_retrieve, mock_prepare, mock_run, pipeline
    ):
        mock_files = [
            {"id": "1", "url": "http://example.com/file1"},
            {"id": "2", "url": "http://example.com/file2"},
        ]
        mock_retrieve.return_value = mock_files
        mock_prepare.return_value = mock_files

        pipeline.run()

        mock_setup.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_prepare.assert_called_once_with(mock_files)
        mock_run.assert_called_once_with(mock_files)

    @patch("download_pipeline_processor.processing_pipeline.ProcessingPipeline.run")
    @patch.object(TranscriptionPipeline, "prepare_file_data")
    @patch.object(TranscriptionPipeline, "retrieve_file_data")
    @patch.object(TranscriptionPipeline, "setup_configuration")
    def test_run_no_files(
        self, mock_setup, mock_retrieve, mock_prepare, mock_run, pipeline
    ):
        mock_retrieve.return_value = []

        pipeline.run()

        mock_setup.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_prepare.assert_not_called()
        mock_run.assert_not_called()

    @patch("download_pipeline_processor.processing_pipeline.ProcessingPipeline.run")
    @patch.object(TranscriptionPipeline, "setup_configuration")
    def test_run_configuration_error(self, mock_setup, mock_run, pipeline):
        mock_setup.side_effect = ValueError("Configuration error")

        with pytest.raises(ValueError, match="Configuration error"):
            pipeline.run()
        mock_run.assert_not_called()

    @patch("download_pipeline_processor.processing_pipeline.ProcessingPipeline.run")
    @patch.object(TranscriptionPipeline, "retrieve_file_data")
    @patch.object(TranscriptionPipeline, "setup_configuration")
    def test_run_retrieve_error(self, mock_setup, mock_retrieve, mock_run, pipeline):
        mock_retrieve.side_effect = Exception("Retrieval error")

        with pytest.raises(Exception, match="Retrieval error"):
            pipeline.run()
        mock_run.assert_not_called()

    @patch("transcription_pipeline.main.TranscriptionPipeline")
    def test_main_success(self, mock_pipeline, mock_transcriber):
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        with patch(
            "sys.argv",
            ["main.py", "--debug", "--api-key", "test_key", "--domain", "test_domain"],
        ):
            main()

        mock_pipeline.assert_called_once_with(
            api_key="test_key",
            domain="test_domain",
            debug=True,
            limit=None,
            min_id=None,
            max_id=None,
            from_s3=False,
            processing_limit=DEFAULT_PROCESSING_LIMIT,
            download_queue_size=DEFAULT_DOWNLOAD_QUEUE_SIZE,
            download_cache=DEFAULT_DOWNLOAD_CACHE,
            simulate_downloads=False,
        )
        mock_pipeline_instance.run.assert_called_once()

    def test_cli_arguments_with_filters(self):
        """Test that CLI arguments for filters are properly parsed."""
        test_args = [
            "--api-key", "test_api_key",
            "--domain", "test_domain",
            "--min-id", "100",
            "--max-id", "200",
            "--from-s3"
        ]
        with patch("sys.argv", ["main.py"] + test_args):
            args = parse_arguments()
            assert args.min_id == 100
            assert args.max_id == 200
            assert args.from_s3 is True

    @patch("transcription_pipeline.main.TranscriptionPipeline")
    def test_main_with_filters(self, mock_pipeline, mock_transcriber):
        """Test that main() properly passes filter arguments to TranscriptionPipeline."""
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        with patch("sys.argv", [
            "main.py",
            "--api-key", "test_key",
            "--domain", "test_domain",
            "--min-id", "100",
            "--max-id", "200",
            "--from-s3"
        ]):
            main()

        mock_pipeline.assert_called_once_with(
            api_key="test_key",
            domain="test_domain",
            min_id=100,
            max_id=200,
            from_s3=True,
            debug=False,
            limit=None,
            processing_limit=DEFAULT_PROCESSING_LIMIT,
            download_queue_size=DEFAULT_DOWNLOAD_QUEUE_SIZE,
            download_cache=DEFAULT_DOWNLOAD_CACHE,
            simulate_downloads=False,
        )
        mock_pipeline_instance.run.assert_called_once()

@patch("transcription_pipeline.main.load_configuration")
def test_main_configuration_error(mock_load_config):
    mock_load_config.side_effect = ValueError("Test error")
    with patch("sys.argv", ["main.py"]):
        with patch("transcription_pipeline.main.fail_hard") as mock_fail:
            main()
            mock_fail.assert_called_once_with("Test error")


def test_parse_arguments():
    test_args = ["--api-key", "test_api_key", "--domain", "test_domain", "--debug"]
    with patch("sys.argv", ["main.py"] + test_args):
        args = parse_arguments()
        assert args.api_key == "test_api_key"
        assert args.domain == "test_domain"
        assert args.debug is True
