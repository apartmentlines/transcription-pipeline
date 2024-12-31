import requests
from unittest.mock import patch, Mock
from transcription_pipeline.main import (
    parse_arguments,
    retrieve_file_data,
    prepare_file_data,
    main,
)

@patch('transcription_pipeline.main.configure_logging')
@patch('transcription_pipeline.main.load_configuration')
@patch('transcription_pipeline.main.set_environment_variables')
@patch('transcription_pipeline.main.retrieve_file_data')
@patch('transcription_pipeline.main.prepare_file_data')
@patch('transcription_pipeline.main.ProcessingPipeline')
def test_main_success(mock_pipeline, mock_prepare, mock_retrieve,
                     mock_set_env, mock_load_config, mock_configure_logging):
    mock_load_config.return_value = ('test_key', 'test_domain')
    mock_retrieve.return_value = [{'url': 'test_url'}]
    mock_prepare.return_value = [{'url': 'test_url_with_key'}]
    mock_pipeline_instance = Mock()
    mock_pipeline.return_value = mock_pipeline_instance

    with patch('sys.argv', ['main.py', '--debug']):
        main()

    mock_configure_logging.assert_called_once_with(True)
    mock_load_config.assert_called_once()
    mock_set_env.assert_called_once_with('test_key', 'test_domain')
    mock_retrieve.assert_called_once_with('test_domain')
    mock_prepare.assert_called_once()
    mock_pipeline_instance.run.assert_called_once()

@patch('transcription_pipeline.main.load_configuration')
def test_main_configuration_error(mock_load_config):
    mock_load_config.side_effect = ValueError("Test error")
    with patch('sys.argv', ['main.py']):
        with patch('transcription_pipeline.main.fail_hard') as mock_fail:
            main()
            mock_fail.assert_called_once_with("Test error")

def test_parse_arguments():
    test_args = ['--api-key', 'test_api_key', '--domain', 'test_domain', '--debug']
    with patch('sys.argv', ['main.py'] + test_args):
        args = parse_arguments()
        assert args.api_key == 'test_api_key'
        assert args.domain == 'test_domain'
        assert args.debug is True

@patch('transcription_pipeline.main.get_request')
def test_retrieve_file_data_success(mock_get_request):
    mock_response = Mock()
    mock_response.json.return_value = {'success': True, 'files': [{'id': '1', 'url': 'url1'}]}
    mock_get_request.return_value = mock_response
    files = retrieve_file_data('test_domain')
    assert files == [{'id': '1', 'url': 'url1'}]

@patch('transcription_pipeline.main.get_request')
def test_retrieve_file_data_failure(mock_get_request):
    mock_response = Mock()
    mock_response.json.return_value = {'success': False}
    mock_get_request.return_value = mock_response
    with patch('transcription_pipeline.main.fail_hard') as mock_fail_hard:
        retrieve_file_data('test_domain')
        mock_fail_hard.assert_called_once()

@patch('transcription_pipeline.main.get_request')
def test_retrieve_file_data_request_exception(mock_get_request):
    mock_get_request.side_effect = requests.exceptions.RequestException("Connection error")
    with patch('transcription_pipeline.main.fail_hard') as mock_fail:
        retrieve_file_data('test_domain')
        mock_fail.assert_called_once_with("Error retrieving files: Connection error")

def test_prepare_file_data():
    files = [{'url': 'http://example.com/file1'}, {'url': 'http://example.com/file2'}]
    updated_files = prepare_file_data(files, 'test_api_key', limit=1)
    assert len(updated_files) == 1
    assert updated_files[0]['url'] == 'http://example.com/file1?api_key=test_api_key'

def test_prepare_file_data_with_existing_params():
    files = [{'url': 'http://example.com/file1?param=value'}]
    result = prepare_file_data(files, 'test_api_key', None)
    assert result[0]['url'] == 'http://example.com/file1?param=value&api_key=test_api_key'

def test_prepare_file_data_empty_list():
    files = []
    result = prepare_file_data(files, 'test_api_key', None)
    assert result == []
