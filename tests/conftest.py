import pytest
from download_pipeline_processor.file_data import FileData


@pytest.fixture
def file_data() -> FileData:
    file_data = FileData(id="123", name="test_file", url="http://example.com/file")
    file_data.record_name = "Test Apartments"
    return file_data
