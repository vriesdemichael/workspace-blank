import os
from sentence_embedder.model_downloader import download_model
from pytest_mock import MockerFixture
import pytest

def test_download_model(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        # Mock the responses
        m.setattr("model_downloader.os.path.exists", lambda x: False)
        mock_makedirs = mocker.patch("model_downloader.os.makedirs", return_value=None)
        mock_get = mocker.patch("model_downloader.requests.get")
        mock_get.return_value.json.return_value = ["file1.txt", "file2.txt"]
        mock_get.return_value.content = b"file content"

        url = "http://example.com/model"
        destination_folder = "models"

        download_model(url, destination_folder)

        # Check if the destination folder was created
        mock_makedirs.assert_called_once_with(destination_folder)

        # Check if the files were downloaded
        assert mock_get.call_count == 3
        mock_get.assert_any_call(url)
        mock_get.assert_any_call(f"{url}/file1.txt")
        mock_get.assert_any_call(f"{url}/file2.txt")

        # Check if the files were saved
        with open(os.path.join(destination_folder, "file1.txt"), "rb") as file:
            assert file.read() == b"file content"
        with open(os.path.join(destination_folder, "file2.txt"), "rb") as file:
            assert file.read() == b"file content"
