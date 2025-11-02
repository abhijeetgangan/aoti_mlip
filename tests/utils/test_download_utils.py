from pathlib import Path

from aoti_mlip.utils.download_utils import download_file


class _DummyResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def test_download_file_writes_content(tmp_path, monkeypatch):
    expected = b"hello-world"

    def fake_get(url):  # noqa: ANN001
        return _DummyResp(expected)

    monkeypatch.setattr("aoti_mlip.utils.download_utils.requests.get", fake_get)

    out_path = Path(tmp_path) / "file.bin"
    download_file("http://example.invalid/file", str(out_path))
    assert out_path.exists()
    assert out_path.read_bytes() == expected
