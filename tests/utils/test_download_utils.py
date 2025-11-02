from importlib import import_module
from pathlib import Path


class _DummyResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def test_download_file_writes_content(tmp_path, monkeypatch):
    expected = b"hello-world"

    def fake_get(url):  # noqa: ANN001
        return _DummyResp(expected)

    monkeypatch.setattr("requests.get", fake_get)
    dl_mod = import_module("aoti_mlip.utils.download_utils")

    out_path = Path(tmp_path) / "file.bin"
    dl_mod.download_file("http://example.invalid/file", str(out_path))  # type: ignore[attr-defined]
    assert out_path.exists()
    assert out_path.read_bytes() == expected
