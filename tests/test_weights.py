"""fetch() puts each weight in the cache once and checked, also when several callers race on an empty cache."""

from __future__ import annotations

import hashlib
import threading
import time
import urllib.request

from edocr2.weights import Weight, fetch

PAYLOAD = b"0123456789abcdef" * 4096


def _slow_urlretrieve(_url, filename):
    """Writes PAYLOAD in pieces, slowly enough that concurrent downloads overlap."""
    with open(filename, "wb") as f:
        for start in range(0, len(PAYLOAD), 8192):
            f.write(PAYLOAD[start : start + 8192])
            f.flush()
            time.sleep(0.005)
    return filename, None


def test_concurrent_fetch_into_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EDOCR2_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("EDOCR2_WEIGHTS_DIR", raising=False)
    monkeypatch.setattr(urllib.request, "urlretrieve", _slow_urlretrieve)
    weight = Weight("w.bin", "https://example.invalid/w.bin", hashlib.sha256(PAYLOAD).hexdigest())
    results = {}

    def run(name):
        try:
            results[name] = fetch(weight)
        # Whatever a thread raises is what the assertion compares.
        except Exception as exc:  # noqa: BLE001
            results[name] = exc

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == {i: tmp_path / "w.bin" for i in range(4)}
    assert (tmp_path / "w.bin").read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("*.part"))
