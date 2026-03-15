import pytest
import threading

from themis.storage.artifact_store import ArtifactStore
from themis.errors import StorageError


def test_artifact_store_put_and_get_blob(tmp_path):
    store = ArtifactStore(base_path=tmp_path)

    payload = b'{"complex":["data",123]}'
    blob_ref = store.put_blob(payload, "application/json")

    assert blob_ref.startswith("sha256:")
    assert store.exists(blob_ref)

    hex_hash = blob_ref.split(":")[1]
    expected_path = tmp_path / hex_hash[:2] / hex_hash[2:4] / f"{hex_hash}.zst"
    assert expected_path.exists()

    assert store.get_blob(blob_ref) == payload


def test_artifact_store_read_not_found(tmp_path):
    store = ArtifactStore(base_path=tmp_path)
    with pytest.raises(StorageError, match="Artifact sha256:fakehash not found"):
        store.get_blob("sha256:fakehash")


def test_artifact_store_write_json_wraps_non_json_safe_payloads(tmp_path):
    store = ArtifactStore(base_path=tmp_path)

    with pytest.raises(StorageError, match="application/json"):
        store.write_json({"bad": object()})


def test_artifact_store_uses_thread_local_codecs(tmp_path):
    store = ArtifactStore(base_path=tmp_path)
    thread_count = 4
    start = threading.Barrier(thread_count + 1)
    finish = threading.Barrier(thread_count + 1)
    compressors: list[object] = []
    decompressors: list[object] = []

    def capture_codec_ids() -> None:
        start.wait()
        compressors.append(store._compressor)
        decompressors.append(store._decompressor)
        finish.wait()

    threads = [threading.Thread(target=capture_codec_ids) for _ in range(thread_count)]
    for thread in threads:
        thread.start()

    start.wait()
    finish.wait()
    for thread in threads:
        thread.join()

    assert len({id(codec) for codec in compressors}) == thread_count
    assert len({id(codec) for codec in decompressors}) == thread_count
