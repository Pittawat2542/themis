import pytest

from themis.storage.artifact_store import ArtifactStore
from themis.errors.exceptions import StorageError


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
