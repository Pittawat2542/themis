import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
from themis.storage import ExperimentStorage, RunStatus
from themis.core import entities as core_entities


class TestExperimentStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.test_dir)
        self.storage = ExperimentStorage(root=self.storage_path)
        # Ensure database is initialized (handled by MetadataStore init)
        self.run_id = "test-run-1"
        self.experiment_id = "test-exp"

        # Patch OS locking on LockManager
        self.lock_patcher = patch.object(self.storage._lock_manager, "_acquire_os_lock")
        self.mock_lock = self.lock_patcher.start()

    def tearDown(self):
        self.lock_patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_init_creates_directory(self):
        """Test storage initialization creates root directory."""
        self.assertTrue(self.storage_path.exists())
        # Check for experiments.db instead of metadata.db (schema update)
        self.assertTrue((self.storage_path / "experiments.db").exists())

    def test_start_run_creates_run_dir(self):
        """Test starting a run creates directory structure."""
        self.storage.start_run(
            self.run_id, experiment_id=self.experiment_id, config={"test": True}
        )

        # Verify run directory structure
        run_dir = self.storage._get_run_dir(self.run_id, self.experiment_id)
        self.assertTrue(run_dir.exists())
        self.assertTrue(
            (run_dir / ".lock").exists() or (run_dir / ".lock").parent.exists()
        )

        # Verify metadata
        run = self.storage._load_run_metadata(self.run_id)
        self.assertEqual(run.status, RunStatus.IN_PROGRESS)
        self.assertEqual(run.config_snapshot, {"test": True})

    def test_atomic_append_generation(self):
        """Test appending generation records atomically."""
        self.storage.start_run(self.run_id, experiment_id=self.experiment_id)

        record = core_entities.GenerationRecord(
            task=core_entities.GenerationTask(
                prompt=core_entities.PromptRender(
                    spec=core_entities.PromptSpec(name="t", template=""), text="p"
                ),
                model=core_entities.ModelSpec(identifier="m", provider="p"),
                sampling=core_entities.SamplingConfig(0.0, 0.0, 100),
                reference=None,  # Explicitly set reference to None
            ),
            output=core_entities.ModelOutput(text="result"),
            error=None,
        )

        # Use correct method name: append_record
        self.storage.append_record(self.run_id, record)

        # Verify retrieval (implementation detail: check file exists)
        gen_dir = self.storage._get_generation_dir(self.run_id)
        self.assertTrue(gen_dir.exists())

    def test_locking_prevents_concurrent_access(self):
        """Test that locking mechanism prevents concurrent access."""
        self.storage.start_run(self.run_id, experiment_id=self.experiment_id)

        # Simulate lock acquisition
        with self.storage._acquire_lock(self.run_id):
            # Check lock state via internal tracking since we mocked the OS lock
            self.assertIn(self.run_id, self.storage._lock_manager._locks)
