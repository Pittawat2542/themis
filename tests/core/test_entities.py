import unittest
from dataclasses import FrozenInstanceError
from themis.core import entities


class TestCoreEntities(unittest.TestCase):
    def test_sampling_config_immutability(self):
        """Test SamplingConfig is frozen."""
        config = entities.SamplingConfig(0.0, 1.0, 100)
        with self.assertRaises(FrozenInstanceError):
            config.temperature = 0.5

    def test_model_spec_key(self):
        """Test ModelSpec key generation."""
        spec = entities.ModelSpec("id", "provider")
        self.assertEqual(spec.model_key, "provider:id")
