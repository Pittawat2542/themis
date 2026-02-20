import unittest
from themis.core import entities


class TestCoreEntities(unittest.TestCase):
    def test_model_spec_key(self):
        """Test ModelSpec key generation."""
        spec = entities.ModelSpec("id", "provider")
        self.assertEqual(spec.model_key, "provider:id")
