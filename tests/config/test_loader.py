from themis.config import loader


def test_load_experiment_config_supports_overrides(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        generation:
          sampling:
            temperature: 0.3
        max_samples: 5
        """
    )

    config = loader.load_experiment_config(
        config_path, overrides=["generation.model_identifier=my-model"]
    )

    assert config.generation.sampling.temperature == 0.3
    assert config.generation.model_identifier == "my-model"
    assert config.max_samples == 5
