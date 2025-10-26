from themis import __version__, config, evaluation, experiment, generation


def test_version_string_format():
    assert isinstance(__version__, str)
    assert __version__
    assert "." in __version__


def test_package_exports_accessible():
    assert config is not None
    assert evaluation is not None
    assert experiment is not None
    assert generation is not None
