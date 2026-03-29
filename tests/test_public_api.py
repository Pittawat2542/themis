def test_root_package_exports_phase1_symbols() -> None:
    from themis import Experiment, RunSnapshot, sqlite_store

    assert Experiment is not None
    assert RunSnapshot is not None
    assert sqlite_store is not None
