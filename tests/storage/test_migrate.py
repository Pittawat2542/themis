from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest

from themis.core.stores.postgres import postgres_store

pytestmark = pytest.mark.skipif(
    not os.getenv("THEMIS_TEST_POSTGRES_ADMIN_URL"),
    reason="THEMIS_TEST_POSTGRES_ADMIN_URL is required for Postgres integration tests",
)


def _psycopg() -> Any:
    import psycopg  # type: ignore[import-not-found]

    return psycopg


@pytest.fixture
def postgres_database(tmp_path: Path):
    psycopg = _psycopg()
    admin_url = os.environ["THEMIS_TEST_POSTGRES_ADMIN_URL"]
    database_name = f"themis_test_{uuid.uuid4().hex}"
    with psycopg.connect(admin_url, autocommit=True) as connection:
        connection.execute(f'CREATE DATABASE "{database_name}"')

    parsed = urlsplit(admin_url)
    database_url = urlunsplit(parsed._replace(path=f"/{database_name}"))

    try:
        yield database_url, tmp_path / "postgres-blobs"
    finally:
        with psycopg.connect(admin_url, autocommit=True) as connection:
            connection.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                (database_name,),
            )
            connection.execute(f'DROP DATABASE IF EXISTS "{database_name}"')


def test_postgres_initialize_creates_schema_version_and_is_idempotent(
    postgres_database,
) -> None:
    database_url, blob_root = postgres_database
    store = postgres_store(database_url, blob_root)

    store.initialize()
    store.initialize()

    psycopg = _psycopg()
    with psycopg.connect(database_url) as connection:
        row = connection.execute(
            """
            SELECT value
            FROM run_store_meta
            WHERE key = %s
            """,
            ("schema_version",),
        ).fetchone()

    assert row is not None
    assert row[0] == "1"


def test_postgres_initialize_migrates_from_version_zero(postgres_database) -> None:
    database_url, blob_root = postgres_database
    psycopg = _psycopg()
    with psycopg.connect(database_url) as connection:
        connection.execute(
            """
            CREATE TABLE run_store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
        )
        connection.execute(
            """
            INSERT INTO run_store_meta (key, value)
            VALUES (%s, %s)
            """,
            ("schema_version", "0"),
        )
        connection.commit()

    store = postgres_store(database_url, blob_root)
    store.initialize()

    with psycopg.connect(database_url) as connection:
        row = connection.execute(
            """
            SELECT value
            FROM run_store_meta
            WHERE key = %s
            """,
            ("schema_version",),
        ).fetchone()
        events_table = connection.execute(
            """
            SELECT to_regclass('public.run_events')
            """
        ).fetchone()

    assert row is not None
    assert row[0] == "1"
    assert events_table is not None
    assert events_table[0] == "run_events"
