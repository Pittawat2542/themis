from __future__ import annotations

import sqlite3
import types
from typing import cast


class FakeCursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    def fetchone(self):
        row = self._cursor.fetchone()
        return dict(row) if row is not None else None

    def fetchall(self):
        return [dict(row) for row in self._cursor.fetchall()]


class FakeConnection:
    def __init__(self, path: str) -> None:
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row

    def execute(self, query: str, params=()):
        translated = query.replace("%s", "?")
        return FakeCursor(self._connection.execute(translated, params))

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._connection.__exit__(exc_type, exc, tb)

    def commit(self) -> None:
        self._connection.commit()


def fake_psycopg_module():
    module = types.SimpleNamespace()
    module.rows = types.SimpleNamespace(dict_row=object())
    module.connect = lambda url, row_factory=None: FakeConnection(url)
    return module


class FakeCollection:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.indexes: list[tuple[tuple[tuple[str, int], ...], bool]] = []

    def replace_one(
        self,
        query: dict[str, object],
        document: dict[str, object],
        upsert: bool = False,
    ) -> None:
        for index, row in enumerate(self.rows):
            if all(row.get(key) == value for key, value in query.items()):
                self.rows[index] = dict(document)
                return
        if upsert:
            self.rows.append(dict(document))

    def insert_one(self, document: dict[str, object]) -> None:
        self.rows.append(dict(document))

    def find(self, query: dict[str, object]) -> list[dict[str, object]]:
        return [
            row
            for row in self.rows
            if all(row.get(key) == value for key, value in query.items())
        ]

    def find_one(self, query: dict[str, object]) -> dict[str, object] | None:
        for row in self.rows:
            if all(row.get(key) == value for key, value in query.items()):
                return row
        return None

    def find_one_and_update(
        self,
        query: dict[str, object],
        update: dict[str, object],
        *,
        upsert: bool = False,
        return_document=None,
    ) -> dict[str, object] | None:
        del return_document
        row = self.find_one(query)
        if row is None:
            if not upsert:
                return None
            row = dict(query)
            self.rows.append(row)
        increments = update.get("$inc", {})
        if isinstance(increments, dict):
            for key, value in increments.items():
                current_value = row.get(key, 0)
                current_int = (
                    current_value
                    if isinstance(current_value, int)
                    else int(cast(str | bytes | bytearray, current_value))
                )
                row[key] = current_int + int(cast(int, value))
        return row

    def delete_many(self, query: dict[str, object]) -> None:
        self.rows = [
            row
            for row in self.rows
            if not all(row.get(key) == value for key, value in query.items())
        ]

    def create_index(self, keys, unique: bool = False) -> None:
        self.indexes.append((tuple(keys), unique))


class FakeDatabase:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def __getitem__(self, name: str) -> FakeCollection:
        if name not in self.collections:
            self.collections[name] = FakeCollection()
        return self.collections[name]


class FakeMongoClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.databases: dict[str, FakeDatabase] = {}

    def __getitem__(self, name: str) -> FakeDatabase:
        if name not in self.databases:
            self.databases[name] = FakeDatabase()
        return self.databases[name]


def fake_pymongo_module():
    module = types.SimpleNamespace()
    module.MongoClient = lambda url: FakeMongoClient(url)
    return module
