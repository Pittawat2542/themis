# Postgres Storage

Postgres-backed connection management for the shared storage contract.

!!! warning "Implementation detail page"
    This page documents the backend manager used by the Postgres storage
    implementation. Treat it as implementation detail unless you are extending
    the storage backend itself; the stable extension surface remains
    `PostgresBlobStorageSpec` on the specs side.

::: themis.storage.postgres.manager
    options:
      show_root_heading: false
