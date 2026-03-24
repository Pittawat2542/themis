# Use the Quickcheck CLI

`themis-quickcheck` reads stored SQLite projections without importing your
benchmark code.

## Scores

```bash
themis-quickcheck scores \
  --db .cache/themis-examples/01-hello-world-benchmark-first/themis.sqlite3 \
  --metric exact_match
```

Output:

```text
ev:edc937106dcf	demo-model	arithmetic	exact_match	1.0000	1
```

Filter by slice:

```bash
themis-quickcheck scores \
  --db .cache/themis-examples/04-compare-models-benchmark-first/themis.sqlite3 \
  --metric exact_match \
  --slice qa
```

Filter by benchmark dimension:

```bash
themis-quickcheck scores \
  --db .cache/themis-examples/01-hello-world-benchmark-first/themis.sqlite3 \
  --metric exact_match \
  --dimension source=synthetic
```

## Failures

```bash
themis-quickcheck failures \
  --db .cache/themis-examples/01-hello-world-benchmark-first/themis.sqlite3 \
  --limit 20
```

For a fully successful run, `failures` prints nothing.
