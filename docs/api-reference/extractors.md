# Extractors

Built-in extractor implementations shipped with Themis. Every fresh
`PluginRegistry()` auto-registers `regex`, `json_schema`, `first_number`, and
`choice_letter`.

`json_schema` depends on the optional `extractors` extra because it validates
candidate text with `jsonschema`.

::: themis.extractors.builtin
    options:
      show_root_heading: false
