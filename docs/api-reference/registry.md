# Registry

Plugin registrations, execution capability metadata, and planning-time
validation.

`themis.registry` is the stable namespace for runtime plugin registration and
capability checks. Import from it when you need `PluginRegistry`,
`CompatibilityChecker`, or `EngineCapabilities` without drilling into the module
layout.

::: themis.registry
    options:
      show_root_heading: false

::: themis.registry.plugin_registry
    options:
      show_root_heading: false

::: themis.registry.compatibility
    options:
      show_root_heading: false
