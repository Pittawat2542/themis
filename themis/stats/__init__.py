from themis._optional import import_optional

__all__ = ["StatsEngine", "ComparisonResult"]


def __getattr__(name: str):
    if name in __all__:
        module = import_optional("themis.stats.stats_engine", extra="stats")
        return getattr(module, name)
    raise AttributeError(f"module 'themis.stats' has no attribute {name!r}")
