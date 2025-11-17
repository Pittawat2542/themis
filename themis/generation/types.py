"""Backwards-compatible aliases for core entities."""

from themis.core import entities as core_entities

SamplingParameters = core_entities.SamplingConfig
ModelOutput = core_entities.ModelOutput
GenerationError = core_entities.ModelError
GenerationRequest = core_entities.GenerationTask
GenerationResult = core_entities.GenerationRecord
