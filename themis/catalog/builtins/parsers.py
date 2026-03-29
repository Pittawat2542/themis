"""Builtin parsers."""

from __future__ import annotations

from themis.core.contexts import ParseContext
from themis.core.models import ParsedOutput, ReducedCandidate


class JsonIdentityParser:
    component_id = "builtin/json_identity"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-json-identity-fingerprint"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        return ParsedOutput(value=candidate.final_output, format="json")
