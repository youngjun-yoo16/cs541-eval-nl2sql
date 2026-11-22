"""Data structures used to represent checks in the JSONL memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CheckRecord:
    """A reusable verification check and the metadata needed to apply it."""

    id: str
    version: int
    check_type: str
    rule_id: str
    dialect: str
    applicability: dict[str, Any]
    diagnostic: dict[str, Any]
    explanation: str
    suggested_correction: str
    confidence: float
    provenance: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckRecord":
        """Create a check from stored JSON and reject missing or invalid fields."""
        required_fields = {
            "id",
            "version",
            "check_type",
            "rule_id",
            "dialect",
            "applicability",
            "diagnostic",
            "explanation",
            "suggested_correction",
            "confidence",
            "provenance",
        }
        missing_fields = required_fields - data.keys()
        if missing_fields:
            raise ValueError(f"Missing fields: {sorted(missing_fields)}")

        if data["check_type"] != "static_rule":
            raise ValueError("Only allowlisted static-rule checks are supported.")

        confidence = float(data["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

        record_data = {field: data[field] for field in required_fields}
        return cls(**record_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert this check record into a JSON-serializable dictionary."""
        return asdict(self)
