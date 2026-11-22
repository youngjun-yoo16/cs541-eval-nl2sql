"""Read and update the JSONL file that stores learned verification checks."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable

from .models import CheckRecord


class CheckStore:
    """Manage check records in a local JSONL file."""

    def __init__(self, path: Path) -> None:
        """Remember the file path that will hold the check records."""
        self.path = path

    def load(self) -> list[CheckRecord]:
        """Read all saved checks and report which line is invalid, if any."""
        if not self.path.exists():
            return []

        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                try:
                    record_data = json.loads(line)
                    record = CheckRecord.from_dict(record_data)
                except Exception as exc:
                    message = f"Invalid check at {self.path}, line {line_number}: {exc}"
                    raise ValueError(message) from exc

                records.append(record)

        return records

    def upsert(self, record: CheckRecord) -> None:
        """Insert or update one check while preserving all of its source example IDs."""
        records_by_id = {existing.id: existing for existing in self.load()}
        previous_record = records_by_id.get(record.id)

        if previous_record:
            _merge_provenance(previous_record, record)

        records_by_id[record.id] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_records_atomically(self.path, records_by_id.values())


def _merge_provenance(previous: CheckRecord, updated: CheckRecord) -> None:
    """Carry forward source IDs and the strongest confidence when merging a check."""
    previous_ids = previous.provenance.get("source_example_ids", [])
    updated_ids = updated.provenance.get("source_example_ids", [])
    merged_ids = sorted(set(previous_ids) | set(updated_ids))

    updated.provenance["source_example_ids"] = merged_ids
    updated.confidence = max(previous.confidence, updated.confidence)


def _write_records_atomically(
    path: Path,
    records: Iterable[CheckRecord],
) -> None:
    """Write all checks to a temporary file, then replace the old memory file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        sorted_records = sorted(records, key=lambda item: item.id)
        for record in sorted_records:
            line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
            handle.write(line + "\n")

    temporary_path.replace(path)
