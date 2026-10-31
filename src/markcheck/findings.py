"""The shared shape every detector returns.

Each detector looks at one file and returns a list of Finding objects.
A Finding is either a label we found, a check that found nothing,
a check we skipped (missing library, wrong file type), or an error.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Status = Literal["found", "not_found", "skipped", "error"]


@dataclass
class Finding:
    detector: str            # machine name: "c2pa", "metadata", "sd_watermark", "trustmark"
    label: str               # plain name shown to people, e.g. "Signed label (C2PA)"
    status: Status
    summary: str             # one plain-English sentence
    details: dict = field(default_factory=dict)       # facts for --json and the detail lines
    proves: list[str] = field(default_factory=list)   # what this result does show
    does_not_prove: list[str] = field(default_factory=list)  # what it does not show

    @property
    def found(self) -> bool:
        return self.status == "found"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["found"] = self.found
        return d


@dataclass
class FileReport:
    path: str
    kind: str                # "image", "video", "audio", "pdf", "text", "other"
    findings: list[Finding] = field(default_factory=list)
    makers: list[dict] = field(default_factory=list)  # "only the maker can check this" entries
    notes: list[str] = field(default_factory=list)    # plain-English notes about how the file was read

    @property
    def any_found(self) -> bool:
        return any(f.found for f in self.findings)

    @property
    def any_error(self) -> bool:
        return any(f.status == "error" for f in self.findings)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "kind": self.kind,
            "label_found": self.any_found,
            "findings": [f.to_dict() for f in self.findings],
            "only_the_maker_can_check": self.makers,
            "notes": self.notes,
        }
