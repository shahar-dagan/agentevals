"""Abstract base class for trace loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """Normalized representation of a trace span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation_name: str
    start_time: int  # microseconds
    duration: int  # microseconds
    tags: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    def get_tag(self, key: str, default: Any = None) -> Any:
        return self.tags.get(key, default)

    @property
    def end_time(self) -> int:
        return self.start_time + self.duration


@dataclass
class Trace:
    trace_id: str
    root_spans: list[Span] = field(default_factory=list)
    all_spans: list[Span] = field(default_factory=list)

    def find_spans_by_operation(self, operation_prefix: str) -> list[Span]:
        return [s for s in self.all_spans if s.operation_name.startswith(operation_prefix)]

    def find_spans_by_tag(self, key: str, value: Any) -> list[Span]:
        return [s for s in self.all_spans if s.get_tag(key) == value]


class TraceLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> list[Trace]:
        """Load traces from a source (file path, URL, etc.)."""
        ...

    @abstractmethod
    def format_name(self) -> str:
        """Return the name of the trace format this loader handles."""
        ...
