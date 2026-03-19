"""Resolve remote grader references to local cached files."""

from __future__ import annotations

import logging
from pathlib import Path

from .sources import GraderSource, get_sources

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "agentevals" / "graders"


class GraderResolver:
    """Downloads and caches remote graders, converting them to local paths."""

    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._sources: dict[str, GraderSource] = {}

    def register_source(self, source: GraderSource) -> None:
        self._sources[source.source_name] = source

    async def resolve(self, grader_def) -> CodeGraderDef:
        """Download a remote grader and return a CodeGraderDef pointing to the cached file."""
        from ..config import CodeGraderDef, RemoteGraderDef

        if not isinstance(grader_def, RemoteGraderDef):
            raise TypeError(f"Expected RemoteGraderDef, got {type(grader_def).__name__}")

        source = self._sources.get(grader_def.source)
        if source is None:
            raise ValueError(f"Unknown grader source '{grader_def.source}'. Available: {sorted(self._sources.keys())}")

        dest = self._cache_dir / grader_def.source / grader_def.ref
        if not dest.exists():
            logger.info(
                "Downloading grader '%s' from %s (ref: %s)",
                grader_def.name,
                grader_def.source,
                grader_def.ref,
            )
            await source.fetch_grader(grader_def.ref, dest)
        else:
            logger.debug("Using cached grader '%s' at %s", grader_def.name, dest)

        return CodeGraderDef(
            name=grader_def.name,
            path=str(dest),
            threshold=grader_def.threshold,
            timeout=grader_def.timeout,
            config=grader_def.config,
            executor=grader_def.executor,
        )


_default_resolver: GraderResolver | None = None


def get_default_resolver() -> GraderResolver:
    """Return a lazily-initialized resolver with all registered sources."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = GraderResolver()
        for source in get_sources():
            _default_resolver.register_source(source)
    return _default_resolver
