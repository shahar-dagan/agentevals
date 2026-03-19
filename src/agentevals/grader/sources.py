"""Grader source backends: discover and fetch graders from various registries."""

from __future__ import annotations

import abc
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_REPO = "agentevals-dev/graders"
_DEFAULT_BRANCH = "main"
_DEFAULT_INDEX = "index.yaml"


@dataclass
class GraderInfo:
    """Metadata for a single grader, regardless of where it comes from."""

    name: str
    description: str
    source: str
    language: str | None = None
    ref: str | None = None
    tags: list[str] = field(default_factory=list)
    author: str | None = None


class GraderSource(abc.ABC):
    """Registry backend that can list and fetch graders."""

    @property
    @abc.abstractmethod
    def source_name(self) -> str: ...

    @abc.abstractmethod
    async def list_graders(self) -> list[GraderInfo]: ...

    @abc.abstractmethod
    async def fetch_grader(self, ref: str, dest: Path) -> Path:
        """Download a grader identified by *ref* and write it to *dest*.

        Returns the path to the downloaded file.
        """


_CACHE_TTL_SECONDS = 86400  # 24 hours


def _cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "agentevals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_cache(key: str, ttl: int = _CACHE_TTL_SECONDS) -> list[GraderInfo] | None:
    cache_file = _cache_dir() / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        if time.time() - data.get("ts", 0) > ttl:
            return None
        return [GraderInfo(**item) for item in data["graders"]]
    except Exception:
        return None


def _write_cache(key: str, graders: list[GraderInfo]) -> None:
    cache_file = _cache_dir() / f"{key}.json"
    try:
        cache_file.write_text(
            json.dumps(
                {
                    "ts": time.time(),
                    "graders": [asdict(g) for g in graders],
                }
            )
        )
    except Exception:
        pass


class BuiltinGraderSource(GraderSource):
    """Wraps ADK's built-in metric registry as a grader source."""

    @property
    def source_name(self) -> str:
        return "builtin"

    async def list_graders(self) -> list[GraderInfo]:
        import asyncio
        import warnings

        cached = _read_cache("builtin")
        if cached is not None:
            return cached

        def _load() -> list[GraderInfo]:
            infos: list[GraderInfo] = []
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from google.adk.evaluation.metric_evaluator_registry import (
                        DEFAULT_METRIC_EVALUATOR_REGISTRY,
                    )

                for m in DEFAULT_METRIC_EVALUATOR_REGISTRY.get_registered_metrics():
                    infos.append(
                        GraderInfo(
                            name=m.metric_name,
                            description=m.description or "No description",
                            source=self.source_name,
                            language=None,
                            ref=None,
                        )
                    )
            except ImportError:
                from google.adk.evaluation.eval_metrics import PrebuiltMetrics

                for pm in PrebuiltMetrics:
                    infos.append(
                        GraderInfo(
                            name=pm.value,
                            description="(install google-adk[eval] for full details)",
                            source=self.source_name,
                        )
                    )
            return infos

        result = await asyncio.to_thread(_load)
        _write_cache("builtin", result)
        return result

    async def fetch_grader(self, ref: str, dest: Path) -> Path:
        raise NotImplementedError("Built-in graders are part of ADK and cannot be fetched as files.")


class GitHubGraderSource(GraderSource):
    """Fetches graders from a GitHub repository with a CI-generated index.yaml."""

    def __init__(
        self,
        repo: str | None = None,
        branch: str | None = None,
        index_path: str | None = None,
        token: str | None = None,
    ):
        self._repo = repo or os.environ.get("AGENTEVALS_GRADER_REPO", _DEFAULT_REPO)
        self._branch = branch or os.environ.get("AGENTEVALS_GRADER_BRANCH", _DEFAULT_BRANCH)
        self._index_path = index_path or _DEFAULT_INDEX
        self._token = token or os.environ.get("AGENTEVALS_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")

    @property
    def source_name(self) -> str:
        return "github"

    def _raw_url(self, path: str) -> str:
        return f"https://raw.githubusercontent.com/{self._repo}/{self._branch}/{path}"

    def _headers(self) -> dict[str, str]:
        if self._token:
            return {"Authorization": f"token {self._token}"}
        return {}

    async def list_graders(self) -> list[GraderInfo]:
        import httpx

        url = self._raw_url(self._index_path)
        logger.debug("Fetching grader index from %s", url)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=self._headers(), timeout=15)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch grader index from %s: %s", url, exc)
            return []

        data = yaml.safe_load(resp.text)
        if not isinstance(data, dict):
            logger.warning("Grader index at %s is not a valid YAML mapping", url)
            return []

        infos: list[GraderInfo] = []
        for entry in data.get("graders", []):
            infos.append(
                GraderInfo(
                    name=entry.get("name", "unknown"),
                    description=entry.get("description", ""),
                    source=self.source_name,
                    language=entry.get("language"),
                    ref=entry.get("path"),
                    tags=entry.get("tags", []),
                    author=entry.get("author"),
                )
            )
        return infos

    async def fetch_grader(self, ref: str, dest: Path) -> Path:
        import httpx

        url = self._raw_url(ref)
        logger.info("Downloading grader from %s", url)

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=self._headers(), timeout=30)
            resp.raise_for_status()

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(resp.text, encoding="utf-8")
        return dest


class FileGraderSource(GraderSource):
    """Reads graders from a local index.yaml file (same schema as GitHubGraderSource).

    Useful for testing and local development.  Not registered in the default
    source list — instantiate directly when needed.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Grader index file not found: {self._path}")

    @property
    def source_name(self) -> str:
        return "file"

    async def list_graders(self) -> list[GraderInfo]:
        data = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("Grader index at %s is not a valid YAML mapping", self._path)
            return []

        infos: list[GraderInfo] = []
        for entry in data.get("graders", []):
            infos.append(
                GraderInfo(
                    name=entry.get("name", "unknown"),
                    description=entry.get("description", ""),
                    source=self.source_name,
                    language=entry.get("language"),
                    ref=entry.get("path"),
                    tags=entry.get("tags", []),
                    author=entry.get("author"),
                )
            )
        return infos

    async def fetch_grader(self, ref: str, dest: Path) -> Path:
        src = (self._path.parent / ref).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Grader file not found: {src} (ref: {ref}, index: {self._path})")
        import shutil

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_SOURCES: list[GraderSource] | None = None


def get_sources() -> list[GraderSource]:
    """Return all registered grader sources (lazily initialized)."""
    global _SOURCES
    if _SOURCES is None:
        _SOURCES = [
            BuiltinGraderSource(),
            GitHubGraderSource(),
        ]
    return _SOURCES


def register_source(source: GraderSource) -> None:
    """Add a custom grader source to the registry."""
    get_sources().append(source)
