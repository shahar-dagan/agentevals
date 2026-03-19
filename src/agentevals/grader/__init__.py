"""Grader management: sources, templates, and resolution."""

from .resolver import GraderResolver, get_default_resolver
from .sources import (
    BuiltinGraderSource,
    FileGraderSource,
    GitHubGraderSource,
    GraderInfo,
    GraderSource,
    get_sources,
)
from .templates import scaffold_grader

__all__ = [
    "BuiltinGraderSource",
    "FileGraderSource",
    "GitHubGraderSource",
    "GraderInfo",
    "GraderResolver",
    "GraderSource",
    "get_default_resolver",
    "get_sources",
    "scaffold_grader",
]
