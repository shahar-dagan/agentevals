"""agentevals: Standalone CLI to evaluate agent traces using ADK's scoring framework."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("agentevals")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
