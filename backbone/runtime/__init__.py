"""Runtime: the orchestrator that wires the eight Backbone sub-modules.

Public API:
    * ``Orchestrator(config_path)`` — build and run the full pipeline.
"""

from .orchestrator import Orchestrator

__all__ = ["Orchestrator"]
