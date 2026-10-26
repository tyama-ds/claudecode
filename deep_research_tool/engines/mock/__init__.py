"""Deterministic mock engines (identical to the orchestrator's runners/mock).

Never shown in the Web UI unless the developer toggle in ⚙設定 → 機能 is
on; used by the tests and for trying the engine path without any
network or paid API.
"""

from .mock_engines import ALL_ENGINES  # noqa: F401
