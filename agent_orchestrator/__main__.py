"""Entry point so ``python -m agent_orchestrator`` works."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
