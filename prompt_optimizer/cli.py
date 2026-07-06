"""
CLI entry point for the Prompt Optimization Agent.

Usage:
    python -m prompt_optimizer            # Start the web server
    python -m prompt_optimizer --port 8080
"""

import argparse

from .config import HOST, PORT


def main():
    parser = argparse.ArgumentParser(description="Prompt Optimization Agent")
    parser.add_argument("--host", default=HOST, help=f"Host to bind (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "prompt_optimizer.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
