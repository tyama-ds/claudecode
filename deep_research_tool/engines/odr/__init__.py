"""Open Deep Research engine (LangChain ``open-deep-research``).

- ``engine.py``      Engine implementation + pure logic (never imports the
                     engine's packages; runs on the tool's own Python)
- ``worker.py``      the subprocess that runs the graph in the engine's
                     virtualenv (self-contained; JSONL protocol)
- ``mcp_search.py``  the in-process MCP server that lends the tool's own
                     search (DuckDuckGo / browser / SearXNG) to the engine
"""
