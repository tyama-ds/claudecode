"""
Configuration for the Prompt Optimization Agent.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Server
HOST = os.getenv("PROMPT_OPT_HOST", "0.0.0.0")
PORT = int(os.getenv("PROMPT_OPT_PORT", "8501"))

# Defaults
DEFAULT_PROVIDER = os.getenv("PROMPT_OPT_PROVIDER", "openai")
DEFAULT_MODEL_OPENAI = os.getenv("PROMPT_OPT_MODEL_OPENAI", "gpt-5-mini")
DEFAULT_MODEL_ANTHROPIC = os.getenv("PROMPT_OPT_MODEL_ANTHROPIC", "claude-3-5-sonnet-20241022")
DEFAULT_TEMPERATURE = float(os.getenv("PROMPT_OPT_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("PROMPT_OPT_MAX_TOKENS", "4096"))
