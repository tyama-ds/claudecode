"""Per-engine Python environments for the external research engines.

GPT Researcher and Open Deep Research pin dependency sets that do not
coexist with this package (and need Python 3.12), so each runs in its
own interpreter as a subprocess. This module only FINDS and CHECKS that
interpreter; it never installs anything by itself (see
``deep_research_tool/scripts/setup_engine_venv.py`` for the setup).

Resolution order for an engine's Python:

1. ``config.research.engine_python[engine_id]`` (Web UI ⚙設定 / create_config)
2. the environment variable ``DRT_ENGINE_PYTHON_<ENGINE_ID>`` (``-`` → ``_``)
3. ``<engines_venv>/<engine_id>/bin/python`` (``Scripts\\python.exe`` on
   Windows) where ``<engines_venv>`` is ``$DRT_ENGINES_VENV_DIR`` or the
   repository's ``engines_venv/`` folder

A missing interpreter is reported, never silently replaced by the server's
own Python (its packages are not the engine's).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Pinned package sets (identical to deep-research-orchestrator's runners).
ENGINE_SPECS: Dict[str, Dict[str, Any]] = {
    "gpt-researcher": {
        "name": "GPT Researcher",
        "import": "gpt_researcher",
        "dist": "gpt-researcher",
        "version": "0.16.0",
        "min_python": (3, 12),
        "packages": ["gpt-researcher==0.16.0"],
    },
    "open-deep-research": {
        "name": "Open Deep Research",
        "import": "open_deep_research",
        "dist": "open-deep-research",
        "version": "0.0.16",
        "min_python": (3, 12),
        "packages": [
            "open-deep-research==0.0.16",
            "mcp==1.28.1",
            "httpx==0.28.1",
            "langchain-mcp-adapters==0.3.0",
        ],
    },
}

_CHECK_TTL_SECONDS = 60.0
_check_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
_check_lock = threading.Lock()


def venv_root() -> Path:
    """Where the per-engine virtualenvs live by default."""
    env = os.environ.get("DRT_ENGINES_VENV_DIR")
    if env:
        return Path(env).expanduser()
    # <repo>/engines_venv (the package sits at <repo>/deep_research_tool)
    return Path(__file__).resolve().parents[2] / "engines_venv"


def default_venv_python(engine_id: str) -> Path:
    base = venv_root() / engine_id
    if os.name == "nt":
        return base / "Scripts" / "python.exe"
    return base / "bin" / "python"


def env_var_name(engine_id: str) -> str:
    return "DRT_ENGINE_PYTHON_" + engine_id.upper().replace("-", "_")


def resolve_engine_python(engine_id: str,
                          configured: Optional[Dict[str, str]] = None
                          ) -> Tuple[Optional[str], str]:
    """(interpreter path or None, where it came from: config / env /
    default / none). Only an EXISTING file is returned."""
    cand = (configured or {}).get(engine_id)
    if cand:
        p = Path(str(cand)).expanduser()
        return (str(p), "config") if p.is_file() else (str(p), "config-missing")
    cand = os.environ.get(env_var_name(engine_id))
    if cand:
        p = Path(cand).expanduser()
        return (str(p), "env") if p.is_file() else (str(p), "env-missing")
    p = default_venv_python(engine_id)
    if p.is_file():
        return str(p), "default"
    return None, "none"


def setup_command(engine_id: str) -> str:
    """The one command that creates the engine's environment."""
    py = "py -3.12" if os.name == "nt" else "python3.12"
    return f"{py} -m deep_research_tool.scripts.setup_engine_venv {engine_id}"


def _check_code(spec: Dict[str, Any]) -> str:
    mod = spec["import"]
    dist = spec.get("dist") or mod
    return (
        "import sys, json\n"
        "out = {'python_version': '.'.join(map(str, sys.version_info[:3]))}\n"
        "try:\n"
        f"    import {mod}\n"
        "    out['ok'] = True\n"
        "    try:\n"
        "        from importlib.metadata import version as _v\n"
        f"        out['package_version'] = _v({dist!r})\n"
        "    except Exception:\n"
        f"        out['package_version'] = getattr({mod}, '__version__', None)\n"
        "except Exception as e:\n"
        "    out['ok'] = False\n"
        "    out['error'] = type(e).__name__ + ': ' + str(e)\n"
        "print(json.dumps(out))\n"
    )


def check_engine_python(engine_id: str, python_path: str,
                        timeout: float = 30.0, use_cache: bool = True
                        ) -> Dict[str, Any]:
    """Run the interpreter once and report whether the engine's package
    imports there. ``{ok, python_version, package_version, error}``;
    cached for a minute per (engine, path, mtime) so the catalog stays cheap."""
    spec = ENGINE_SPECS.get(engine_id)
    if spec is None:
        return {"ok": False, "error": f"unknown engine: {engine_id}"}
    try:
        mtime = str(os.path.getmtime(python_path))
    except OSError:
        return {"ok": False, "error": f"interpreter not found: {python_path}"}
    key = (engine_id, f"{python_path}|{mtime}")
    now = time.monotonic()
    if use_cache:
        with _check_lock:
            hit = _check_cache.get(key)
            if hit and now - hit[0] < _CHECK_TTL_SECONDS:
                return dict(hit[1])
    try:
        proc = subprocess.run(
            [python_path, "-I", "-c", _check_code(spec)],
            capture_output=True, text=True, timeout=timeout)
        line = (proc.stdout or "").strip().splitlines()
        out = json.loads(line[-1]) if line else {}
        if not isinstance(out, dict) or "python_version" not in out:
            out = {"ok": False,
                   "error": (proc.stderr or proc.stdout or "no output").strip()[-400:]}
    except subprocess.TimeoutExpired:
        out = {"ok": False, "error": f"interpreter did not answer within {timeout:.0f}s"}
    except (OSError, ValueError) as e:
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if out.get("python_version"):
        try:
            parts = tuple(int(x) for x in str(out["python_version"]).split(".")[:2])
            out["python_ok"] = parts >= tuple(spec["min_python"])
        except ValueError:
            out["python_ok"] = False
    with _check_lock:
        _check_cache[key] = (now, dict(out))
    return out


def clear_check_cache() -> None:
    with _check_lock:
        _check_cache.clear()


def engine_health(engine_id: str,
                  configured: Optional[Dict[str, str]] = None,
                  check: bool = True) -> Dict[str, Any]:
    """Catalog entry fragment for one external engine:
    ``health`` = available | needs_setup | unsupported, a Japanese
    ``reason`` the UI shows under the badge, the resolved ``python`` and
    the ``setup_command``."""
    spec = ENGINE_SPECS.get(engine_id) or {}
    min_py = ".".join(str(x) for x in spec.get("min_python", (3, 12)))
    path, source = resolve_engine_python(engine_id, configured)
    out: Dict[str, Any] = {"python": path, "python_source": source,
                           "setup_command": setup_command(engine_id),
                           "min_python": min_py,
                           "env_var": env_var_name(engine_id)}
    if path is None or source.endswith("-missing"):
        where = f"設定された Python（{path}）が見つかりません。" if path else \
            "専用の Python 環境がありません。"
        out.update(health="needs_setup",
                   reason=f"{where}Python {min_py} 以上で "
                          f"`{setup_command(engine_id)}` を実行してください")
        return out
    if not check:
        out.update(health="available", reason=None)
        return out
    res = check_engine_python(engine_id, path)
    out["check"] = res
    if res.get("python_version") and not res.get("python_ok", True):
        out.update(health="unsupported",
                   reason=f"Python {min_py} 以上が必要です（{path} は "
                          f"{res.get('python_version')}）")
        return out
    if not res.get("ok"):
        err = str(res.get("error") or "import failed")
        out.update(health="needs_setup",
                   reason=f"{path} に {spec.get('dist', engine_id)} が入っていません"
                          f"（{err[:160]}）。`{setup_command(engine_id)}` で作成し直してください")
        return out
    out.update(health="available", reason=None,
               package_version=res.get("package_version"),
               python_version=res.get("python_version"))
    return out


def current_python_meets(engine_id: str) -> bool:
    spec = ENGINE_SPECS.get(engine_id) or {}
    return tuple(sys.version_info[:2]) >= tuple(spec.get("min_python", (3, 12)))
