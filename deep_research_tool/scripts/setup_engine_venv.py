"""Create the virtualenv of an external research engine.

    python3.12 -m deep_research_tool.scripts.setup_engine_venv gpt-researcher
    python3.12 -m deep_research_tool.scripts.setup_engine_venv open-deep-research --dir /opt/drt_envs

The interpreter that runs this script becomes the engine's Python (it
must satisfy the engine's minimum version), the pinned packages are
installed with pip, and the resulting path is printed. The Web UI's
調査エンジン cards pick it up on the next refresh (default location) or
after the path is entered in ⚙設定 / the DRT_ENGINE_PYTHON_<ID> variable
(custom ``--dir``).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import venv
from pathlib import Path


def _load_venv_module():
    """engines/venv.py by path: this script runs under a bare Python 3.12
    that has none of the package's dependencies, so the package itself
    (deep_research_tool/__init__) must not be imported."""
    path = Path(__file__).resolve().parent.parent / "engines" / "venv.py"
    spec = importlib.util.spec_from_file_location("_drt_engines_venv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_venv = _load_venv_module()
ENGINE_SPECS, env_var_name, venv_root = _venv.ENGINE_SPECS, _venv.env_var_name, _venv.venv_root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("engine", choices=sorted(ENGINE_SPECS))
    parser.add_argument("--dir", help="base folder for the environments "
                        "(default: $DRT_ENGINES_VENV_DIR or <repo>/engines_venv)")
    parser.add_argument("--no-install", action="store_true",
                        help="create the environment only (print the pip command)")
    args = parser.parse_args(argv)

    spec = ENGINE_SPECS[args.engine]
    need = tuple(spec["min_python"])
    if tuple(sys.version_info[:2]) < need:
        print(f"error: {args.engine} needs Python {need[0]}.{need[1]}+ "
              f"(this interpreter is {sys.version.split()[0]}). "
              f"Run this script with a newer interpreter.", file=sys.stderr)
        return 2

    base = Path(args.dir).expanduser() if args.dir else venv_root()
    target = base / args.engine
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"creating {target} with {sys.executable}")
    venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=False).create(str(target))
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    cmd = [str(python), "-m", "pip", "install", "--upgrade", *spec["packages"]]
    if args.no_install:
        print("install with: " + " ".join(cmd))
    else:
        print("installing: " + " ".join(spec["packages"]))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print("error: pip install failed", file=sys.stderr)
            return proc.returncode
    print(f"\n{args.engine} python: {python}")
    if args.dir:
        print(f"custom location: set {env_var_name(args.engine)}={python} "
              f"(or enter the path in ⚙設定 → 調査エンジン) so the tool finds it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
