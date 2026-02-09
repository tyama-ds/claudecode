#!/usr/bin/env python
"""
Launcher script for Deep Research Tool GUI.

Usage:
    python -m deep_research_tool2.run_gui
    or
    python deep_research_tool2/run_gui.py
"""

import sys
import os

# Add parent directory to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_research_tool2.gui import main

if __name__ == "__main__":
    main()
