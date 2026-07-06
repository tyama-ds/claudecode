"""Tests for server-side helpers (workspace directory creation)."""

import os
import tempfile
import unittest

from agent_orchestrator.server.app import _ensure_workspace_dir


class TestWorkspaceDir(unittest.TestCase):
    def test_creates_directory_no_git(self):
        path = os.path.join(tempfile.mkdtemp(), "fresh", "proj")
        self.assertEqual(_ensure_workspace_dir(path), "created")
        self.assertTrue(os.path.isdir(path))
        self.assertFalse(os.path.isdir(os.path.join(path, ".git")))  # no git involved

    def test_existing_directory_reports_exists(self):
        path = tempfile.mkdtemp()
        self.assertEqual(_ensure_workspace_dir(path), "exists")


if __name__ == "__main__":
    unittest.main()
