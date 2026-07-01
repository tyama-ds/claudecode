"""Tests for server-side helpers (workspace repo creation)."""

import os
import tempfile
import unittest

from agent_orchestrator.server.app import _init_workspace_repo


class TestWorkspaceRepo(unittest.TestCase):
    def test_creates_directory_and_reports_status(self):
        path = os.path.join(tempfile.mkdtemp(), "fresh", "repo")
        result = _init_workspace_repo(path)
        self.assertTrue(os.path.isdir(path))  # created even if git is absent
        self.assertIn(result, ("init", "exists", "nogit"))

    def test_second_init_reports_existing_repo(self):
        path = os.path.join(tempfile.mkdtemp(), "repo")
        first = _init_workspace_repo(path)
        if first == "init":  # git available -> a .git now exists
            self.assertTrue(os.path.isdir(os.path.join(path, ".git")))
            self.assertEqual(_init_workspace_repo(path), "exists")


if __name__ == "__main__":
    unittest.main()
