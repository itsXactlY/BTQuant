#!/usr/bin/env python3
"""
Tests for the list_agent_branches.py script
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the current directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import list_agent_branches  # noqa: E402


class TestGetAgentBranches(unittest.TestCase):
    """Test cases for the get_agent_branches function"""

    @patch('subprocess.run')
    def test_get_agent_branches_success(self, mock_run):
        """Test that the function correctly extracts agent branches from git output"""
        # Mock the git command output
        mock_result = MagicMock()
        mock_result.stdout = """  main
* ralphy-base
  ralphy/agent-1-test-branch
  ralphy/agent-2-another-test
  feature/new-feature
  ralphy/agent-3-yet-another
  old-branch"""
        mock_result.check_returncode.return_value = None
        mock_run.return_value = mock_result

        # Call the function
        branches = list_agent_branches.get_agent_branches()

        # Verify the results
        expected_branches = [
            'ralphy/agent-1-test-branch',
            'ralphy/agent-2-another-test',
            'ralphy/agent-3-yet-another'
        ]
        self.assertEqual(branches, expected_branches)

        # Verify subprocess.run was called correctly
        mock_run.assert_called_once_with(['git', 'branch', '-a'],
                                         capture_output=True,
                                         text=True,
                                         check=True)

    @patch('subprocess.run')
    def test_get_agent_branches_no_matches(self, mock_run):
        """Test that the function returns empty list when no agent branches exist"""
        # Mock the git command output with no agent branches
        mock_result = MagicMock()
        mock_result.stdout = """  main
* ralphy-base
  feature/new-feature
  old-branch"""
        mock_result.check_returncode.return_value = None
        mock_run.return_value = mock_result

        # Call the function
        branches = list_agent_branches.get_agent_branches()

        # Verify the results
        self.assertEqual(branches, [])

    @patch('subprocess.run')
    def test_get_agent_branches_current_branch_is_agent_branch(self, mock_run):
        """Test that the function correctly handles when current branch is an
        agent branch"""
        # Mock the git command output with current branch being an agent branch
        mock_result = MagicMock()
        mock_result.stdout = """  main
  ralphy/agent-1-test-branch
* ralphy/agent-2-current-branch"""
        mock_result.check_returncode.return_value = None
        mock_run.return_value = mock_result

        # Call the function
        branches = list_agent_branches.get_agent_branches()

        # Verify the results
        expected_branches = [
            'ralphy/agent-1-test-branch',
            'ralphy/agent-2-current-branch'
        ]
        self.assertEqual(branches, expected_branches)

    @patch('subprocess.run')
    def test_get_agent_branches_error_handling(self, mock_run):
        """Test that the function handles git command errors gracefully"""
        # Mock a CalledProcessError
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, ['git', 'branch', '-a'])

        # Call the function
        branches = list_agent_branches.get_agent_branches()

        # Verify the results
        self.assertEqual(branches, [])

    @patch('subprocess.run')
    def test_get_agent_branches_empty_output(self, mock_run):
        """Test that the function handles empty git output"""
        # Mock empty git command output
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.check_returncode.return_value = None
        mock_run.return_value = mock_result

        # Call the function
        branches = list_agent_branches.get_agent_branches()

        # Verify the results
        self.assertEqual(branches, [])


class TestMainFunction(unittest.TestCase):
    """Test cases for the main function"""

    @patch('builtins.print')
    @patch('list_agent_branches.get_agent_branches')
    def test_main_with_branches(self, mock_get_branches, mock_print):
        """Test main function when there are agent branches"""
        # Mock the get_agent_branches function to return some branches
        mock_get_branches.return_value = [
            'ralphy/agent-1-test',
            'ralphy/agent-2-test'
        ]

        # Call the main function
        list_agent_branches.main()

        # Verify that print was called appropriately
        self.assertTrue(mock_print.called)
        # Check that it prints the header
        mock_print.assert_any_call("Finding branches with pattern 'ralphy/agent-*':")
        # Check that it prints the total count
        mock_print.assert_any_call("Total: 2 branches")

    @patch('builtins.print')
    @patch('list_agent_branches.get_agent_branches')
    def test_main_without_branches(self, mock_get_branches, mock_print):
        """Test main function when there are no agent branches"""
        # Mock the get_agent_branches function to return no branches
        mock_get_branches.return_value = []

        # Call the main function
        list_agent_branches.main()

        # Verify that print was called appropriately
        self.assertTrue(mock_print.called)
        # Check that it prints the header
        mock_print.assert_any_call("Finding branches with pattern 'ralphy/agent-*':")
        # Check that it prints the no branches message
        mock_print.assert_any_call(
            "No branches found matching the pattern 'ralphy/agent-*'"
        )


if __name__ == '__main__':
    unittest.main()
