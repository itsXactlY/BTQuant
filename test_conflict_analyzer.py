#!/usr/bin/env python3
"""
Tests for the conflict analyzer
"""

import tempfile
import os
import unittest
from conflict_analyzer import find_conflicted_files, analyze_conflict_details


class TestConflictAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up temporary directory with test files"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a file with conflicts
        self.conflict_file = os.path.join(self.temp_dir, "conflict_test.py")
        with open(self.conflict_file, 'w') as f:
            f.write("""# This is a test file
<<<<<<< HEAD
def hello():
    print("Hello from HEAD")
=======
def hello():
    print("Hello from branch")
>>>>>>> branch-name
""")
        
        # Create a file without conflicts
        self.clean_file = os.path.join(self.temp_dir, "clean_test.py")
        with open(self.clean_file, 'w') as f:
            f.write("""# This is a clean file
def hello():
    print("Hello world")
""")
        
        # Create another file with multiple conflicts
        self.multi_conflict_file = os.path.join(self.temp_dir, "multi_conflict_test.py")
        with open(self.multi_conflict_file, 'w') as f:
            f.write("""# Another test file
<<<<<<< HEAD
x = 1
=======
x = 2
>>>>>>> branch-name

some_code = True

<<<<<<< HEAD
y = 10
=======
y = 20
>>>>>>> branch-name
""")

    def test_find_conflicted_files(self):
        """Test finding files with conflicts"""
        conflicted = find_conflicted_files(self.temp_dir, ['.py'])
        
        # Should find the two files with conflicts
        self.assertIn(self.conflict_file, conflicted)
        self.assertIn(self.multi_conflict_file, conflicted)
        # Should not find the clean file
        self.assertNotIn(self.clean_file, conflicted)
        
        # Test with specific extensions
        conflicted_py = find_conflicted_files(self.temp_dir, ['.py'])
        self.assertEqual(len(conflicted_py), 2)

    def test_analyze_conflict_details_single(self):
        """Test analyzing conflict details in a single-conflict file"""
        stats = analyze_conflict_details(self.conflict_file)
        
        self.assertEqual(stats['conflict_blocks'], 1)
        self.assertEqual(stats['conflict_start_markers'], 1)
        self.assertEqual(stats['conflict_separator_markers'], 1)
        self.assertEqual(stats['conflict_end_markers'], 1)
        self.assertGreater(stats['lines_in_conflicts'], 0)

    def test_analyze_conflict_details_multiple(self):
        """Test analyzing conflict details in a multi-conflict file"""
        stats = analyze_conflict_details(self.multi_conflict_file)
        
        self.assertEqual(stats['conflict_blocks'], 2)
        self.assertEqual(stats['conflict_start_markers'], 2)
        self.assertEqual(stats['conflict_separator_markers'], 2)
        self.assertEqual(stats['conflict_end_markers'], 2)
        self.assertGreater(stats['lines_in_conflicts'], 0)

    def test_analyze_conflict_details_clean(self):
        """Test analyzing conflict details in a clean file"""
        stats = analyze_conflict_details(self.clean_file)
        
        self.assertEqual(stats['conflict_blocks'], 0)
        self.assertEqual(stats['conflict_start_markers'], 0)
        self.assertEqual(stats['conflict_separator_markers'], 0)
        self.assertEqual(stats['conflict_end_markers'], 0)
        self.assertEqual(stats['lines_in_conflicts'], 0)

    def test_empty_file(self):
        """Test analyzing an empty file"""
        empty_file = os.path.join(self.temp_dir, "empty.py")
        with open(empty_file, 'w') as f:
            f.write("")
        
        stats = analyze_conflict_details(empty_file)
        
        self.assertEqual(stats['total_lines'], 0)
        self.assertEqual(stats['conflict_blocks'], 0)

    def test_file_with_conflict_markers_but_not_conflicts(self):
        """Test a file that contains conflict-like text but isn't actually conflicted"""
        text_like_conflict = os.path.join(self.temp_dir, "text_like_conflict.py")
        with open(text_like_conflict, 'w') as f:
            f.write("# This is not a conflict: <<<<<<< HEAD\n")
            f.write("# This is not a conflict: =======\n")
            f.write("# This is not a conflict: >>>>>>> branch\n")
        
        stats = analyze_conflict_details(text_like_conflict)
        
        # Should not count as conflicts since they're not on separate lines
        # with proper formatting
        self.assertEqual(stats['conflict_blocks'], 0)


if __name__ == '__main__':
    unittest.main()