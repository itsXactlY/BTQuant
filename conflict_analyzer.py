#!/usr/bin/env python3
"""
Conflict Analyzer - A tool to analyze merge conflicts in files
"""

import os
import argparse
from typing import Dict, List


def find_conflicted_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Find all files with merge conflict markers in the given directory

    Args:
        directory: Directory to search in
        extensions: List of file extensions to check (e.g., ['.py', '.js', '.md'])

    Returns:
        List of file paths with conflicts
    """
    conflicted_files = []

    # Default to common text-based extensions if none provided
    if extensions is None:
        extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.txt', '.json',
                      '.yaml', '.yml', '.html', '.css', '.java', '.cpp', '.h', '.c']

    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.vscode', '.ralphy-worktrees']]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                        # Check for common conflict markers
                        has_conflicts = (
                            '<<<<<<< HEAD' in content
                            or '=======\n' in content
                            or '>>>>>>> ' in content
                            or '<<<<<<< ' in content
                        )

                        if has_conflicts:
                            conflicted_files.append(file_path)

                except Exception:
                    # Skip files that can't be read
                    continue

    return conflicted_files


def analyze_conflict_details(file_path: str) -> Dict[str, int]:
    """
    Analyze conflict details in a specific file

    Args:
        file_path: Path to the file to analyze

    Returns:
        Dictionary with conflict statistics
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    stats = {
        'total_lines': len(lines),
        'conflict_blocks': 0,
        'conflict_start_markers': 0,
        'conflict_separator_markers': 0,
        'conflict_end_markers': 0,
        'lines_in_conflicts': 0
    }

    in_conflict = False
    current_conflict_lines = 0

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith('<<<<<<<'):
            stats['conflict_start_markers'] += 1
            in_conflict = True
            current_conflict_lines = 0

        elif stripped_line == '=======':
            stats['conflict_separator_markers'] += 1
            if in_conflict:
                current_conflict_lines += 1

        elif stripped_line.startswith('>>>>>>>'):
            stats['conflict_end_markers'] += 1
            if in_conflict:
                current_conflict_lines += 1
                stats['lines_in_conflicts'] += current_conflict_lines
                stats['conflict_blocks'] += 1
                in_conflict = False
                current_conflict_lines = 0
        else:
            if in_conflict:
                current_conflict_lines += 1

    return stats


def print_conflict_report(conflicted_files: List[str], verbose: bool = False):
    """
    Print a detailed report of conflicts found

    Args:
        conflicted_files: List of files with conflicts
        verbose: Whether to show detailed analysis for each file
    """
    print(f"\n🔍 Found {len(conflicted_files)} files with merge conflicts:\n")

    if not conflicted_files:
        print("✅ No conflicts detected!")
        return

    total_conflict_blocks = 0
    total_lines_in_conflicts = 0

    for file_path in conflicted_files:
        stats = analyze_conflict_details(file_path)
        total_conflict_blocks += stats['conflict_blocks']
        total_lines_in_conflicts += stats['lines_in_conflicts']

        print(f"📄 {file_path}")
        print(f"   📊 Conflict blocks: {stats['conflict_blocks']}")
        print(f"   📝 Lines in conflicts: {stats['lines_in_conflicts']}")
        print(f"   📏 Total lines: {stats['total_lines']}")

        if verbose:
            print("   🔍 Details:")
            print(f"      - Start markers: {stats['conflict_start_markers']}")
            print(f"      - Separator markers: {stats['conflict_separator_markers']}")
            print(f"      - End markers: {stats['conflict_end_markers']}")
            print()

    print("\n📈 Summary:")
    print(f"   📦 Total files with conflicts: {len(conflicted_files)}")
    print(f"   🔄 Total conflict blocks: {total_conflict_blocks}")
    print(f"   📄 Total lines in conflicts: {total_lines_in_conflicts}")


def main():
    parser = argparse.ArgumentParser(description='Analyze merge conflicts in files')
    parser.add_argument(
        'directory', nargs='?', default='.',
        help='Directory to search for conflicts (default: current directory)'
    )
    parser.add_argument(
        '-e', '--extensions', nargs='+',
        help='File extensions to check (e.g., .py .js .md)'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show detailed analysis for each file'
    )
    parser.add_argument(
        '--list-files-only', action='store_true',
        help='Only list the conflicted files without analysis'
    )

    args = parser.parse_args()

    print("🔍 Analyzing merge conflicts...")

    conflicted_files = find_conflicted_files(args.directory, args.extensions)

    if args.list_files_only:
        for file_path in conflicted_files:
            print(file_path)
        return

    print_conflict_report(conflicted_files, args.verbose)


if __name__ == "__main__":
    main()
