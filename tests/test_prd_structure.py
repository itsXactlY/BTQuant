"""
Test script to validate the structure of the reworked PRD_look.md file
"""
import os
import re


def test_prd_structure():
    """Validate that the PRD file has proper structure for parallel development"""
    
    # Check if the new PRD file exists
    prd_file_path = "PRD_look_new.md"
    assert os.path.exists(prd_file_path), f"PRD file {prd_file_path} does not exist"
    
    with open(prd_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that file contains module structure
    assert "## Module" in content, "File should contain module sections"
    
    # Check that file contains complexity indicators
    assert "[Complexity:" in content, "File should contain complexity indicators"
    
    # Check that file contains dependencies information
    assert "**Dependencies:**" in content, "File should contain dependencies information"
    
    # Check that file contains prerequisites information
    assert "**Prerequisites:**" in content, "File should contain prerequisites information"
    
    # Count number of modules
    modules = re.findall(r'## Module \d+:', content)
    assert len(modules) > 10, f"Expected at least 10 modules, found {len(modules)}"
    
    # Check that tasks have checkboxes
    tasks = re.findall(r'- \[ \]', content)  # Unchecked tasks
    checked_tasks = re.findall(r'- \[x\]', content)  # Checked tasks
    all_tasks = tasks + checked_tasks
    
    assert len(all_tasks) > 100, f"Expected at least 100 tasks, found {len(all_tasks)}"
    
    print(f"✓ Validation passed!")
    print(f"  - Found {len(modules)} modules")
    print(f"  - Found {len(all_tasks)} tasks")
    print(f"  - Contains complexity indicators: {'[Complexity:' in content}")
    print(f"  - Contains dependencies info: {'**Dependencies:**' in content}")
    print(f"  - Contains prerequisites info: {'**Prerequisites:**' in content}")


def test_original_vs_new_structure():
    """Compare original and new PRD structures"""
    
    original_file = "PRD_look.md"
    new_file = "PRD_look_new.md"
    
    # Check both files exist
    assert os.path.exists(original_file), f"Original PRD file {original_file} does not exist"
    assert os.path.exists(new_file), f"New PRD file {new_file} does not exist"
    
    with open(original_file, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    with open(new_file, 'r', encoding='utf-8') as f:
        new_content = f.read()
    
    # Original should have flat structure (just tasks in a list)
    original_tasks = re.findall(r'- \[ \]', original_content) + re.findall(r'- \[x\]', original_content)
    
    # New should have modular structure
    new_modules = re.findall(r'## Module \d+:', new_content)
    new_complexity = "[Complexity:" in new_content
    new_dependencies = "**Dependencies:**" in new_content
    
    print(f"\nComparison:")
    print(f"  Original: {len(original_tasks)} tasks in flat structure")
    print(f"  New: {len(new_modules)} modules with hierarchical structure")
    print(f"  New has complexity indicators: {new_complexity}")
    print(f"  New has dependencies: {new_dependencies}")
    
    assert len(new_modules) > 0, "New PRD should have modules"
    assert new_complexity, "New PRD should have complexity indicators"
    assert new_dependencies, "New PRD should have dependencies"


if __name__ == "__main__":
    print("Testing PRD structure...")
    test_prd_structure()
    test_original_vs_new_structure()
    print("\nAll tests passed! ✓")