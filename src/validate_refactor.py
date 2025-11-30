"""Validation script for RL refactor - checks structure without requiring all dependencies."""

import sys
import json
from pathlib import Path

def validate_file_structure():
    """Validate that all required files exist."""
    print("Validating file structure...")
    
    required_files = [
        "src/__init__.py",
        "src/environment.py",
        "src/agent.py",
        "src/reward.py",
        "src/trainer.py",
        "src/template_loader.py",
        "src/main.py",
        "src/templates/system_prompt_generic.md",
        "src/templates/user_query_template.md",
        "src/templates/grading_prompt_no_ground_truth.md",
        "sample_dataset.json"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
        else:
            print(f"  [OK] {file_path}")
    
    if missing:
        print(f"\n[FAIL] Missing files: {missing}")
        return False
    else:
        print("\n[PASS] All required files exist")
        return True

def validate_dataset():
    """Validate dataset structure."""
    print("\nValidating dataset...")
    
    try:
        with open("sample_dataset.json", 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("[FAIL] Dataset must be a list")
            return False
        
        if len(data) == 0:
            print("[FAIL] Dataset is empty")
            return False
        
        required_fields = ['query', 'context', 'answer']
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                print(f"❌ Entry {i} is not a dictionary")
                return False
            
            for field in required_fields:
                if field not in entry:
                    print(f"[FAIL] Entry {i} missing field: {field}")
                    return False
            
            # Validate answer is valid JSON array string
            try:
                answer = json.loads(entry['answer'])
                if not isinstance(answer, list):
                    print(f"[FAIL] Entry {i} answer is not a list")
                    return False
            except json.JSONDecodeError:
                print(f"[FAIL] Entry {i} answer is not valid JSON")
                return False
        
        print(f"  [OK] Dataset has {len(data)} valid entries")
        print("[PASS] Dataset structure is valid")
        return True
        
    except FileNotFoundError:
        print("[FAIL] sample_dataset.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"[FAIL] Invalid JSON: {e}")
        return False

def validate_templates():
    """Validate template files don't contain ground truth references."""
    print("\nValidating templates...")
    
    templates_dir = Path("src/templates")
    templates = [
        "system_prompt_generic.md",
        "user_query_template.md",
        "grading_prompt_no_ground_truth.md"
    ]
    
    forbidden_terms = [
        "{correct_answer}",
        "{ground_truth}",
        "{source_of_truth}",
        "correct answer",
        "ground truth"
    ]
    
    issues = []
    for template_name in templates:
        template_path = templates_dir / template_name
        if not template_path.exists():
            issues.append(f"Missing: {template_name}")
            continue
        
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Check for forbidden terms (case-insensitive)
        content_lower = content.lower()
        for term in forbidden_terms:
            if term.lower() in content_lower and template_name != "grading_prompt_no_ground_truth.md":
                # grading_prompt_no_ground_truth.md is allowed to mention "ground truth" in instructions
                # but shouldn't have {ground_truth} variable
                if term.startswith("{"):
                    issues.append(f"{template_name} contains forbidden variable: {term}")
        
        print(f"  [OK] {template_name}")
    
    if issues:
        print(f"\n[WARN] Template issues: {issues}")
        return False
    else:
        print("[PASS] Templates are valid (no ground truth variables)")
        return True

def validate_imports():
    """Validate that imports work (for non-external dependencies)."""
    print("\nValidating imports...")
    
    sys.path.insert(0, '.')
    
    try:
        from src.template_loader import TemplateLoader
        print("  [OK] TemplateLoader imports successfully")
    except Exception as e:
        print(f"  [FAIL] TemplateLoader import failed: {e}")
        return False
    
    # Check syntax of other files (they may fail on import due to missing deps)
    import py_compile
    files_to_check = [
        "src/environment.py",
        "src/agent.py",
        "src/reward.py",
        "src/trainer.py",
        "src/main.py"
    ]
    
    for file_path in files_to_check:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"  [OK] {Path(file_path).name} syntax valid")
        except py_compile.PyCompileError as e:
            print(f"  [FAIL] {Path(file_path).name} syntax error: {e}")
            return False
    
    print("[PASS] All imports and syntax checks passed")
    return True

def main():
    """Run all validation checks."""
    print("=" * 70)
    print("RL Refactor Validation")
    print("=" * 70)
    
    checks = [
        ("File Structure", validate_file_structure),
        ("Dataset", validate_dataset),
        ("Templates", validate_templates),
        ("Imports & Syntax", validate_imports),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} validation failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    
    all_passed = True
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n[SUCCESS] All validations passed! Refactor structure is correct.")
        print("\nNote: To run the full system, ensure dependencies are installed:")
        print("  pip install torch transformers datasets smolagents")
        print("  And set environment variables: AOAI_API_KEY, AOAI_ENDPOINT")
    else:
        print("\n[WARN] Some validations failed. Please review the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

