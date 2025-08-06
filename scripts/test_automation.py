#!/usr/bin/env python3
"""
Test script for coverage automation.

This script simulates the coverage checking and fixing process
to validate our pre-push hooks and GitHub Actions work correctly.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"🔄 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result


def check_coverage():
    """Check current test coverage."""
    print("📊 Checking current test coverage...")
    result = run_command(
        "python -m pytest --cov=src --cov-report=term-missing", check=False
    )

    # Extract coverage percentage
    output_lines = result.stdout.split("\n")
    coverage_line = [line for line in output_lines if "TOTAL" in line and "%" in line]

    if coverage_line:
        # Extract percentage from line like "TOTAL    2213    846    380     21    54%"
        coverage_pct = float(coverage_line[0].split()[-1].rstrip("%"))
        print(f"Current coverage: {coverage_pct}%")
        return coverage_pct

    print("❌ Could not determine coverage percentage")
    return 0


def test_pre_push_hook():
    """Test the pre-push hook."""
    print("\n🔧 Testing pre-push hook...")

    # The pre-push hook should be installed
    hook_path = Path(".git/hooks/pre-push")
    if not hook_path.exists():
        print("❌ Pre-push hook not installed")
        return False

    print("✅ Pre-push hook is installed")

    # Test running the coverage check directly
    print("🔍 Testing coverage check logic...")
    coverage = check_coverage()

    if coverage >= 80:
        print(f"✅ Coverage ({coverage}%) meets requirement")
        return True
    else:
        print(f"❌ Coverage ({coverage}%) below requirement (80%)")
        print("💡 The pre-push hook would attempt to fix this automatically")
        return False


def test_github_actions():
    """Test GitHub Actions workflow."""
    print("\n🔧 Testing GitHub Actions workflow...")

    workflow_path = Path(".github/workflows/pr-auto-review-fix.yml")
    if not workflow_path.exists():
        print("❌ GitHub Actions workflow not found")
        return False

    print("✅ GitHub Actions workflow exists")

    # Validate workflow syntax
    result = run_command(
        'python -c "import yaml; '
        "yaml.safe_load(open('.github/workflows/pr-auto-review-fix.yml'))\"",
        check=False,
    )
    if result.returncode == 0:
        print("✅ Workflow YAML syntax is valid")
        return True
    else:
        print("❌ Workflow YAML syntax error:")
        print(result.stderr)
        return False


def main():
    """Run all automation tests."""
    print("🤖 Testing VOYAGER-Trader Automation")
    print("=" * 40)

    all_passed = True

    # Test pre-push hook
    if not test_pre_push_hook():
        all_passed = False

    # Test GitHub Actions
    if not test_github_actions():
        all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("✅ All automation tests passed!")
        print("\n📝 Summary of implemented automation:")
        print("  • Pre-push hook enforces 80% coverage requirement")
        print("  • Hook attempts to auto-fix coverage issues")
        print("  • GitHub Actions provides automated PR review")
        print("  • Workflow auto-fixes code quality issues")
        print("  • PR labeling based on quality metrics")
    else:
        print("❌ Some automation tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
