#!/usr/bin/env python
"""MemoryMesh validation runner — subprocess-based without relying on uv."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.absolute()
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"


def run_command(cmd: list[str], description: str) -> tuple[int, str]:
    """Run a command and return (exit_code, output)."""
    print(f"[*] {description}…", flush=True)
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = result.stdout + result.stderr
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, "Command timed out after 600 seconds"
    except Exception as e:
        return 1, f"Error running command: {e}"


def parse_pytest_output(output: str) -> dict:
    """Parse pytest summary line to extract pass/skip/fail counts."""
    # Look for pattern like "10 passed, 2 skipped in 5.23s"
    match = re.search(
        r"(\d+)\s+passed(?:,\s+(\d+)\s+skipped)?(?:,\s+(\d+)\s+failed)?",
        output,
    )
    if match:
        passed = int(match.group(1))
        skipped = int(match.group(2) or 0)
        failed = int(match.group(3) or 0)
        return {"passed": passed, "skipped": skipped, "failed": failed}
    return {"passed": 0, "skipped": 0, "failed": 0}


def check_benchmark_imports() -> bool:
    """Check that all three benchmark files can be imported."""
    modules = [
        "benchmarks.bench_indexing",
        "benchmarks.bench_search_latency",
        "benchmarks.bench_embedding_models",
    ]
    all_ok = True
    for module in modules:
        print(f"  • Testing import: {module}…", flush=True)
        code, output = run_command(
            [str(VENV_PYTHON), "-c", f"import {module}"],
            f"Import {module}",
        )
        if code != 0:
            print(f"    ERROR: {output}")
            all_ok = False
        else:
            print(f"    ✓ {module}")
    return all_ok


def main() -> int:
    """Run the full validation suite."""
    print("=" * 50)
    print("MemoryMesh — Test & Lint Validation")
    print("=" * 50)
    print()

    # Check that venv python exists
    if not VENV_PYTHON.exists():
        print(f"ERROR: venv python not found at {VENV_PYTHON}")
        return 1

    print(f"Using Python: {VENV_PYTHON}")
    print()

    # Step 1: Show versions
    print("[1/5] Checking environment…")
    code, output = run_command([str(VENV_PYTHON), "--version"], "Python version")
    if code == 0:
        print(f"  ✓ Python: {output.strip()}")
    else:
        print("  ERROR: Could not get Python version")
        return 1

    # Check for ruff
    code, ruff_ver = run_command(
        [str(VENV_PYTHON), "-m", "ruff", "--version"],
        "Ruff version",
    )
    if code == 0:
        print(f"  ✓ Ruff: {ruff_ver.strip()}")
    else:
        print("  ERROR: ruff not found")
        return 1

    # Check for pytest
    code, pytest_ver = run_command(
        [str(VENV_PYTHON), "-m", "pytest", "--version"],
        "Pytest version",
    )
    if code == 0:
        print(f"  ✓ Pytest: {pytest_ver.strip()}")
    else:
        print("  ERROR: pytest not found")
        return 1

    print()

    # Step 2: Ruff check
    print("[2/5] Running ruff check…")
    code, output = run_command(
        [str(VENV_PYTHON), "-m", "ruff", "check", "."],
        "ruff check .",
    )
    if code != 0:
        print(f"  ERROR: ruff check failed (exit code: {code})")
        print(output)
        return 1
    print("  ✓ ruff check: 0 errors")
    print()

    # Step 3: Pytest
    print("[3/5] Running pytest…")
    code, output = run_command(
        [str(VENV_PYTHON), "-m", "pytest", "tests/", "-q", "--tb=short"],
        "pytest tests/ -q --tb=short",
    )
    pytest_stats = parse_pytest_output(output)
    if code != 0 and code != 5:
        print(f"  ERROR: pytest failed (exit code: {code})")
        print(output)
        return 1
    print(
        f"  ✓ Tests: {pytest_stats['passed']} passed, "
        f"{pytest_stats['skipped']} skipped, {pytest_stats['failed']} failed"
    )
    print()

    # Step 4: Benchmark imports
    print("[4/5] Checking benchmark imports…")
    if not check_benchmark_imports():
        print("  ERROR: One or more benchmark files failed to import")
        return 1
    print("  ✓ All benchmark files importable")
    print()

    # Summary
    print("=" * 50)
    print("Validation Summary")
    print("=" * 50)
    print("✓ ruff check: 0 errors")
    print(
        f"✓ pytest: {pytest_stats['passed']} passed, "
        f"{pytest_stats['skipped']} skipped, {pytest_stats['failed']} failed"
    )
    print("✓ benchmarks: 3/3 files importable")
    print()
    print("All checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
