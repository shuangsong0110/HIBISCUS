#!/usr/bin/env python3
"""Run every test_*.py in this folder with plain asserts (no pytest needed): python tests/run_tests.py"""
from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main() -> int:
    failures = 0
    total = 0
    for path in sorted(HERE.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            total += 1
            start = time.perf_counter()
            try:
                getattr(module, name)()
                print(f"PASS  {path.stem}.{name}  ({time.perf_counter() - start:.1f}s)")
            except Exception:
                failures += 1
                print(f"FAIL  {path.stem}.{name}")
                traceback.print_exc()
    print(f"{total - failures}/{total} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
