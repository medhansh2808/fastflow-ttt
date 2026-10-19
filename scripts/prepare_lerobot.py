"""Limit the private LeRobot checkout's eager policy registry to π0.5.

The upstream package imports GR00T and other optional policy implementations
even when only π0.5 is used. No model/data/processor implementation is changed.
"""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1] / "vendor/lerobot"
revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if revision != "8fff0fde7c79f23a93d845d1a50e985de01f8b8a":
    raise RuntimeError(f"Unexpected LeRobot revision: {revision}")
path = root / "src/lerobot/policies/__init__.py"
original = subprocess.check_output(["git", "-C", str(root), "show", "HEAD:src/lerobot/policies/__init__.py"], text=True)
header = original.split("from .act.", 1)[0]
path.write_text(header + 'from .pi05.configuration_pi05 import PI05Config as PI05Config\n\n__all__ = ["PI05Config"]\n')
print("π0.5-only registry prepared")
