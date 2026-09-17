"""Atomic local artifact writing and environment metadata."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_yaml(path: str | Path, value: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(value, sort_keys=False))


def git_state(cwd: str | Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty_files": run("status", "--short").splitlines(),
    }


def environment_metadata(cwd: str | Path) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "PyYAML", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_device_availability": {
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
        },
        "packages": packages,
        "git": git_state(cwd),
    }
