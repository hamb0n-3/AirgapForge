#!/usr/bin/env python3
"""Shared helpers for the pipeline engine and preflight. Manifest loading, path and
interpreter resolution, and hardware probes. Domain-agnostic."""
import os
import shutil
import subprocess
import sys

import yaml


def load_manifest(path):
    with open(path) as f:
        return yaml.safe_load(f)


def expand(path):
    return os.path.expanduser(str(path))


def populated(path):
    return os.path.isdir(path) and any(not n.startswith(".") for n in os.listdir(path))


def resolve_root(manifest):
    env_name = manifest.get("root_env", "PIPELINE_ROOT")
    return expand(os.environ.get(env_name) or manifest["repo_root"])


def resolve_python(manifest):
    env_name = manifest.get("venv_env", "VENV_PYTHON")
    candidates = [os.environ.get(env_name), manifest.get("venv_python"), sys.executable]
    for cand in candidates:
        if not cand:
            continue
        path = expand(cand)
        if os.path.exists(path) or cand == sys.executable:
            return path
    return sys.executable


def total_ram_gb():
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1e9
    except Exception:
        try:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        except Exception:
            return 0.0


def free_disk_gb(path):
    target = path if os.path.exists(path) else os.path.expanduser("~")
    return shutil.disk_usage(target).free / 1e9


def can_import(py, module):
    try:
        return subprocess.run([py, "-c", f"import {module}"],
                              capture_output=True).returncode == 0
    except Exception:
        return False
