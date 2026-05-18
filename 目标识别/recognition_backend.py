# -*- coding: utf-8 -*-
"""仅用于在本子工程内安全加载上级目录中的 Python 模块（避免与本地脚本同名冲突）。"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)


def ensure_project_root_on_path() -> None:
    if _PARENT not in sys.path:
        sys.path.insert(0, _PARENT)


def load_module_from_parent(filename: str, internal_name: str) -> Any:
    path = os.path.join(_PARENT, filename)
    spec = importlib.util.spec_from_file_location(internal_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[internal_name] = mod
    spec.loader.exec_module(mod)
    return mod
