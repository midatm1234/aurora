"""Opt-in workflow tests coexist with the fork's older Python/Windows CI matrix."""
import importlib.util
import sys

import pytest

if sys.version_info < (3, 12) or sys.platform != "linux":
    pytest.skip("Portable workflow suite targets Python 3.12+ on Linux; legacy Aurora tests remain available",
                allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("Optional workflow dependencies are absent; use the documented workflow bootstrap",
                allow_module_level=True)
