"""MCI World Model adapters — integration bridges to sibling projects.

This package provides zero-dependency adapters that bridge MCI World Model
with su-memory-sdk (memory engine) and mci-huan (clinical harness).
All adapters use a probe-and-import pattern: they work when the target
package is installed and degrade gracefully when it is not.
"""

from __future__ import annotations

__all__: list[str] = []
