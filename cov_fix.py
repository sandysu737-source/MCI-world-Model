"""
Early-load pytest plugin to fix coverage.py 7.x + numpy 2.x C-extension conflict.

Registered via pyproject.toml ``[tool.pytest.ini_options] addopts = "-p cov_fix"``.

Fixes: when ``--cov=<submodule>`` is passed (e.g. ``--cov=mci_world_model._sys.causal``),
coverage.py's import tracing causes numpy's ``_multiarray_umath`` C extension
to be loaded twice, raising ``ImportError: cannot load module more than once``.

Workaround: monkey-patch ``coverage.Coverage.__init__`` to rewrite
``source=['mci_world_model.<sub>']`` → ``source=['mci_world_model']``.
"""

from __future__ import annotations

import coverage as _cov_mod

_orig_init = _cov_mod.Coverage.__init__


def _patched_init(self, *args, **kwargs):
    """Rewrite source kwarg: submodule → top-level package."""
    source = kwargs.get("source")
    if source:
        kwargs["source"] = [
            "mci_world_model" if s.startswith("mci_world_model.") else s
            for s in source
        ]
    return _orig_init(self, *args, **kwargs)


_cov_mod.Coverage.__init__ = _patched_init
