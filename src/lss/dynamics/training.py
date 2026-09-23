"""Stable compatibility entry point for established training recipes.

The configurable implementation is isolated in :mod:`lss.dynamics._compat`.
New model construction should use :mod:`lss.dynamics.canonical`.
"""

import sys

from ._compat import training as _implementation

sys.modules[__name__] = _implementation
