"""Compatibility import for the archived path-propagator study."""

import sys
from ..past_experiments.dynamics import path_propagator as _implementation
sys.modules[__name__] = _implementation
