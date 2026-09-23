"""Compatibility import for the archived factorized simulator experiment."""

import sys
from ..past_experiments.dynamics import factorized_simulator as _implementation
sys.modules[__name__] = _implementation
