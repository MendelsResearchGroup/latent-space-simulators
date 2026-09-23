"""Compatibility import for the archived observed-history study."""

import sys
from ..past_experiments.dynamics import observed_history_propagator as _implementation
sys.modules[__name__] = _implementation
