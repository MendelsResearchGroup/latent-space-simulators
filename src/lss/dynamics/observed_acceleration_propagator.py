"""Compatibility import for the archived observed-acceleration study."""

import sys
from ..past_experiments.dynamics import observed_acceleration_propagator as _implementation
sys.modules[__name__] = _implementation
