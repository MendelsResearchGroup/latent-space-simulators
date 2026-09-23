"""Compatibility import for archived trajectory-propagator studies."""

import sys
from ..past_experiments.dynamics import trajectory_propagator as _implementation
sys.modules[__name__] = _implementation
