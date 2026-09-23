"""Compatibility import for the archived local-dynamics study."""

import sys
from ..past_experiments.dynamics import local_dynamics_propagator as _implementation
sys.modules[__name__] = _implementation
