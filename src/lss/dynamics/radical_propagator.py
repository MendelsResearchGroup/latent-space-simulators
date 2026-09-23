"""Compatibility import for archived radical propagator studies."""

import sys
from ..past_experiments.dynamics import radical_propagator as _implementation
sys.modules[__name__] = _implementation
