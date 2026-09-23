"""Compatibility import for the archived static-autoencoder study."""

import sys
from ..past_experiments.dynamics import static_autoencoder as _implementation
sys.modules[__name__] = _implementation
