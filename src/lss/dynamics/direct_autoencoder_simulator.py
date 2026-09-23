"""Compatibility import for the archived direct-autoencoder simulator study."""

import sys
from ..past_experiments.dynamics import direct_autoencoder_simulator as _implementation
sys.modules[__name__] = _implementation
