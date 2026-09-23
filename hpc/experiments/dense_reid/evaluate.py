"""Local entry point used by this study\'s collectors and frozen submit recipe.

Implementation lives in lss.dynamics.position_evaluation. Keep this small
entry point only while the study runner imports ``evaluate`` locally.
"""
from lss.dynamics.position_evaluation import (
    evaluate,
    evaluate_autoencoder,
    physical,
    _pref,
    _p_ratio,
)
