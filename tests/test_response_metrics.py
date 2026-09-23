from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[1]
METRICS_PATH = PROJECT_ROOT / 'src/lss/dynamics/response_metrics.py'


def _load_response_metrics():
    spec = importlib.util.spec_from_file_location('response_metrics_test', METRICS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


response_metrics = _load_response_metrics()
scalar_prediction_metrics = response_metrics.scalar_prediction_metrics


def test_scalar_prediction_metrics_perfect_predictions() -> None:
    score = scalar_prediction_metrics([1.0, 2.0, 4.0], [1.0, 2.0, 4.0])

    assert score['p_ratio_r2'] == 1.0
    assert score['mae'] == 0.0
    assert score['target_variance'] == pytest.approx(14.0 / 9.0)
    assert {key: score[key] for key in ('total', 'valid', 'true_valid',
                                        'true_nonfinite', 'pred_nonfinite')} == {
        'total': 3, 'valid': 3, 'true_valid': 3,
        'true_nonfinite': 0, 'pred_nonfinite': 0,
    }


def test_scalar_prediction_metrics_uses_target_centered_r2_for_biased_predictions() -> None:
    score = scalar_prediction_metrics([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])

    assert score['p_ratio_r2'] == pytest.approx(-0.5)
    assert score['mae'] == pytest.approx(1.0)
    assert score['target_variance'] == pytest.approx(2.0 / 3.0)


def test_scalar_prediction_metrics_tracks_nonfinite_values_without_scoring_them() -> None:
    score = scalar_prediction_metrics(
        [1.0, np.nan, np.inf, 4.0], [1.0, 2.0, 3.0, np.nan]
    )

    assert score['total'] == 4
    assert score['valid'] == 1
    assert score['true_valid'] == 2
    assert score['true_nonfinite'] == 2
    assert score['pred_nonfinite'] == 1
    assert score['p_ratio_r2'] != score['p_ratio_r2']
    assert score['mae'] == 0.0
    assert score['target_variance'] == 0.0


def test_scalar_prediction_metrics_constant_targets_are_undefined() -> None:
    score = scalar_prediction_metrics([2.0, 2.0, 2.0], [2.0, 1.0, 3.0])

    assert score['total'] == score['valid'] == score['true_valid'] == 3
    assert np.isnan(score['p_ratio_r2'])
    assert score['mae'] == pytest.approx(2.0 / 3.0)
    assert score['target_variance'] == 0.0


def test_diagnostic_summarize_preserves_legacy_columns_and_values(monkeypatch) -> None:
    graph_utils = types.ModuleType('graph_utils')
    graph_utils.calc_p_ratio_rollout_sides = lambda *args: None
    graph_utils.directional_side_indices_from_box = lambda *args: None
    monkeypatch.setitem(sys.modules, 'graph_utils', graph_utils)
    lss = types.ModuleType('lss')
    dynamics = types.ModuleType('lss.dynamics')
    monkeypatch.setitem(sys.modules, 'lss', lss)
    monkeypatch.setitem(sys.modules, 'lss.dynamics', dynamics)
    monkeypatch.setitem(sys.modules, 'lss.dynamics.response_metrics', response_metrics)
    module_path = PROJECT_ROOT / 'hpc/experiments/runners/latent_diagnostic_metrics.py'
    spec = importlib.util.spec_from_file_location('latent_diagnostic_metrics_test', module_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows = pd.DataFrame({
        'source': ['a', 'a', 'b', 'b'], 'true_p_ratio': [1.0, 3.0, 2.0, np.nan],
        'pred_p_ratio': [2.0, 2.0, 2.0, 3.0], 'position_mse': [1.0, 3.0, 5.0, 7.0],
    })

    result = runner.summarize(rows, ['source']).set_index('source')

    assert list(result.columns) == [
        'total', 'valid', 'true_valid', 'p_ratio_r2', 'p_ratio_mae',
        'target_variance', 'position_mse',
    ]
    assert result.loc['a', 'total'] == 2
    assert result.loc['a', 'valid'] == result.loc['a', 'true_valid'] == 2
    assert result.loc['a', 'p_ratio_r2'] == 0.0
    assert result.loc['a', 'p_ratio_mae'] == 1.0
    assert result.loc['a', 'target_variance'] == 1.0
    assert result.loc['a', 'position_mse'] == 2.0
    assert result.loc['b', 'total'] == 2
    assert result.loc['b', 'valid'] == 1
    assert result.loc['b', 'true_valid'] == 1
    assert np.isnan(result.loc['b', 'p_ratio_r2'])
