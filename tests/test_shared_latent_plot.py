"""The shared helper preserves raw trajectories, frame limits and selections."""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('shared_plotting', Path(__file__).resolve().parents[1]/'src/lss/plotting.py')
plotting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotting)


def test_raw_values_and_frame_cutoff():
    table = pd.DataFrame({'source': ['node']*4, 'trajectory_id': [47]*4,
                          'frame': [3, 0, 2, 1], 'z0': [13., 10., 12., 11.],
                          'z1': [-3., 0., -2., -1.]})
    fig, axes = plotting.plot_latent_coordinates(table, {'node': [47]}, {'node': 'Node'},
        {'node': 'blue'}, frame_stop=3, title='Raw')
    np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), [0, 1, 2])
    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), [10., 11., 12.])
    np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), [0., -1., -2.])
    plt.close(fig)


def test_missing_selected_network_fails():
    table = pd.DataFrame(columns=['source', 'trajectory_id', 'frame', 'z0', 'z1'])
    with pytest.raises(ValueError, match='Missing or duplicate'):
        plotting.plot_latent_coordinates(table, {'node': [47]}, {'node': 'Node'},
            {'node': 'blue'}, frame_stop=100, title='Raw')
    plt.close('all')
