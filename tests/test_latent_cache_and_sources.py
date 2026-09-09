import pandas as pd
import pytest

from lss.dynamics.analysis import label_evaluation_sources
from lss.dynamics.experiment import (
    DEFAULT_PROPAGATOR_CHECKPOINT_METRIC,
    _expand_component_configs,
    latent_experiment_cache_key,
)


def test_nested_component_configs_expand_without_notebook_prefixes():
    expanded = _expand_component_configs(
        {
            "ae_config": {
                "model": "attention",
                "target_mode": "normalized_delta",
                "max_epochs": 12,
                "lr": 1e-4,
            },
            "propagator_config": {
                "model": "delta_mlp",
                "loss": "delta",
                "max_epochs": 8,
                "context_dim": 16,
                "fixed_observed_frames": (1, 5),
            },
        }
    )
    assert expanded["autoencoder_model"] == "attention"
    assert expanded["ae_target_mode"] == "normalized_delta"
    assert expanded["ae_max_epochs"] == 12
    assert expanded["ae_lr"] == 1e-4
    assert expanded["propagator_model"] == "delta_mlp"
    assert expanded["propagator_loss"] == "delta"
    assert expanded["dyn_max_epochs"] == 8
    assert expanded["graph_context_dim"] == 16
    assert expanded["fixed_observed_frames"] == (1, 5)


@pytest.mark.parametrize(
    ("component", "legacy_key"),
    [
        ("ae_config", "ae_max_epochs"),
        ("ae_config", "autoencoder_model"),
        ("propagator_config", "propagator_loss"),
        ("propagator_config", "dyn_max_epochs"),
    ],
)
def test_nested_component_configs_reject_prefixed_keys(component, legacy_key):
    with pytest.raises(ValueError, match="without a component prefix"):
        _expand_component_configs({component: {legacy_key: 1}})


def test_source_labels_come_from_evaluation_rows_not_simulation_indices():
    rows = pd.DataFrame(
        {
            "sim_idx": [0, 0, 1],
            "source": ["reid", "lj_noisy", "reid"],
        }
    )

    labeled = label_evaluation_sources(
        rows,
        {"reid": "Reid", "lj_noisy": "noisy LJ"},
    )

    assert labeled["source_family"].tolist() == ["Reid", "noisy LJ", "Reid"]
    assert "source_family" not in rows


def test_source_labeling_rejects_missing_authoritative_metadata():
    with pytest.raises(KeyError, match="authoritative 'source' metadata"):
        label_evaluation_sources(pd.DataFrame({"sim_idx": [0]}), {})


def test_cache_fingerprint_changes_with_model_config_but_not_runtime_path():
    source = {"path": "data.pt", "dataset_name": "example"}
    base = latent_experiment_cache_key(
        source,
        {
            "latent_dim": 2,
            "cache_path": "first.pt",
            "force_train": False,
            "force_train_autoencoder": False,
        },
    )
    same_model = latent_experiment_cache_key(
        source,
        {
            "latent_dim": 2,
            "cache_path": "second.pt",
            "force_train": True,
            "force_train_autoencoder": True,
        },
    )
    changed_model = latent_experiment_cache_key(
        source,
        {"latent_dim": 8, "cache_path": "first.pt", "force_train": False},
    )

    assert base == same_model
    assert base != changed_model


def test_default_propagator_selection_is_state_loss_and_distinct_from_response_recipe():
    """Default caches represent state-loss selection without epoch rollouts."""

    assert DEFAULT_PROPAGATOR_CHECKPOINT_METRIC == "val_loss"
    default_metric = {}.get(
        "propagator_checkpoint_metric", DEFAULT_PROPAGATOR_CHECKPOINT_METRIC
    )
    rollout_callback_enabled = bool(
        {}.get("propagator_rollout_eval_every_epoch", False)
    ) or default_metric not in {None, "val_loss"}
    assert not rollout_callback_enabled

    source = {"path": "data.pt", "dataset_name": "example"}
    state_loss_key = latent_experiment_cache_key(source, {"latent_dim": 2})
    response_selected_key = latent_experiment_cache_key(
        source,
        {"latent_dim": 2, "propagator_checkpoint_metric": "val_rollout_p_ratio_r2"},
    )
    assert state_loss_key != response_selected_key


@pytest.mark.parametrize('matches', [False, True])
def test_cache_loading_checks_recipe_by_default(tmp_path, monkeypatch, matches):
    import torch
    import lss.dynamics.experiment as experiment
    path=tmp_path/'ae.pt'
    source={'path':'unused.pt','dataset_name':'example'}
    cfg={'latent_dim':2,'cache_path':str(path),'should_rollout':False,'should_train_propagator':False}
    key=experiment.latent_experiment_cache_key(source,cfg)
    torch.save({'cache_key':key if matches else 'different-recipe'},path)
    calls=[]
    monkeypatch.setattr(experiment,'_load_ae_cache',lambda *a,**kw:calls.append('load') or {})
    monkeypatch.setattr(experiment,'train_latent_autoencoder_experiment',lambda *a,**kw:calls.append('train') or {})
    monkeypatch.setattr(experiment,'_save_ae_cache',lambda *a,**kw:None)
    experiment.run_latent_experiment(source,cfg,device='cpu')
    assert calls==(['load'] if matches else ['train'])
