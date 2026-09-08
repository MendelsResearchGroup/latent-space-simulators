# Retired test modules

`test_latent_strain_loss.py.disabled` tested removed strain-to-p-ratio training helpers. The active project keeps strain and p-ratio as post-fit diagnostics and no longer exposes those loss APIs.

`test_network_design.py.disabled` imports the separate `notebooks/network_design/design.py` pipeline. That optional engineering workflow is intentionally excluded from this standalone simulator package.

The `.py.disabled` suffix keeps these historical expectations out of normal pytest discovery. They remain as provenance, not skipped active tests.
