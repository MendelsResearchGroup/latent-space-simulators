#!/usr/bin/env python3
"""Read-only forty-original periodic-planarity audit; performs no optimization or physics."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch

from lss.engineering.geometry import boundary_constraints, valid_geometry
from lss.data import normalize_dataset_edges, normalize_trajectory_to_reference_box, tag_simulation_source
from lss.engineering.planarity import PlanarConstraints


def audit(recipe: dict, row: dict, raw) -> dict:
    key = 'reid' if row['cohort'] == 'same10_reid' else 'thermal_arm_a'
    physical = deepcopy(raw[row['index']][0])
    trajectory = [deepcopy(raw[row['index']][0])]
    normalize_dataset_edges([trajectory], edge_multiplicity=1, edge_vector_dim=2)
    normalize_trajectory_to_reference_box(trajectory, pos_dim=2)
    tag_simulation_source(trajectory, row['source'])
    reference = trajectory[0]
    position0 = reference.reference_context_positions[:, :2].detach()
    planar = PlanarConstraints(position0.numpy(), reference.edge_index.numpy(),
                               [physical.box.x, physical.box.y],
                               recipe[key]['optimization'].get('spring_clearance_fraction', .05))
    sides, _, _, _ = boundary_constraints(reference, physical)
    valid = valid_geometry(reference, physical, torch.zeros_like(reference.edge_attr[:, 3]),
                           torch.zeros_like(position0), sides, planar, position0)
    return {**row, 'pristine_valid': bool(valid), 'periodic_pristine_violating_pairs': planar.violating_pairs(position0.numpy()).tolist(),
            'planar_constraints': planar.summary(), 'audit_kind': 'read_only_pristine_zero_edit'}


def main(output: Path, offset: int, limit: int | None) -> None:
    output = Path(output)
    recipe = json.loads((output / 'cohort_and_recipe.json').read_text())
    declared = recipe['rows'][offset:] if limit is None else recipe['rows'][offset:offset + limit]
    data = {key: torch.load(value['dataset'], weights_only=False, map_location='cpu', mmap=True)
            for key, value in ((key, recipe[key]) for key in ('reid', 'thermal_arm_a'))}
    rows = [audit(recipe, row, data['reid' if row['cohort'] == 'same10_reid' else 'thermal_arm_a']) for row in declared]
    result = {'expected': 40, 'observed': len(rows), 'valid': sum(row['pristine_valid'] for row in rows),
              'invalid': sum(not row['pristine_valid'] for row in rows), 'rows': rows,
              'no_optimization': True, 'no_physics': True}
    name = 'pristine_periodic_audit.json' if offset == 0 and limit is None else f'pristine_periodic_audit_{offset}_{len(rows)}.json'
    (output / name).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    main(args.output, args.offset, args.limit)
