"""Model variants for the pair-level ablation experiments."""
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn

from model import trainer


VARIANTS = {
    'full': ('Full', 'full_global'),
    'no-diff': ('w/o Diff', 'no_diffusion_global'),
    'no-trans': ('w/o Trans', 'no_transformer_global'),
    'single-strength': ('Single-rho', 'single_strength'),
    'no-topology': ('w/o Topo', 'no_topology'),
    'concat': ('Concat', 'concatenation'),
    'uniform': ('Uniform Fusion', 'fixed_fusion'),
}


def variant_recipe(recipe, variant):
    """Keep the full training recipe and apply the selected intervention."""
    _, arm = VARIANTS[variant]
    config = dict(recipe['config'], arm=arm, variant=arm)
    if variant == 'single-strength':
        config.update(full_alphas=None, single_alpha=.8)
    return dict(recipe, config=config)


def branch_widths(hidden):
    width, remainder = divmod(hidden, 3)
    return [width + int(i < remainder) for i in range(3)]


class WidthMatchedConcatenation(nn.Module):
    def __init__(self, projections, hidden):
        super().__init__()
        self.proj = projections
        start = 0
        for layer, width in zip(self.proj, branch_widths(hidden)):
            # Slice the initialized Full projection, preserving the parent RNG state.
            end = start + width
            layer.weight = nn.Parameter(layer.weight.detach()[start:end].clone())
            layer.bias = nn.Parameter(layer.bias.detach()[start:end].clone())
            layer.out_features = width
            start = end

    def forward(self, diff_feats):
        return torch.cat([layer(x).relu() for layer, x in zip(self.proj, diff_feats)], dim=-1)


def empty_graph(positive_indices, node_count):
    edges = positive_indices.new_empty((2, 0))
    weights = torch.empty(0, dtype=torch.float32, device=edges.device)
    return torch.sparse_coo_tensor(edges, weights, (node_count, node_count)).coalesce(), edges


def make_trainer(variant):
    """Build the requested variant with the same training and inference routines."""
    _, arm = VARIANTS[variant]
    if variant not in ('no-topology', 'concat', 'uniform'):
        return trainer

    spec = importlib.util.spec_from_file_location(f'targon_{arm}', trainer.__file__)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    parent_model = engine.SingleBackbone
    no_topology = variant == 'no-topology'
    concat = variant == 'concat'
    engine.VARIANT_SEMANTICS[arm] = {
        'diffusion': 'Three copies of the initial features' if no_topology else 'Full branch coefficients',
        'transformer': 'Full blocks on an empty edge set' if no_topology else 'Full blocks on the training context',
        'fusion': {
            'no-topology': 'Learned global weights',
            'concat': 'Direct concatenation at fixed total width',
            'uniform': 'Global softmax of three frozen zero logits',
        }[variant],
    }

    class VariantModel(parent_model):
        def __init__(self, features, visible_nodes, config):
            super().__init__(features, visible_nodes, dict(config, arm='full_global', variant='full_global'))
            self.config = dict(config)
            self.variant = arm
            if no_topology:
                self.diffusion_enabled = False
            elif concat:
                self.fusion_mode = arm
                self.encoder.fusion = WidthMatchedConcatenation(self.encoder.fusion.proj, config['hidden'])
            else:
                self.encoder.fusion.gamma.requires_grad_(False)

        def encode(self, adjacency, edge_index, augment=False):
            if no_topology:
                adjacency = None
                edge_index = self.features.new_empty((2, 0), dtype=torch.long)
            return super().encode(adjacency, edge_index, augment=augment)

    engine.SingleBackbone = VariantModel
    if no_topology:
        engine.make_graph = empty_graph
    original_write = engine.write_json

    def write_metadata(path, metadata):
        path = Path(path)
        if path.name == 'model_description.json':
            metadata.update(interaction_topology_used=not no_topology,
                            gt_neighbor_messages_enabled=not no_topology,
                            fusion_weights_trainable=no_topology,
                            effective_fusion_mode=arm if concat else 'learned_global' if no_topology else 'fixed_uniform')
            if concat:
                hidden = json.loads((path.parent / 'config.json').read_text(encoding='utf8'))['hidden']
                metadata.update(fusion_mode=arm, branch_output_widths=branch_widths(hidden), fused_width=hidden,
                                post_concat_projection=False)
        original_write(path, metadata)

    engine.write_json = write_metadata
    return engine
