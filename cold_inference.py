"""Paired Original and Restored inference for the cold-drug Full model."""
import json
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def paired_predictions(trainer, folder, train, test, ids):
    """Score the same holdout twice using a saved Full checkpoint and training graph.

    Returns (original, restored), in test row order. Only held-out drug rows
    change before the three branch projections. No parameters are updated.
    """
    folder = Path(folder)
    ids = np.asarray(ids, dtype=str)
    np.testing.assert_array_equal(ids, np.load(folder / 'node_ids.npy', allow_pickle=False))
    training_nodes = set(train.drug) | set(train.target)
    heldout_drugs = sorted(set(test.drug))
    if not heldout_drugs or not set(heldout_drugs).isdisjoint(training_nodes | set(test.target)):
        raise ValueError('Expected held-out drugs absent from all training pairs and target IDs.')
    if not set(train.canonical_drug).isdisjoint(test.canonical_drug):
        raise ValueError('A canonical drug occurs in both the training and held-out partitions.')
    audit = json.loads((folder / 'training_audit.json').read_text(encoding='utf-8'))
    if trainer.train_pair_hash(train) != audit['training_pair_sha256']:
        raise ValueError('The specified training pairs differ from those used for this checkpoint.')
    features = np.load(folder / 'features.npy', allow_pickle=False)
    visible = np.load(folder / 'train_visible_nodes.npy', allow_pickle=False)
    np.testing.assert_array_equal(np.isin(ids, list(training_nodes)), visible)
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if config['arm'] != 'full_global' or config['id_dim'] != 32:
        raise ValueError('Expected the Full model with 32-dimensional masked ID embeddings.')
    device = trainer.DEFAULT_DEVICE
    model = trainer.SingleBackbone(features, visible, config).to(device)
    model.load_state_dict(trainer._load_torch(folder / 'model.pt', device))
    model.eval()
    lookup = {node: i for i, node in enumerate(ids)}
    pairs = trainer.frame_indices(test, lookup, device)
    positive = trainer.frame_indices(train.loc[train.label == 1], lookup, device)
    adjacency, edges = trainer.make_graph(positive, len(ids))
    cold = torch.tensor([lookup[node] for node in heldout_drugs], dtype=torch.long, device=device)
    if torch.isin(edges, cold).any() or model.visible_nodes[cold].any():
        raise ValueError('A held-out drug has an interaction edge or a visible ID embedding.')

    identity = model.identity.weight * model.visible_nodes[:, None]
    inputs = torch.cat([model.features, identity], dim=1)
    branches = trainer.original.multi_scale_diffusion(adjacency, inputs, list(model.full_alphas))
    restored = []
    for strength, branch in zip(model.full_alphas, branches):
        if not torch.equal(branch[cold], (1 - strength) * inputs[cold]):
            raise ValueError('The original diffusion differs from the isolated-drug formula.')
        updated = branch.clone()
        updated[cold] = inputs[cold]
        restored.append(updated)

    encoded_a = model.encoder(branches, edges)
    encoded_b = model.encoder(restored, edges)
    keep = torch.ones(len(ids), dtype=torch.bool, device=device)
    keep[cold] = False
    if not torch.equal(encoded_a[keep], encoded_b[keep]):
        raise ValueError('Restoration changed an embedding outside the held-out drug set.')
    probabilities = tuple(torch.sigmoid(model.decode(z, pairs)).cpu().numpy().astype(np.float64)
                          for z in (encoded_a, encoded_b))
    return probabilities
