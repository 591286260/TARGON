"""Model components and training utilities for graph-based DTI prediction."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL_SOURCE = HERE / 'reference_graph.py'
_spec = importlib.util.spec_from_file_location('strict_original_graph', ORIGINAL_SOURCE)
original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(original)
torch.set_num_threads(2)
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VARIANT_SEMANTICS = {
    'full': {
        'diffusion': '(1-alpha)X + alpha D^-1 A X, alpha=[0.1,0.3,0.6]; one propagation step per branch',
        'fusion': 'Node-dependent softmax(att(ReLU(proj_k(branch_k))))',
        'transformer': 'Original configured number of Graph Transformer layers',
    },
    'no_transformer': {
        'diffusion': 'Original single-step branches alpha=[0.1,0.3,0.6]',
        'fusion': 'Original node-dependent softmax attention',
        'transformer': 'Zero Graph Transformer layers; fusion and output LayerNorm retained',
    },
    'global_fusion': {
        'diffusion': 'Original single-step branches alpha=[0.1,0.3,0.6]',
        'fusion': 'sum_k softmax(gamma)_k ReLU(proj_k(branch_k)); gamma is a learned length-3 vector shared by all nodes; gamma starts at zero; node attention Linear removed',
        'transformer': 'Original configured number of Graph Transformer layers',
    },
    'no_transformer_global': {
        'diffusion': 'Original single-step branches alpha=[0.1,0.3,0.6]',
        'fusion': 'sum_k softmax(gamma)_k ReLU(proj_k(branch_k)); learned length-3 vector shared by all nodes; zero initialization; node attention Linear removed',
        'transformer': 'Zero Graph Transformer layers; fusion and output LayerNorm retained',
    },
    'no_graph': {
        'diffusion': 'Three inputs [X,X,X]; no adjacency propagation; three independent projections retained',
        'fusion': 'Original node-dependent softmax attention',
        'transformer': 'Zero Graph Transformer layers; fusion and output LayerNorm retained; encoder independent of adjacency and edge_index',
    },
    'no_gt_neighbors': {
        'diffusion': 'Original single-step branches alpha=[0.1,0.3,0.6], using the real current training-context adjacency',
        'fusion': 'Original node-dependent softmax attention',
        'transformer': 'Keep every configured original GT layer, FFN, residual, LayerNorm and out_proj bias; pass empty edge_index only to GT. Each layer: u=norm1(x+dropout(out_proj.bias)); output=norm2(u+FFN(u)). q/k/v parameters and out_proj.weight retained with zero task gradients; AdamW can still decay them.',
    },
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def pairset(frame):
    return set(map(tuple, frame[['drug', 'target']].astype(str).values))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def frame_indices(frame, idmap, device):
    return torch.tensor([(idmap[str(d)], idmap[str(t)])
                         for d, t in frame[['drug', 'target']].values],
                        dtype=torch.long, device=device)


def make_graph(positive_indices, node_count):
    """Build both directions from exactly the supplied positive pairs."""
    assert len(positive_indices) > 0
    edges = torch.cat([positive_indices, positive_indices.flip(1)], dim=0).T.contiguous()
    values = torch.ones(edges.shape[1], dtype=torch.float32, device=edges.device)
    adjacency = torch.sparse_coo_tensor(edges, values, (node_count, node_count)).coalesce()
    return adjacency, edges


class GlobalScaleFusion(nn.Module):
    """Keep the existing branch projections and learn node-shared weights.

    Reusing the initialized projections preserves their names and initial
    values; creating gamma does not consume additional random draws. The old
    attention module is not registered in this replacement.
    """
    def __init__(self, projections):
        super().__init__()
        self.proj = projections
        self.gamma = nn.Parameter(torch.zeros(len(projections)))

    def forward(self, diff_feats):
        stack = torch.stack([F.relu(proj(feat))
                             for proj, feat in zip(self.proj, diff_feats)], dim=1)
        return (stack * torch.softmax(self.gamma, dim=0)[None, :, None]).sum(dim=1)


class SingleBackbone(nn.Module):
    def __init__(self, features, visible_nodes, config):
        super().__init__()
        self.config = dict(config)
        self.variant = config.get('variant', 'full')
        if self.variant not in VARIANT_SEMANTICS:
            raise ValueError(f'Unknown ablation variant: {self.variant}')
        expected_fusion = ('global' if self.variant in ('global_fusion', 'no_transformer_global')
                           else 'nodeadaptive')
        self.fusion_mode = config.get('fusion_mode', expected_fusion)
        if self.fusion_mode != expected_fusion:
            raise ValueError(f'fusion_mode={self.fusion_mode} contradicts variant={self.variant}')
        if 'temperature' in config or 'branch_alphas' in config:
            raise ValueError('This diagnosis preserves the original branch alphas and attention mechanism; temperature/branch_alphas are not tunable in this version')
        features = np.asarray(features, dtype=np.float32)
        self.register_buffer('features', torch.as_tensor(features))
        self.register_buffer('visible_nodes', torch.as_tensor(visible_nodes, dtype=torch.bool))
        h = int(config.get('hidden', 128))
        identity_dim = int(config.get('id_dim', 0))
        self.identity = nn.Embedding(len(features), identity_dim) if identity_dim else None
        if self.identity is not None:
            nn.init.normal_(self.identity.weight, std=.1)
        self.encoder = original.DiffusionGraphTransformer(
            input_dim=features.shape[1] + identity_dim,
            hidden_dim=h,
            num_heads=int(config.get('heads', 4)),
            num_layers=(0 if self.variant in ('no_transformer', 'no_transformer_global', 'no_graph')
                        else int(config.get('layers', 2))),
            dropout=float(config.get('dropout', .1)),
            num_scales=3,
        )
        if self.fusion_mode == 'global':
            self.encoder.fusion = GlobalScaleFusion(self.encoder.fusion.proj)
        self.head = config.get('head', 'interaction')
        head_dropout = float(config.get('head_dropout', .2))
        if self.head == 'concat':
            self.decoder = nn.Sequential(nn.Linear(2*h, h), nn.ReLU(),
                                         nn.Dropout(head_dropout), nn.Linear(h, 1))
        else:
            assert self.head in ('interaction', 'bilinear', 'hybrid'), self.head
            self.drug_role = nn.Linear(h, h)
            self.target_role = nn.Linear(h, h)
            if self.head in ('interaction', 'hybrid'):
                self.decoder = nn.Sequential(nn.Linear(4*h, h), nn.ReLU(),
                                             nn.Dropout(head_dropout), nn.Linear(h, 1))
            if self.head in ('bilinear', 'hybrid'):
                self.output_bias = nn.Parameter(torch.zeros(()))

    def encode(self, adjacency, edge_index, augment=False):
        x = self.features
        if self.identity is not None:
            # Never train or expose a random identity row of a node that occurs
            # only in validation/test; IDs are an encoder input, never a score.
            identity = self.identity.weight * self.visible_nodes[:, None]
            x = torch.cat([x, identity], dim=1)
        diff = ([x, x, x] if self.variant == 'no_graph' else
                original.multi_scale_diffusion(adjacency, x, [.1, .3, .6]))
        if augment:
            diff = [original.augment_features(value, feat_drop=.1, noise_std=.01)
                    for value in diff]
        encoder_edges = (torch.empty((2, 0), dtype=torch.long, device=x.device)
                         if self.variant == 'no_gt_neighbors' else edge_index)
        return self.encoder(diff, encoder_edges)

    def decode(self, representation, pairs):
        d, t = representation[pairs[:, 0]], representation[pairs[:, 1]]
        if self.head == 'concat':
            return self.decoder(torch.cat([d, t], dim=1)).squeeze(1)
        u, v = self.drug_role(d), self.target_role(t)
        if self.head in ('interaction', 'hybrid'):
            interaction = self.decoder(torch.cat([u, v, u*v, torch.abs(u-v)], dim=1)).squeeze(1)
        if self.head in ('bilinear', 'hybrid'):
            bilinear = (u*v).sum(1) / math.sqrt(u.shape[1]) + self.output_bias
        if self.head == 'interaction':
            return interaction
        if self.head == 'bilinear':
            return bilinear
        return bilinear + interaction


def _metrics(y, probability):
    assert np.isfinite(probability).all()
    return dict(AUC=float(roc_auc_score(y, probability)),
                AUPR=float(average_precision_score(y, probability)))


def _validated_frame(frame):
    result = frame.copy()
    result['drug'] = result.drug.astype(str)
    result['target'] = result.target.astype(str)
    assert set(result.label.unique()) == {0, 1}
    assert len(result) == len(pairset(result)), 'Duplicate training/evaluation pair'
    return result


def train_candidate(train_df, val_df, node_ids, features, config, outdir, seed,
                    refit_epochs=None):
    """Train using train_df alone, optionally choose an epoch on val_df.

    refit_epochs=None: validation is mandatory, model saved at eligible best
    validation AUC/AP (eligible epochs >=min_epochs).
    refit_epochs=int: val_df must be None; fixed supervised epochs, no selection.
    No evaluation/test CSV is opened by this function.
    """
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    train = _validated_frame(train_df)
    validation = _validated_frame(val_df) if val_df is not None else None
    assert (refit_epochs is None) == (validation is not None)
    if validation is not None:
        assert pairset(train).isdisjoint(pairset(validation)), 'Validation pair entered training'
    node_ids = np.asarray(node_ids, dtype=str)
    assert len(node_ids) == len(set(node_ids))
    features = np.asarray(features, dtype=np.float32)
    assert features.ndim == 2 and len(features) == len(node_ids) and np.isfinite(features).all()
    idmap = {identifier: i for i, identifier in enumerate(node_ids)}
    train_nodes = set(train.drug) | set(train.target)
    assert train_nodes <= set(idmap)
    visible = np.asarray([identifier in train_nodes for identifier in node_ids])
    positive_frame = train.loc[train.label == 1, ['drug', 'target']]
    negative_frame = train.loc[train.label == 0, ['drug', 'target']]
    assert len(positive_frame) >= 2 and len(negative_frame) >= 2
    train.to_csv(folder / 'training_pairs.csv', index=False)
    positive_frame.to_csv(folder / 'train_graph.csv', index=False)
    if validation is not None:
        validation.to_csv(folder / 'validation_pairs.csv', index=False)
    np.save(folder / 'features.npy', features)
    np.save(folder / 'node_ids.npy', node_ids)
    np.save(folder / 'train_visible_nodes.npy', visible)
    write_json(folder / 'config.json', dict(config))

    set_seed(int(seed))
    model = SingleBackbone(features, visible, config).to(DEFAULT_DEVICE)
    inactive_neighbor_parameters = [name for name, _ in model.named_parameters()
        if model.variant == 'no_gt_neighbors' and name.startswith('encoder.layers.')
        and (any(f'.{projection}.' in name for projection in ('q_proj', 'k_proj', 'v_proj'))
             or name.endswith('.out_proj.weight'))]
    model_description = dict(
        variant=model.variant,
        variant_semantics=VARIANT_SEMANTICS[model.variant],
        total_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameter_count=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        configured_transformer_layers=int(config.get('layers', 2)),
        effective_transformer_layers=len(model.encoder.layers),
        fusion_mode=model.fusion_mode,
        feature_mode=config.get('feature_mode', 'caller_supplied'),
        feature_input_dimension=int(features.shape[1]),
        learned_identity_dimension=int(config.get('id_dim', 0)),
        decoder_head=model.head,
        branch_alphas=None if model.variant == 'no_graph' else [.1, .3, .6],
        gt_neighbor_messages_enabled=len(model.encoder.layers) > 0 and model.variant != 'no_gt_neighbors',
        retained_parameters_without_neighbor_task_gradient=inactive_neighbor_parameters,
        retained_parameters_without_neighbor_task_gradient_count=sum(parameter.numel()
            for name, parameter in model.named_parameters() if name in inactive_neighbor_parameters),
        inactive_neighbor_parameter_optimization=(
            'q/k/v and out_proj.weight task gradients are zero tensors for empty neighbors; parameters remain in AdamW and may undergo decoupled weight decay; out_proj.bias remains active'
            if model.variant == 'no_gt_neighbors' else None),
        contrastive_unchanged=True,
        pretrain_epochs=int(config.get('pretrain_epochs', 0)),
        joint_contrastive_weight=float(config.get('contrastive', 0.)),
    )
    write_json(folder / 'model_description.json', model_description)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('lr', .001)),
                                 weight_decay=float(config.get('weight_decay', .001)))
    positive = frame_indices(positive_frame, idmap, DEFAULT_DEVICE)
    negative = frame_indices(negative_frame, idmap, DEFAULT_DEVICE)
    full_adj, full_edges = make_graph(positive, len(node_ids))
    if validation is not None:
        val_ix = frame_indices(validation, idmap, DEFAULT_DEVICE)
    pretraining_history = []
    for epoch in range(1, int(config.get('pretrain_epochs', 0)) + 1):
        model.train()
        optimizer.zero_grad()
        z1 = model.encode(full_adj, full_edges, augment=True)
        z2 = model.encode(full_adj, full_edges, augment=True)
        loss = original.contrastive_loss(z1[model.visible_nodes], z2[model.visible_nodes])
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        pretraining_history.append(dict(epoch=epoch, loss=float(loss)))

    max_epochs = int(refit_epochs) if refit_epochs is not None else int(config.get('max_epochs', 160))
    check_every = int(config.get('check_every', 5))
    min_epochs = int(config.get('min_epochs', 30))
    patience = int(config.get('patience', 35))
    if validation is not None:
        assert max_epochs >= min_epochs > 0
    history, mask_records, pos_permutations, neg_permutations = [], [], [], []
    best = None
    best_state = None
    best_epoch = 0
    best_probability = None
    positive_np = positive.detach().cpu().numpy()
    negative_np = negative.detach().cpu().numpy()
    for epoch in range(1, max_epochs + 1):
        pos_perm = torch.randperm(len(positive), device=DEFAULT_DEVICE)
        neg_perm = torch.randperm(len(negative), device=DEFAULT_DEVICE)
        pos_order = pos_perm.cpu().numpy().astype(np.int32)
        neg_order = neg_perm.cpu().numpy().astype(np.int32)
        pos_permutations.append(pos_order)
        neg_permutations.append(neg_order)
        positive_parts = torch.tensor_split(pos_perm, 2)
        negative_parts = torch.tensor_split(neg_perm, 2)
        epoch_losses = []
        # Each epoch supervises every training positive/negative exactly once.
        for step in range(2):
            model.train()
            optimizer.zero_grad()
            query_pos = positive[positive_parts[step]]
            context = positive[positive_parts[1-step]]
            query_neg = negative[negative_parts[step]]
            adjacency, edges = make_graph(context, len(node_ids))
            query_codes = (query_pos[:, 0]*len(node_ids) + query_pos[:, 1]).cpu().numpy()
            reverse_codes = (query_pos[:, 1]*len(node_ids) + query_pos[:, 0]).cpu().numpy()
            edge_codes = (edges[0]*len(node_ids) + edges[1]).cpu().numpy()
            assert not np.intersect1d(query_codes, edge_codes).size
            assert not np.intersect1d(reverse_codes, edge_codes).size
            assert not np.intersect1d((query_neg[:, 0]*len(node_ids)+query_neg[:, 1]).cpu().numpy(), edge_codes).size
            mask_records.append(dict(epoch=epoch, step=step,
                query_positive_count=len(query_pos), query_negative_count=len(query_neg),
                context_positive_count=len(context), directed_edge_count=edges.shape[1],
                positive_query_context_overlap=0, positive_reverse_context_overlap=0,
                context_directed_edge_sha256=array_digest(edge_codes), query_positive_sha256=array_digest(query_codes)))
            query = torch.cat([query_pos, query_neg], dim=0)
            labels = torch.cat([torch.ones(len(query_pos), device=DEFAULT_DEVICE),
                                torch.zeros(len(query_neg), device=DEFAULT_DEVICE)])
            z1 = model.encode(adjacency, edges)
            loss = F.binary_cross_entropy_with_logits(model.decode(z1, query), labels)
            contrastive = float(config.get('contrastive', 0.))
            if contrastive:
                z2 = model.encode(adjacency, edges, augment=True)
                loss = loss + contrastive*original.contrastive_loss(z1[model.visible_nodes], z2[model.visible_nodes])
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            epoch_losses.append(float(loss))
        if validation is not None and (epoch % check_every == 0 or epoch == max_epochs):
            model.eval()
            with torch.no_grad():
                probability = torch.sigmoid(model.decode(model.encode(full_adj, full_edges), val_ix)).cpu().numpy()
            result = _metrics(validation.label.to_numpy(), probability)
            eligible = epoch >= min_epochs
            history.append(dict(epoch=epoch, loss=float(np.mean(epoch_losses)), eligible_for_selection=eligible, **result))
            if eligible and (best is None or (result['AUC'], result['AUPR']) > (best['AUC'], best['AUPR'])):
                best = result
                best_epoch = epoch
                best_probability = probability.copy()
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if eligible and best is not None and epoch-best_epoch >= patience:
                break
        if epoch % 25 == 0:
            print(json.dumps(dict(candidate=config.get('name', 'unnamed'), seed=int(seed), epoch=epoch,
                                  mean_training_loss=float(np.mean(epoch_losses)), best=best)), flush=True)
    if validation is not None:
        assert best_state is not None
        model.load_state_dict(best_state)
        output = validation.copy()
        output['probability'] = best_probability
        output.to_csv(folder / 'validation_predictions.csv', index=False)
    else:
        best_epoch = max_epochs
    with (folder / 'model.pt').open('wb') as handle:
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, handle)
    np.savez_compressed(folder / 'epoch_query_assignments.npz',
                        positive_permutation=np.stack(pos_permutations),
                        negative_permutation=np.stack(neg_permutations))
    write_json(folder / 'pretraining_history.json', pretraining_history)
    write_json(folder / 'history.json', history)
    write_json(folder / 'step_mask_checks.json', mask_records)
    result = dict(AUC=best['AUC'] if best else None, AUPR=best['AUPR'] if best else None,
                  epochs=int(best_epoch), epochs_executed=epoch,
                  val_prediction_path=str(folder / 'validation_predictions.csv') if validation is not None else None,
                  artifact_dir=str(folder))
    write_json(folder / 'training_audit.json', dict(**result, seed=int(seed),
        train_pairs=len(train), train_positive=len(positive), train_negative=len(negative),
        train_visible_nodes=int(visible.sum()), feature_dimension=features.shape[1],
        pretraining_epochs=len(pretraining_history), contrastive_loss_nodes='Training-pair node IDs only',
        query_context_positive_overlap=0,
        diffusion_recomputed_each_supervised_step=model.variant != 'no_graph',
        training_edge_supervision='Every positive and every negative exactly once per epoch across two complementary context/query splits',
        encoder='Original DiffusionGraphTransformer with the explicitly registered component ablation only',
        model_description=model_description,
        no_direct_identity_score=True, source_sha256=digest(Path(__file__)),
        original_encoder_source_sha256=digest(ORIGINAL_SOURCE), train_graph_sha256=digest(folder / 'train_graph.csv'),
        feature_sha256=digest(folder / 'features.npy'), config=dict(config), device=str(DEFAULT_DEVICE)))
    return result


def predict_saved(outdir, evaluation_df, device=None):
    """Reconstruct the saved model and train-only graph; never call fit."""
    folder = Path(outdir)
    frame = evaluation_df.copy()
    frame['drug'] = frame.drug.astype(str)
    frame['target'] = frame.target.astype(str)
    training = pd.read_csv(folder / 'training_pairs.csv', dtype={'drug': str, 'target': str})
    assert pairset(training).isdisjoint(pairset(frame)), 'Evaluation pair was used in training'
    graph = pd.read_csv(folder / 'train_graph.csv', dtype=str)
    assert pairset(graph) == pairset(training.loc[training.label == 1])
    ids = np.load(folder / 'node_ids.npy', allow_pickle=False).astype(str)
    x = np.load(folder / 'features.npy', allow_pickle=False)
    visible = np.load(folder / 'train_visible_nodes.npy', allow_pickle=False)
    config = json.loads((folder / 'config.json').read_text(encoding='utf8'))
    selected_device = torch.device(device) if device is not None else DEFAULT_DEVICE
    model = SingleBackbone(x, visible, config).to(selected_device)
    with (folder / 'model.pt').open('rb') as handle:
        state = torch.load(handle, map_location=selected_device)
    model.load_state_dict(state)
    idmap = {identifier: i for i, identifier in enumerate(ids)}
    positive = frame_indices(graph, idmap, selected_device)
    adjacency, edges = make_graph(positive, len(ids))
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model.decode(model.encode(adjacency, edges),
                                      frame_indices(frame, idmap, selected_device))).cpu().numpy()
    assert np.isfinite(p).all()
    del model
    if selected_device.type == 'cuda':
        torch.cuda.empty_cache()
    return p


def fit_predict(train_df, test_df, node_ids, features, config, outdir, seed, epochs):
    """Fixed-epoch outer refit, followed by prediction of disjoint test pairs."""
    assert pairset(train_df).isdisjoint(pairset(test_df))
    train_candidate(train_df, None, node_ids, features, config, outdir, seed, refit_epochs=epochs)
    return predict_saved(outdir, test_df)
