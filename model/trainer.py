"""Fold-local graph construction, training, and inference for TARGON."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import hashlib
import importlib.util
import json
import platform
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
REFERENCE_SOURCE = HERE / 'reference_trainer.py'
ORIGINAL_SOURCE = HERE / 'reference_graph.py'
_spec = importlib.util.spec_from_file_location('targon_backbone', REFERENCE_SOURCE)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)
original = reference.original
torch.set_num_threads(2)
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUN_ID = 'targon'
EXPECTED_REFERENCE_SHA256 = '418af12ac07c7fb9e59f364c6ab20627c3c61f6e6377f7d1364ef06b93ed68dd'
EXPECTED_GRAPH_SHA256 = '0ae425036e7d963032e1840d955888c68b110d6eaad5074b8eb27ede28ccbe55'
ALPHA_GRID = (.01, .05, .1, .2, .3, .4, .6, .7, .8, .9, .95, .99, .5, .53, .56, .59)
DEFAULT_FULL_ALPHAS = (.1, .3, .6)
FULL_STRENGTH_CASES = (
    (.1, .3, .6), (.05, .1, .3), (.6, .8, .95),
    (.05, .3, .8), (.1, .6, .95), (.05, .8, .95),
    (.01, .05, .1), (.9, .95, .99),
    (.2, .8, .95), (.4, .8, .95),
    (.7, .8, .95),
    (.6, .9, .95),
    (.6, .8, .9),
    (.6, .8, .99),
    (.8, .9, .95),
    (.7, .8, .9),
    (.5, .8, .9),
    (.5, .7, .8),
    (.5, .6, .9), (.5, .7, .9),
    (.53, .7, .9), (.56, .7, .9), (.59, .7, .9),
    (.4, .6, .9), (.4, .8, .9),
    (.4, .7, .8), (.4, .7, .9),
    (.5, .6, .8), (.5, .6, .7),
    (.4, .6, .8), (.4, .6, .7),
    (.2, .7, .95), (.3, .7, .95), (.3, .8, .95), (.01, .2, .95),
)
VARIANT_SEMANTICS = {
    'full_global': dict(diffusion='one-step explicit parent-selected config.full_alphas; no default fallback', transformer='parent layers', fusion='learned shared softmax gamma3'),
    'no_diffusion_global': dict(diffusion='[X,X,X], no diffusion call; three independent projections retained', transformer='parent layers and real context edges', fusion='learned shared softmax gamma3'),
    'no_transformer_global': dict(diffusion='one-step explicit parent-selected config.full_alphas; no default fallback', transformer='construct parent model including decoder, then remove GT layers; retain out_norm', fusion='learned shared softmax gamma3'),
    'single_strength': dict(diffusion='one-step config.single_alpha from .05/.1/.3/.6/.8/.95 repeated three times, [Halpha,Halpha,Halpha]; selected by external ordinary-fivefold CV; no inner fold', transformer='parent layers', fusion='learned shared softmax gamma3; three independent projections retained'),
}

write_json = reference.write_json
digest = reference.digest
array_digest = reference.array_digest
pairset = reference.pairset
set_seed = reference.set_seed
frame_indices = reference.frame_indices
make_graph = reference.make_graph
_metrics = reference._metrics
GlobalScaleFusion = reference.GlobalScaleFusion


def _time():
    return datetime.now().astimezone().isoformat()


def _canonical_keys(frame):
    cols = ('canonical_drug', 'canonical_target')
    if all(c in frame for c in cols):
        assert not frame[list(cols)].isna().any().any()
    else:
        cols = ('drug', 'target')
    return list(map(tuple, frame[list(cols)].astype(str).to_numpy()))


def _validated_frame(frame):
    frame = reference._validated_frame(frame)
    assert np.isin(frame.label, [0, 1]).all()
    frame['label'] = frame.label.astype(int)
    by_key = {}
    for key, label in zip(_canonical_keys(frame), frame.label):
        assert key not in by_key or by_key[key] == label, 'Canonical aliases have contradictory labels'
        by_key[key] = label
    return frame


def _assert_disjoint(train, evaluation):
    assert pairset(train).isdisjoint(pairset(evaluation)), 'Evaluation pair entered training'
    assert set(_canonical_keys(train)).isdisjoint(_canonical_keys(evaluation)), 'Canonical evaluation alias entered training'


def train_pair_hash(train):
    """Root contract: actual train row order; no pair_index/canonical columns."""
    part = train[['drug', 'target', 'label']].copy()
    part['drug'] = part.drug.astype(str)
    part['target'] = part.target.astype(str)
    part['label'] = part.label.astype(int)
    return hashlib.sha256(part.to_csv(index=False).encode('utf8')).hexdigest()


def create_query_schedule(train, seed, max_epochs=160):
    """Pure isolated RNG; also used for explicit synthetic QA fallback."""
    pair_hash = train_pair_hash(train)
    rng_seed = int.from_bytes(hashlib.sha256(
        ('legacy_query_v1|' + pair_hash + '|' + str(int(seed))).encode()).digest()[:8], 'little') % (2**63-1)
    rng = np.random.default_rng(rng_seed)
    np_, nn_ = int((train.label == 1).sum()), int((train.label == 0).sum())
    pp, nparts = [], []
    for _ in range(int(max_epochs)):
        pp.append(rng.permutation(np_).astype(np.int32))
        nparts.append(rng.permutation(nn_).astype(np.int32))
    return dict(positive_perm=np.stack(pp), negative_perm=np.stack(nparts),
                seed=np.asarray(seed, dtype=np.int64), train_pair_sha256=np.asarray(pair_hash),
                query_rng_seed=np.asarray(rng_seed, dtype=np.int64))


def _load_schedule(train, seed, max_epochs, schedule_path, folder):
    required = ('positive_perm', 'negative_perm', 'seed', 'train_pair_sha256', 'query_rng_seed')
    if schedule_path is None:
        schedule = create_query_schedule(train, seed, max(160, max_epochs))
        np.savez_compressed(folder / 'query_schedule.npz', **schedule)
        mode = 'explicitly_recorded_isolated_rng_fallback; formal caller must supply schedule_path'
    else:
        source = Path(schedule_path).resolve()
        assert source.is_file(), 'Missing registered query schedule'
        shutil.copyfile(source, folder / 'query_schedule.npz')
        with np.load(source, allow_pickle=False) as saved:
            assert set(required) <= set(saved.files)
            schedule = {key: saved[key].copy() for key in required}
        mode = 'provided_registered_schedule'
    assert int(schedule['seed']) == int(seed), 'Query schedule network seed mismatch'
    assert str(schedule['train_pair_sha256'].item()) == train_pair_hash(train), 'Query schedule training partition/order mismatch'
    pp, nparts = schedule['positive_perm'], schedule['negative_perm']
    assert pp.ndim == nparts.ndim == 2 and len(pp) == len(nparts) and len(pp) >= max_epochs
    assert pp.shape[1] == int((train.label == 1).sum()) and nparts.shape[1] == int((train.label == 0).sum())
    assert pp.dtype == np.dtype('int32') and nparts.dtype == np.dtype('int32')
    expected = create_query_schedule(train, seed, len(pp))
    assert int(schedule['query_rng_seed']) == int(expected['query_rng_seed'])
    assert np.array_equal(pp, expected['positive_perm']) and np.array_equal(nparts, expected['negative_perm']), 'Query schedule differs from registered isolated RNG rule'
    write_json(folder / 'query_schedule_metadata.json', dict(
        mode=mode, original_path=str(Path(schedule_path).resolve()) if schedule_path is not None else None,
        saved_file_sha256=digest(folder / 'query_schedule.npz'), train_pair_sha256=train_pair_hash(train),
        seed=int(seed), query_rng_seed=int(schedule['query_rng_seed']),
        positive_shape=list(pp.shape), negative_shape=list(nparts.shape),
        positive_array_sha256=array_digest(pp), negative_array_sha256=array_digest(nparts),
        query_rng_independent_of_network_augmentation_rng=True))
    return pp, nparts


class SingleBackbone(reference.SingleBackbone):
    def __init__(self, features, visible_nodes, config):
        variant = config.get('variant', 'full_global')
        assert variant in VARIANT_SEMANTICS, variant
        assert config.get('fusion_mode', 'global') == 'global'
        alpha = config.get('single_alpha', None)
        if variant == 'single_strength':
            assert alpha is not None and np.isfinite(float(alpha)) and float(alpha) in ALPHA_GRID, 'single_alpha must be one registered grid value'
        else:
            assert alpha is None, 'single_alpha only belongs to single_strength'
        if variant in ('full_global', 'no_diffusion_global', 'no_transformer_global'):
            assert config.get('full_alphas') is not None, 'Explicit parent-selected full_alphas required; no default fallback'
            requested = config['full_alphas']
            assert isinstance(requested, (tuple, list)), 'full_alphas must preserve a registered three-branch order'
            full_alphas = tuple(float(a) for a in requested)
            assert len(full_alphas) == 3 and all(np.isfinite(a) for a in full_alphas)
            assert full_alphas in FULL_STRENGTH_CASES, 'full_alphas must be one registered ordered triplet'
        else:
            assert config.get('full_alphas') is None, 'single_strength cannot carry full_alphas'
            full_alphas = DEFAULT_FULL_ALPHAS
        # Construct every parent layer AND decoder before deleting GT. This
        # preserves both retained initial tensors and the post-init RNG state.
        parent = dict(config, variant='global_fusion', fusion_mode='global')
        super().__init__(features, visible_nodes, parent)
        self.config = dict(config)
        self.variant = variant
        self.single_alpha = float(alpha) if alpha is not None else None
        self.full_alphas = full_alphas  # Plain tuple: no state keys, parameters or RNG draws.
        self.diffusion_enabled = variant != 'no_diffusion_global'
        self.transformer_enabled = variant != 'no_transformer_global'
        if not self.transformer_enabled:
            self.encoder.layers = nn.ModuleList([])

    def encode(self, adjacency, edge_index, augment=False):
        x = self.features
        if self.identity is not None:
            x = torch.cat([x, self.identity.weight * self.visible_nodes[:, None]], dim=1)
        if not self.diffusion_enabled:
            diff = [x, x, x]
        elif self.variant == 'single_strength':
            value = original.fast_diffusion(adjacency, x, self.single_alpha)
            diff = [value, value, value]
        else:
            diff = original.multi_scale_diffusion(adjacency, x, list(self.full_alphas))
        if augment:
            diff = [original.augment_features(value, feat_drop=.1, noise_std=.01) for value in diff]
        return self.encoder(diff, edge_index)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return value


def _save_torch(value, path):
    # PyTorch 2.0's Windows C++ filename writer rejects some Unicode paths.
    with Path(path).open('wb') as handle:
        torch.save(value, handle)


def _load_torch(path, map_location):
    with Path(path).open('rb') as handle:
        return torch.load(handle, map_location=map_location)


def state_digests(state):
    per_tensor = {}
    total = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        record = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_digest(value))
        per_tensor[name] = record
        total.update(json.dumps([name, record], sort_keys=True).encode('utf8'))
    return dict(state_sha256=total.hexdigest(), tensors=per_tensor)


def _graph_record(adjacency, edges):
    # CPU metadata reconstructs the original 1/(degree+1e-6) operator; it is
    # descriptive/auditable and does not replace CUDA/CPU training arithmetic.
    a = adjacency.detach().cpu().coalesce()
    degree = torch.sparse.sum(a, dim=1).to_dense()
    inv = 1. / (degree + 1e-6)
    pvals = inv[a.indices()[0]] * a.values()
    return dict(context_directed_edge_sha256=array_digest(edges.detach().cpu().numpy()),
                adjacency_indices_sha256=array_digest(a.indices().numpy()),
                adjacency_values_sha256=array_digest(a.values().numpy()),
                degree_sha256=array_digest(degree.numpy()), inverse_degree_sha256=array_digest(inv.numpy()),
                normalized_adjacency_values_sha256=array_digest(pvals.numpy()),
                normalization='D_inv=1/(sparse_sum(A,dim=1)+1e-6); P=D_inv*A',
                normalization_metadata_backend='CPU float32; training uses original device arithmetic')


def _save_prediction(folder, frame, probability, stem):
    p = np.asarray(probability, dtype=np.float64)
    assert p.shape == (len(frame),) and np.isfinite(p).all()
    np.save(folder / (stem + '_probabilities.npy'), p)
    output = frame.copy()
    output['probability'] = p
    output.to_csv(folder / (stem + '_predictions.csv'), index=False, float_format='%.17g')
    restored = pd.read_csv(folder / (stem + '_predictions.csv'), float_precision='round_trip')['probability'].to_numpy()
    assert np.array_equal(restored, p), 'Probability CSV is not lossless float64'
    return p


def train_candidate(train_df, val_df, node_ids, features, config, outdir, seed,
                    refit_epochs=None, schedule_path=None):
    """Original BCE/optional CL training with independent saved query schedules."""
    assert digest(REFERENCE_SOURCE) == EXPECTED_REFERENCE_SHA256
    assert digest(ORIGINAL_SOURCE) == EXPECTED_GRAPH_SHA256
    folder = Path(outdir)
    assert folder.resolve().is_relative_to((HERE / '_runtime').resolve()), 'Large fit artifacts must stay under this run _runtime'
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'model_initial.pt').exists(), 'Refusing to overwrite an initialized fit'
    train = _validated_frame(train_df)
    validation = _validated_frame(val_df) if val_df is not None else None
    assert (refit_epochs is None) == (validation is not None)
    if validation is not None:
        _assert_disjoint(train, validation)
    node_ids = np.asarray(node_ids, dtype=str)
    assert len(node_ids) == len(set(node_ids))
    features = np.asarray(features, dtype=np.float32)
    assert features.ndim == 2 and len(features) == len(node_ids) and np.isfinite(features).all()
    idmap = {identifier: i for i, identifier in enumerate(node_ids)}
    train_nodes = set(train.drug) | set(train.target)
    assert train_nodes <= set(idmap)
    visible = np.asarray([identifier in train_nodes for identifier in node_ids])
    positive_rows = train.loc[train.label == 1].copy()
    negative_rows = train.loc[train.label == 0].copy()
    positive_frame = positive_rows[['drug', 'target']]
    negative_frame = negative_rows[['drug', 'target']]
    assert len(positive_frame) >= 2 and len(negative_frame) >= 2
    train.to_csv(folder / 'training_pairs.csv', index=False)
    positive_frame.to_csv(folder / 'train_graph.csv', index=False)
    if validation is not None:
        validation.to_csv(folder / 'validation_pairs.csv', index=False)
    np.save(folder / 'features.npy', features)
    np.save(folder / 'node_ids.npy', node_ids)
    np.save(folder / 'train_visible_nodes.npy', visible)
    write_json(folder / 'config.json', dict(config))
    source_dir = folder / 'sources'
    source_dir.mkdir(exist_ok=True)
    for path in (Path(__file__), REFERENCE_SOURCE, ORIGINAL_SOURCE):
        shutil.copyfile(path, source_dir / path.name)
    write_json(folder / 'environment.json', dict(recorded_at=_time(), run_id=RUN_ID,
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__,
        cuda_runtime=torch.version.cuda, device=str(DEFAULT_DEVICE),
        cuda_device=torch.cuda.get_device_name(DEFAULT_DEVICE) if DEFAULT_DEVICE.type == 'cuda' else None,
        source_sha256=digest(Path(__file__)), reference_trainer_sha256=digest(REFERENCE_SOURCE),
        reference_graph_sha256=digest(ORIGINAL_SOURCE)))
    max_epochs = int(refit_epochs) if refit_epochs is not None else int(config.get('max_epochs', 160))
    assert max_epochs > 0
    positive_schedule, negative_schedule = _load_schedule(train, seed, max_epochs, schedule_path, folder)
    set_seed(int(seed))
    model = SingleBackbone(features, visible, config).to(DEFAULT_DEVICE)
    initial_state = _cpu_tree(model.state_dict())
    _save_torch(initial_state, folder / 'model_initial.pt')
    write_json(folder / 'model_initial_digests.json', state_digests(initial_state))
    model_description = dict(variant=model.variant, variant_semantics=VARIANT_SEMANTICS[model.variant],
        total_parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        configured_transformer_layers=int(config.get('layers', 2)), effective_transformer_layers=len(model.encoder.layers),
        fusion_mode='global', feature_mode=config.get('feature_mode', 'caller_supplied'),
        feature_input_dimension=int(features.shape[1]), learned_identity_dimension=int(config.get('id_dim', 0)),
        decoder_head=model.head, diffusion_enabled=model.diffusion_enabled,
        configured_full_alphas=list(model.full_alphas) if model.variant != 'single_strength' else None,
        parent_full_alphas=config.get('parent_full_alphas'),
        branch_alphas=([model.single_alpha]*3 if model.variant == 'single_strength' else list(model.full_alphas)) if model.diffusion_enabled else None,
        gt_neighbor_messages_enabled=bool(len(model.encoder.layers)),
        full_parent_initialized_before_module_removal=True, contrastive_unchanged=True,
        pretrain_epochs=int(config.get('pretrain_epochs', 0)), joint_contrastive_weight=float(config.get('contrastive', 0.)))
    write_json(folder / 'model_description.json', model_description)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('lr', .001)),
                                 weight_decay=float(config.get('weight_decay', .001)))
    _save_torch(_cpu_tree(optimizer.state_dict()), folder / 'optimizer_initial.pt')
    positive = frame_indices(positive_frame, idmap, DEFAULT_DEVICE)
    negative = frame_indices(negative_frame, idmap, DEFAULT_DEVICE)
    full_adj, full_edges = make_graph(positive, len(node_ids))
    write_json(folder / 'full_train_graph_normalization.json', _graph_record(full_adj, full_edges))
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
    _save_torch(_cpu_tree(model.state_dict()), folder / 'model_after_pretrain.pt')
    check_every = int(config.get('check_every', 5))
    min_epochs = int(config.get('min_epochs', 30))
    patience = int(config.get('patience', 35))
    if validation is not None:
        assert max_epochs >= min_epochs > 0
    history, mask_records, pos_permutations, neg_permutations = [], [], [], []
    best = best_state = best_optimizer = best_probability = None
    best_epoch = 0
    positive_keys = _canonical_keys(positive_rows)
    negative_keys = _canonical_keys(negative_rows)
    assert set(positive_keys).isdisjoint(negative_keys), 'Negative sampling includes known positive aliases'
    for epoch in range(1, max_epochs + 1):
        pos_order = positive_schedule[epoch-1].copy()
        neg_order = negative_schedule[epoch-1].copy()
        pos_permutations.append(pos_order)
        neg_permutations.append(neg_order)
        positive_parts = torch.tensor_split(torch.as_tensor(pos_order.astype(np.int64), device=DEFAULT_DEVICE), 2)
        negative_parts = torch.tensor_split(torch.as_tensor(neg_order.astype(np.int64), device=DEFAULT_DEVICE), 2)
        epoch_losses = []
        for step in range(2):
            model.train()
            optimizer.zero_grad()
            query_pos_indices = positive_parts[step]
            context_indices = positive_parts[1-step]
            query_neg_indices = negative_parts[step]
            query_aliases = {positive_keys[i] for i in query_pos_indices.cpu().tolist()}
            query_aliases.update(negative_keys[i] for i in query_neg_indices.cpu().tolist())
            keep = [i for i in context_indices.cpu().tolist() if positive_keys[i] not in query_aliases]
            context_indices = torch.as_tensor(keep, dtype=torch.long, device=DEFAULT_DEVICE)
            query_pos = positive[query_pos_indices]
            context = positive[context_indices]
            query_neg = negative[query_neg_indices]
            adjacency, edges = make_graph(context, len(node_ids))
            query_codes = (query_pos[:,0]*len(node_ids)+query_pos[:,1]).cpu().numpy()
            reverse_codes = (query_pos[:,1]*len(node_ids)+query_pos[:,0]).cpu().numpy()
            edge_codes = (edges[0]*len(node_ids)+edges[1]).cpu().numpy()
            neg_codes = (query_neg[:,0]*len(node_ids)+query_neg[:,1]).cpu().numpy()
            neg_reverse_codes = (query_neg[:,1]*len(node_ids)+query_neg[:,0]).cpu().numpy()
            assert not np.intersect1d(np.concatenate([query_codes,reverse_codes,neg_codes,neg_reverse_codes]), edge_codes).size
            assert not {positive_keys[i] for i in keep} & query_aliases
            mask_records.append(dict(epoch=epoch, step=step,
                query_positive_count=len(query_pos), query_negative_count=len(query_neg),
                context_positive_count=len(context), directed_edge_count=edges.shape[1],
                positive_query_context_overlap=0, positive_reverse_context_overlap=0, canonical_query_context_overlap=0,
                query_positive_sha256=array_digest(query_codes), query_negative_sha256=array_digest(neg_codes),
                context_positive_local_indices_sha256=array_digest(np.asarray(keep,dtype=np.int64)),
                **_graph_record(adjacency, edges)))
            query = torch.cat([query_pos, query_neg], dim=0)
            labels = torch.cat([torch.ones(len(query_pos),device=DEFAULT_DEVICE), torch.zeros(len(query_neg),device=DEFAULT_DEVICE)])
            z1 = model.encode(adjacency, edges)
            loss = F.binary_cross_entropy_with_logits(model.decode(z1, query), labels)
            contrastive = float(config.get('contrastive', 0.))
            if contrastive:
                z2 = model.encode(adjacency, edges, augment=True)
                loss = loss + contrastive*original.contrastive_loss(z1[model.visible_nodes],z2[model.visible_nodes])
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            epoch_losses.append(float(loss))
        record = dict(epoch=epoch, loss=float(np.mean(epoch_losses)), step_losses=epoch_losses, eligible_for_selection=False)
        if validation is not None and (epoch % check_every == 0 or epoch == max_epochs):
            model.eval()
            with torch.no_grad():
                probability = torch.sigmoid(model.decode(model.encode(full_adj,full_edges),val_ix)).cpu().numpy()
            result = _metrics(validation.label.to_numpy(), probability)
            eligible = epoch >= min_epochs
            record.update(eligible_for_selection=eligible, **result)
            if eligible and (best is None or (result['AUC'],result['AUPR']) > (best['AUC'],best['AUPR'])):
                best = result
                best_epoch = epoch
                best_probability = probability.copy()
                best_state = _cpu_tree(model.state_dict())
                best_optimizer = _cpu_tree(optimizer.state_dict())
        history.append(record)
        if epoch % 25 == 0:
            print(json.dumps(dict(candidate=config.get('name','unnamed'),variant=model.variant,seed=int(seed),epoch=epoch,mean_training_loss=record['loss'],best=best)),flush=True)
        if validation is not None and record['eligible_for_selection'] and best is not None and epoch-best_epoch >= patience:
            break
    final_state = _cpu_tree(model.state_dict())
    _save_torch(final_state, folder / 'model_final.pt')
    _save_torch(_cpu_tree(optimizer.state_dict()), folder / 'optimizer_final.pt')
    write_json(folder / 'model_final_digests.json', state_digests(final_state))
    if validation is not None:
        assert best_state is not None
        model.load_state_dict(best_state)
        _save_prediction(folder, validation, best_probability, 'validation')
    else:
        best_epoch = max_epochs
        best_optimizer = _cpu_tree(optimizer.state_dict())
    selected_state = _cpu_tree(model.state_dict())
    _save_torch(selected_state, folder / 'model.pt')
    _save_torch(best_optimizer, folder / 'optimizer.pt')
    write_json(folder / 'model_selected_digests.json', state_digests(selected_state))
    np.savez_compressed(folder / 'epoch_query_assignments.npz', positive_permutation=np.stack(pos_permutations), negative_permutation=np.stack(neg_permutations))
    write_json(folder / 'pretraining_history.json', pretraining_history)
    write_json(folder / 'history.json', history)
    write_json(folder / 'step_mask_checks.json', mask_records)
    result = dict(AUC=best['AUC'] if best else None,AUPR=best['AUPR'] if best else None,
        epochs=int(best_epoch),epochs_executed=epoch,
        val_prediction_path=str(folder/'validation_predictions.csv') if validation is not None else None,artifact_dir=str(folder))
    write_json(folder / 'training_audit.json', dict(**result,run_id=RUN_ID,recorded_at=_time(),seed=int(seed),
        train_pairs=len(train),train_positive=len(positive),train_negative=len(negative),train_visible_nodes=int(visible.sum()),
        feature_dimension=int(features.shape[1]),pretraining_epochs=len(pretraining_history),contrastive_loss_nodes='Training-pair node IDs only',
        query_context_positive_overlap=0,diffusion_recomputed_each_supervised_step=model.diffusion_enabled,
        training_edge_supervision='Every training positive and negative exactly once per epoch across two complementary query halves; context removes all canonical aliases of either query half',
        model_description=model_description,no_direct_identity_score=True,source_sha256=digest(Path(__file__)),
        reference_trainer_source_sha256=digest(REFERENCE_SOURCE),original_encoder_source_sha256=digest(ORIGINAL_SOURCE),
        train_graph_sha256=digest(folder/'train_graph.csv'),feature_sha256=digest(folder/'features.npy'),config=dict(config),device=str(DEFAULT_DEVICE),
        training_pair_sha256=train_pair_hash(train),query_schedule_sha256=digest(folder/'query_schedule.npz'),
        actual_query_assignments_sha256=digest(folder/'epoch_query_assignments.npz'),
        initial_state_sha256=state_digests(initial_state)['state_sha256'],final_state_sha256=state_digests(final_state)['state_sha256'],
        selected_state_sha256=state_digests(selected_state)['state_sha256']))
    return result


def predict_saved(outdir, evaluation_df, device=None):
    """Reconstruct frozen selected weights; original CPU/CUDA policy unchanged."""
    folder = Path(outdir)
    frame = evaluation_df.copy()
    frame['drug'] = frame.drug.astype(str)
    frame['target'] = frame.target.astype(str)
    training = pd.read_csv(folder/'training_pairs.csv',dtype={'drug':str,'target':str},float_precision='round_trip')
    _assert_disjoint(training,frame)
    graph = pd.read_csv(folder/'train_graph.csv',dtype=str)
    assert pairset(graph) == pairset(training.loc[training.label==1])
    ids = np.load(folder/'node_ids.npy',allow_pickle=False).astype(str)
    x = np.load(folder/'features.npy',allow_pickle=False)
    visible = np.load(folder/'train_visible_nodes.npy',allow_pickle=False)
    config = json.loads((folder/'config.json').read_text(encoding='utf8'))
    selected_device = torch.device(device) if device is not None else DEFAULT_DEVICE
    model = SingleBackbone(x,visible,config).to(selected_device)
    model.load_state_dict(_load_torch(folder/'model.pt',selected_device))
    idmap = {identifier:i for i,identifier in enumerate(ids)}
    adjacency,edges = make_graph(frame_indices(graph,idmap,selected_device),len(ids))
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model.decode(model.encode(adjacency,edges),frame_indices(frame,idmap,selected_device))).cpu().numpy()
    assert np.isfinite(p).all()
    del model
    if selected_device.type == 'cuda':
        torch.cuda.empty_cache()
    return p


def fit_predict(train_df,test_df,node_ids,features,config,outdir,seed,epochs,schedule_path=None):
    _assert_disjoint(train_df,test_df)
    train_candidate(train_df,None,node_ids,features,config,outdir,seed,refit_epochs=epochs,schedule_path=schedule_path)
    folder = Path(outdir)
    test_df.to_csv(folder/'test_pairs.csv',index=False)
    p = predict_saved(outdir,test_df)
    _save_prediction(folder,test_df,p,'test')
    return p
