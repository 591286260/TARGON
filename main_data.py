"""Load registered pair/cold-drug partitions and fit the TARGON input per fold."""
import os

for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '2'

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


DATA = Path(__file__).resolve().parent / 'data'
DATASETS = ('Davis', 'DrugBank', 'KIBA')
PROTOCOLS = ('pair', 'cold')


def pair_hash(frame):
    """Use the same CSV serialization that seeded the original query scheduler."""
    payload = frame[['drug', 'target', 'label']].to_csv(index=False, lineterminator='\r\n')
    return hashlib.sha256(payload.encode('utf8')).hexdigest()


def verify_inputs(dataset):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    folder = DATA / dataset
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf8'))
    assert manifest['dataset'] == dataset
    for name, expected in manifest['files_sha256'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'File checksum mismatch: {dataset}/{name}')
    return manifest


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


@lru_cache(maxsize=1)
def _inputs(dataset):
    manifest = verify_inputs(dataset)
    folder = DATA / dataset
    pairs = pd.read_csv(folder / 'pairs.csv.gz', dtype=str, keep_default_na=False)
    pairs['label'] = pairs['label'].astype(np.int64)
    pairs.insert(0, 'pair_index', np.arange(len(pairs)))
    assert pairs.canonical_pair.is_unique and set(pairs.label.unique()) == {0, 1}
    assert int(pairs.label.sum()) == manifest['positive_count']
    assert len(pairs) - int(pairs.label.sum()) == manifest['negative_count']
    attrs = _arrays(folder / 'attributes.npz')
    kernels = _arrays(folder / 'kernels.npz')
    assignments = _arrays(folder / 'folds.npz')
    for group in assignments.values():
        assert group.shape == (len(pairs),) and set(np.unique(group)) == {1, 2, 3, 4, 5}
    return pairs, attrs, kernels, assignments, manifest


def _fit_features(dataset, train, attrs, kernels):
    nodes = attrs['node_ids'].astype(str)
    index = {node:i for i,node in enumerate(nodes)}
    visible = sorted(set(train.drug) | set(train.target))
    visible_set = set(visible)
    fit_rows = np.asarray([index[node] for node in visible])
    base = np.zeros((len(nodes), 32), dtype=np.float32)
    spectral = np.zeros((len(nodes), 64), dtype=np.float32)
    for ids_key, matrix_key, kernel_key in (
        ('drug_ids', 'drug_fingerprints', 'drug_tanimoto'),
        ('protein_ids', 'protein_hashed', 'protein_alignment'),
    ):
        entity_ids = attrs[ids_key].astype(str)
        raw = attrs[matrix_key].astype(np.float64)
        fit = np.asarray([i for i,node in enumerate(entity_ids) if node in visible_set])
        assert len(fit) > 1
        width = min(32, len(fit) - 1, raw.shape[1] - 1)
        svd = TruncatedSVD(n_components=width, random_state=17, n_iter=7).fit(raw[fit])
        components = svd.transform(raw)
        components = StandardScaler().fit(components[fit]).transform(components)
        rows = np.asarray([index[node] for node in entity_ids])
        base[rows, :width] = components

        kernel_order = {node:i for i,node in enumerate(kernels[ids_key].astype(str))}
        perm = np.asarray([kernel_order[node] for node in entity_ids])
        kernel = kernels[kernel_key][np.ix_(perm, perm)].astype(np.float64)
        gram = kernel[np.ix_(fit, fit)]
        symmetric = (gram + gram.T) / 2
        if dataset == 'Davis':
            values, vectors = np.linalg.eigh(symmetric)
        else:
            values, vectors = eigh(symmetric, subset_by_index=[max(0, len(fit)-64), len(fit)-1],
                                   driver='evr', check_finite=True)
        order = np.argsort(values)[::-1]
        keep = order[values[order] > 1e-8][:64]
        projection = (kernel[:, fit] @ vectors[:, keep]) / np.sqrt(values[keep])[None, :]
        projection = StandardScaler().fit(projection[fit]).transform(projection)
        spectral[rows, :len(keep)] = projection
    base = StandardScaler().fit(base[fit_rows]).transform(base).astype(np.float32)
    spectral = StandardScaler().fit(spectral[fit_rows]).transform(spectral).astype(np.float32)
    features = np.concatenate([base, spectral], axis=1)
    assert features.shape == (len(nodes), 96) and np.isfinite(features).all()
    return nodes, features


@lru_cache(maxsize=1)
def _load_partition(dataset, protocol, fold):
    pairs, attrs, kernels, assignments, manifest = _inputs(dataset)
    group = assignments[protocol]
    train = pairs.loc[group != fold].reset_index(drop=True)
    test = pairs.loc[group == fold].reset_index(drop=True)
    expected = manifest['partitions'][protocol][fold-1]
    assert len(train) == expected['train_rows'] and len(test) == expected['heldout_rows']
    assert pair_hash(train) == expected['train_pairs_sha256']
    assert pair_hash(test) == expected['heldout_pairs_sha256']
    assert set(train.canonical_pair).isdisjoint(test.canonical_pair)
    if protocol == 'cold':
        assert set(train.canonical_drug).isdisjoint(test.canonical_drug)
        assert set(train.drug).isdisjoint(test.drug)
    nodes, features = _fit_features(dataset, train, attrs, kernels)
    return train, test, nodes, features


def load_fold(dataset, protocol, fold, work_dir=None):
    """Return train, heldout, node IDs and 96 features. No disk cache is needed.

    The two seeds for the same fold share the fold's in-memory preprocessing fit.
    """
    if protocol not in PROTOCOLS or fold not in range(1, 6):
        raise ValueError('Select protocol pair or cold and a fold from 1 to 5.')
    train, test, nodes, features = _load_partition(dataset, protocol, fold)
    return train.copy(), test.copy(), nodes.copy(), features.copy()
