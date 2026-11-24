"""Run TARGON on the archived pair or cold-drug folds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from model import trainer
from model.ablations import VARIANTS, make_trainer, variant_recipe
from main_data import load_fold, pair_hash

ROOT = Path(__file__).resolve().parent
DATASETS = ('Davis','DrugBank','KIBA')
METRICS = ['ACC','F1','AUC','AUPR']


def score(labels, probability):
    pred = probability > 0.5
    return dict(ACC=accuracy_score(labels, pred), F1=f1_score(labels, pred,zero_division=0),
                AUC=roc_auc_score(labels, probability), AUPR=average_precision_score(labels, probability))


def check(configs):
    manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf8'))
    for name, checksum in manifest['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != checksum:
            raise ValueError('File differs from the release: ' + name)
    for ds, recipe in configs.items():
        cfg = recipe['config']
        print(ds, 'hidden', cfg['hidden'], 'alphas', cfg['full_alphas'], 'epochs', recipe['epochs'])
    print('Verified', len(manifest['sha256']), 'files. PyTorch', torch.__version__, 'device', trainer.DEFAULT_DEVICE)


def fit_seed(ds, protocol, fold, si, recipe, inputs, directory, keep, *, variant='full', engine=trainer):
    train, test, ids, features = inputs
    seed = (97631, 197631)[si] + 100*fold + 9
    work = directory/'_runtime'/ds/f'f{fold}_s{si}'
    work.mkdir(parents=True, exist_ok=False)
    schedule_path = work/'planned.npz'
    schedule = engine.create_query_schedule(train, seed, max_epochs=160)
    if protocol == 'pair':
        schedule = {key:schedule[key] for key in ('positive_perm','negative_perm','seed','query_rng_seed','train_pair_sha256')}
    np.savez_compressed(schedule_path, **schedule)
    folder = work/'fit'
    p = engine.fit_predict(train, test, ids, features, recipe['config'], folder, seed,
                           recipe['epochs'], schedule_path=schedule_path)
    predictions = {VARIANTS[variant][0]: p}
    if protocol == 'cold':
        from cold_inference import paired_predictions
        original, restored = paired_predictions(engine, folder, train, test, ids)
        np.testing.assert_allclose(original, p, rtol=0, atol=3e-6)
        predictions = {'Original':original, 'Restored':restored}
    rows = [dict(dataset=ds, protocol=protocol, fold=fold, seed=seed, seed_index=si,
                 setting=arm, **score(test.label, prob)) for arm, prob in predictions.items()]
    if not keep:
        shutil.rmtree(work)
    return rows


def save_tables(directory, rows, seeds_per_fold):
    seeds = pd.DataFrame(rows)
    seeds.to_csv(directory/'seeds.csv', index=False)
    groups = ['dataset','protocol','setting']
    completed = seeds.groupby(groups+['fold'], sort=False).seed_index.transform('nunique') == seeds_per_fold
    folds = seeds.loc[completed].groupby(groups+['fold'], as_index=False, sort=False)[METRICS].mean()
    folds.to_csv(directory/'folds.csv', index=False)
    summaries = []
    for key, block in folds.groupby(groups, sort=False):
        if len(block) == 5:
            for stat, values in [('Average',block[METRICS].mean()), ('Std',block[METRICS].std(ddof=1))]:
                summaries.append(dict(zip(groups,key), statistic=stat, **values.to_dict()))
    if summaries:
        pd.DataFrame(summaries).to_csv(directory/'summary.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true',help='Verify files and print the fixed recipes.')
    group.add_argument('--train',action='store_true',help='Train and evaluate the requested split.')
    parser.add_argument('--dataset',choices=(*DATASETS,'all'),default='all')
    parser.add_argument('--protocol',choices=('pair','cold'),default='pair')
    parser.add_argument('--variant',choices=VARIANTS,default='full',help='Model variant for pair-level evaluation.')
    parser.add_argument('--fold',type=int,choices=range(1,6),help='Run one fold (default: all five).')
    parser.add_argument('--seed-index',type=int,choices=(0,1),help='Run one seed (default: both).')
    parser.add_argument('--output',type=Path,default=ROOT/'runs',help='Parent directory for new runs.')
    parser.add_argument('--keep-fit-files',action='store_true',help='Keep checkpoints and per-fit diagnostics.')
    args = parser.parse_args()
    if args.protocol == 'cold' and args.variant != 'full':
        parser.error('The cold-drug protocol requires --variant full.')
    configs = json.loads((ROOT/'configs.json').read_text(encoding='utf8'))
    if args.check:
        check(configs)
        return
    configs = {ds: variant_recipe(recipe, args.variant) for ds, recipe in configs.items()}
    engine = make_trainer(args.variant)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # The original query RNG uses a Windows CSV serialization on every platform.
    engine.train_pair_hash = pair_hash
    args.output.mkdir(parents=True,exist_ok=True)
    prefix = args.protocol + '_' if args.variant == 'full' else f'{args.protocol}_{args.variant}_'
    directory = Path(tempfile.mkdtemp(prefix=prefix,dir=str(args.output.resolve())))
    engine.HERE = directory
    engine.RUN_ID = directory.name
    print('Output:',directory,flush=True)
    (directory/'settings.json').write_text(json.dumps(dict(dataset=args.dataset, protocol=args.protocol,
        variant=args.variant, fold=args.fold, seed_index=args.seed_index, recipes=configs,
        torch=str(torch.__version__), device=str(engine.DEFAULT_DEVICE)),indent=2)+'\n', encoding='utf8')
    rows = []
    for ds in (DATASETS if args.dataset=='all' else (args.dataset,)):
        for fold in (range(1,6) if args.fold is None else [args.fold]):
            inputs = load_fold(ds,args.protocol,fold,directory/'_runtime'/'data'/ds/f'f{fold}')
            for si in ((0,1) if args.seed_index is None else (args.seed_index,)):
                rows.extend(fit_seed(ds,args.protocol,fold,si,configs[ds],inputs,directory,args.keep_fit_files,
                                     variant=args.variant,engine=engine))
                save_tables(directory,rows,2 if args.seed_index is None else 1)
                print(ds,'fold',fold,'seed',si,'finished',flush=True)
    print('Results:',directory,flush=True)


if __name__=='__main__':
    main()
