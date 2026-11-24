# TARGON

**Multi-Scale Topology-Aware Graph Representation Learning for Drug-Target Interaction Prediction**

This repository contains the TARGON model and the fixed Davis, DrugBank, and KIBA benchmarks used for pair-level and cold-drug evaluation.

## Installation

The reported experiments used Python 3.9.16 and PyTorch 2.0.1 with CUDA 11.7.

```bash
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
python run.py --check
```

## Reproduce the main experiments

```bash
python run.py --train --dataset all --protocol pair
python run.py --train --dataset all --protocol cold
```

Each command runs five folds with two seeds per fold for each dataset. Use `--dataset Davis`, `--dataset DrugBank`, or `--dataset KIBA` to run one dataset. For a single fit, add `--fold 1 --seed-index 0`.

Each run writes `seeds.csv`, `folds.csv`, and, when all five folds finish, `summary.csv` under `runs/`. `--output` sets the output directory, and `--keep-fit-files` saves checkpoints and predictions. The runner uses CUDA when available.

| Dataset | Propagation coefficients | Hidden width | Training epochs |
| --- | --- | ---: | ---: |
| Davis | 0.50, 0.60, 0.80 | 1024 | 50 |
| DrugBank | 0.40, 0.80, 0.95 | 128 | 20 |
| KIBA | 0.01, 0.20, 0.95 | 128 | 45 |

## Reproduce Figure 2

Run the following pair-level ablations with the same folds, seeds, and training budgets as Full.

```bash
for variant in no-diff no-trans no-topology single-strength concat; do
  python run.py --train --dataset all --protocol pair --variant "$variant"
done
```

| Option | Change to Full |
| --- | --- |
| `no-diff` | Use the initial node features in each branch and keep the graph Transformer. |
| `no-trans` | Keep diffusion and scale fusion, then apply the final layer normalization without the Transformer blocks. |
| `no-topology` | Use initial features in all branches and give the Transformer an empty edge set at training and inference. Pair labels still provide supervision. |
| `single-strength` | Set all three propagation coefficients to 0.8 and retain independent branch projections and learned scale weights. |
| `concat` | Concatenate the three independently projected diffusion branches. Their widths sum to the Transformer width. |

The [per-seed results](results/ablation_seeds.csv), [fold results](results/ablation_folds.csv), and [summary](results/ablation_summary.csv) correspond to the Figure 2 settings. The plot uses the five-fold mean and sample standard deviation after averaging the two seeds within each fold.

```bash
python -m pip install matplotlib==3.7.1
python plot_ablation.py --output runs/fig2.pdf
```

The supplementary Uniform Fusion comparison fixes the three scale weights to one third throughout training and inference. Use `--variant uniform` with the pair-level protocol. Its [per-seed results](results/uniform_seeds.csv), [fold results](results/uniform_folds.csv), and [summary](results/uniform_summary.csv) use the same Full settings for each dataset.

## Evaluation

The `data/` directory contains the benchmark pairs, their fold assignments, and molecular descriptors and similarity matrices. DrugBank pairs are deduplicated by canonical drug and target identity. The loader fits dimensionality reduction and normalization within each training fold. The graph contains the positive training interactions and their reverse edges. Supervised updates exclude the current query pairs and their canonical equivalents from the message-passing graph.

The cold-drug protocol groups pairs by canonical drug identity. It evaluates both the original model and the input-restoration comparison from the same checkpoint. ACC and F1 use a probability threshold of `p > 0.5`. AUC is ROC AUC, and AUPR is average precision. The runner averages the two seeds within each fold and reports the mean and sample standard deviation over folds.

## Files

- `run.py` runs the fixed evaluation protocols.
- `configs.json` contains the model and training settings.
- `main_data.py` handles fold loading and preprocessing.
- `model/` contains the model, training, and scoring code.
- `cold_inference.py` implements the cold-drug input-restoration comparison.
