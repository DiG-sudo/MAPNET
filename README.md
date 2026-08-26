# MAPNet

MAPNet predicts circRNA-RBP binding sites from three complementary feature views: KNF, CircRNA2Vec, and physicochemical descriptors. The implementation contains the full Dual-level Adaptive Multi-view Fusion (DAMF) Block and Gated Local-Importance Pyramid (GLIP) used in the paper.

The repository includes the 37 raw circRNA-RBP datasets and the published CircRNA2Vec vectors. It does not include generated feature archives, grid-search results, MAPNet checkpoints, or final result tables. These files are produced locally by the three stages below.

## Requirements

- Python 3.10 or 3.11
- CUDA-capable PyTorch environment for model search and training

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Repository layout

```text
MAPNet_public/
├── README.md
├── requirements.txt
├── config.yaml
├── prepare_data.py
├── data.py
├── model.py
├── training.py
├── search.py
├── train.py
├── data/circRNA-RBP/
└── weights/circRNA2Vec.txt
```

## Stage 1: generate model inputs

`prepare_data.py` reads each RBP's positive and negative sequence files, loads the provided CircRNA2Vec vectors, and generates the six descriptor tensors used by MAPNet. CircRNA2Vec itself is not retrained.

```bash
python prepare_data.py --config config.yaml
```

The generated NPZ files are written to `generated_features/` and are intentionally excluded from Git.

## Stage 2: search the DAMF branch count

The paper searches the number of candidate DAMF branches independently for each RBP. Candidate configurations are selected only with validation-set metrics; the test set is not used during this stage.

```bash
python search.py --num-experts 2 4 6 8 10 12
```

The selection order is validation AUC, validation loss, validation F1, and then the smaller model when all preceding values tie. The command writes:

- `outputs/grid_search_results.csv`: all candidate validation results;
- `outputs/grid_best_results.csv`: one validation-selected configuration per RBP.

## Stage 3: retrain and test

Each RBP-specific model is initialized again with the same seed, retrained with its validation-selected number of branches, and evaluated on the test split.

```bash
python train.py --use-search-results outputs/grid_best_results.csv
```

This stage writes final metrics to `outputs/test_results.csv` and checkpoints to `outputs/checkpoints/`. Both are excluded from Git.

## Partial runs

All three commands accept `--proteins` for a subset, for example:

```bash
python prepare_data.py --config config.yaml --proteins WTAP
python search.py --num-experts 2 4 --proteins WTAP
python train.py --use-search-results outputs/grid_best_results.csv --proteins WTAP
```

The paper configuration, including seed 55, the 70/15/15 split, seven GLIP scales, early stopping, and optimization settings, is defined in `config.yaml`.
