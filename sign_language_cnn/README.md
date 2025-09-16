# Sign Language Digits CNN (PyTorch)

Complete, explainable CNN pipeline to classify sign language digit images (0-9).

## Dataset
- Expected 64x64 grayscale images for digits 0-9.
- This repo supports two formats:
  1) Numpy arrays at `A1/data/X.npy` and `A1/data/Y.npy` (as in your workspace)
  2) ImageFolder structure: `data/sign_language_digits/<class>/image.png`

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Run
```bash
python main.py --data_mode npy --npy_dir ..\data --max_runs 6 --epochs 30 --device auto
```

Common flags:
- `--data_mode {npy,image}` and `--npy_dir` or `--image_root`
- `--epochs`, `--patience`, `--batch_sizes`, `--lrs`, `--dropouts`, `--weight_decays`
- `--train_fractions 0.5 0.75 1.0` to vary training size

Outputs:
- Models in `results/models`
- Plots in `results/plots`
- Metrics CSV in `results/metrics.csv`

## Notes
- Uses CUDA automatically if available. Set `--device cpu` to force CPU.
- Reproducible via fixed random seeds.
