import argparse
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import csv

from src.utils import set_seed, make_dataloaders_from_npy, make_dataloaders_from_image_folder, show_samples_from_loader, get_device
from src.train import TrainConfig, fit
from src.evaluate import plot_confusion_matrix, compute_metrics, plot_roc_curves


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Sign Language Digits CNN")
	parser.add_argument("--data_mode", choices=["npy", "image"], default="npy")
	parser.add_argument("--npy_dir", type=str, default="..\\data")
	parser.add_argument("--image_root", type=str, default="data/sign_language_digits")
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--patience", type=int, default=7)
	parser.add_argument("--batch_sizes", nargs='+', type=int, default=[16, 32, 64])
	parser.add_argument("--lrs", nargs='+', type=float, default=[1e-3, 1e-2, 1e-4])
	parser.add_argument("--dropouts", nargs='+', type=float, default=[0.2, 0.3, 0.5])
	parser.add_argument("--weight_decays", nargs='+', type=float, default=[1e-3, 1e-2])
	parser.add_argument("--train_fractions", nargs='+', type=float, default=[0.5, 0.75, 1.0])
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--max_runs", type=int, default=9, help="Limit total combinations to run")
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def class_names_map() -> Dict[int, str]:
	return {i: str(i) for i in range(10)}


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	# Resolve device
	if args.device == "auto":
		resolved_device = "cuda" if str(get_device()) == "cuda" else "cpu"
	else:
		resolved_device = args.device

	# Prepare hyperparameter grid and cap runs
	grid = list(product(args.batch_sizes, args.lrs, args.dropouts, args.weight_decays, args.train_fractions))
	if args.max_runs and len(grid) > args.max_runs:
		grid = grid[: args.max_runs]

	metrics_rows: List[Dict[str, float]] = []

	for idx, (batch_size, lr, dropout, weight_decay, train_fraction) in enumerate(grid, start=1):
		run_name = f"bs{batch_size}_lr{lr}_do{dropout}_wd{weight_decay}_frac{train_fraction}"
		print(f"[Run {idx}/{len(grid)}] {run_name}")

		if args.data_mode == "npy":
			train_loader, val_loader, test_loader = make_dataloaders_from_npy(Path(args.npy_dir), batch_size=batch_size, train_fraction=train_fraction, seed=args.seed)
		else:
			train_loader, val_loader, test_loader = make_dataloaders_from_image_folder(Path(args.image_root), batch_size=batch_size, train_fraction=train_fraction, seed=args.seed)

		# Plot sample images once
		if idx == 1:
			show_samples_from_loader(train_loader, [class_names_map()[i] for i in range(10)], save_path=Path("results/plots/samples.png"), num_per_class=3)

		config = TrainConfig(
			batch_size=batch_size,
			lr=lr,
			dropout=dropout,
			weight_decay=weight_decay,
			epochs=args.epochs,
			patience=args.patience,
			device=resolved_device,
			seed=args.seed,
		)
		model, history, best_path = fit(train_loader, val_loader, config, run_name)

		cls_names = class_names_map()
		cm_path = Path("results/plots") / f"cm_{run_name}.png"
		roc_path = Path("results/plots") / f"roc_{run_name}.png"
		cm = plot_confusion_matrix(model, test_loader, cls_names, cm_path)
		per_class = compute_metrics(model, test_loader, cls_names)
		auc_scores = plot_roc_curves(model, test_loader, cls_names, roc_path)

		row = {
			"run_name": run_name,
			"val_best_loss": min(history["val_loss"]),
			"val_best_acc": max(history["val_acc"]),
			**per_class,
			**auc_scores,
		}
		metrics_rows.append(row)

	out_csv = Path("results/metrics.csv")
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	all_keys: List[str] = sorted({k for row in metrics_rows for k in row.keys()})
	with out_csv.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=all_keys)
		writer.writeheader()
		for row in metrics_rows:
			writer.writerow(row)
	print(f"Saved metrics to {out_csv}")


if __name__ == "__main__":
	main()
