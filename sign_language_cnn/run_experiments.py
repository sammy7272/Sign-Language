import argparse
from pathlib import Path
from typing import Dict, List
from itertools import product

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import (
	set_seed,
	make_dataloaders_from_npy,
	show_samples_from_loader,
	get_device,
)
from src.train import TrainConfig, fit
from src.evaluate import plot_confusion_matrix, compute_metrics, plot_roc_curves


def resolved_device_str(device_arg: str) -> str:
	if device_arg == "auto":
		return "cuda" if str(get_device()) == "cuda" else "cpu"
	return device_arg


def class_names_map() -> Dict[int, str]:
	return {i: str(i) for i in range(10)}


def run_single_experiment(npy_dir: Path, cfg: TrainConfig, run_name: str, train_fraction: float, seed: int, light_plots: bool, force_plots: bool = False) -> Dict[str, float]:
	train_loader, val_loader, test_loader = make_dataloaders_from_npy(npy_dir, batch_size=cfg.batch_size, train_fraction=train_fraction, seed=seed)
	model, history, best_path, elapsed = fit(train_loader, val_loader, cfg, run_name)
	cls_names = class_names_map()
	# Keep ROC/CM if not light_plots OR if we explicitly force plots (baseline, combined_best)
	if (not light_plots) or force_plots:
		cm_path = Path("results/plots") / f"cm_{run_name}.png"
		roc_path = Path("results/plots") / f"roc_{run_name}.png"
		plot_confusion_matrix(model, test_loader, cls_names, cm_path)
		plot_roc_curves(model, test_loader, cls_names, roc_path)
	per_class = compute_metrics(model, test_loader, cls_names)
	acc = per_class.get("accuracy", float(max(history["val_acc"])) if "val_acc" in history else None)
	efficiency = float(acc) / max(elapsed, 1e-9)
	row = {
		"run_name": run_name,
		"batch_size": cfg.batch_size,
		"lr": cfg.lr,
		"dropout": cfg.dropout,
		"weight_decay": cfg.weight_decay,
		"epochs": cfg.epochs,
		"patience": cfg.patience,
		"train_fraction": train_fraction,
		"val_best_loss": float(min(history["val_loss"])),
		"val_best_acc": float(max(history["val_acc"])) ,
		"accuracy": float(acc),
		"runtime_sec": float(elapsed),
		"efficiency": efficiency,
	}
	row.update(per_class)
	return row


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
	plt.figure(figsize=(8, 4))
	sns.barplot(data=df, x=x, y=y, color="#4c72b0")
	plt.title(title)
	plt.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=150)
	plt.close()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run hyperparameter experiments")
	parser.add_argument("--npy_dir", type=str, default="data\\processed")
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--light_plots", action="store_true", help="Skip per-run ROC/curves to reduce plot clutter")
	# Baseline
	parser.add_argument("--baseline_bs", type=int, default=32)
	parser.add_argument("--baseline_lr", type=float, default=1e-3)
	parser.add_argument("--baseline_dropout", type=float, default=0.3)
	parser.add_argument("--baseline_patience", type=int, default=10)
	parser.add_argument("--baseline_weight_decay", type=float, default=0.0)
	parser.add_argument("--baseline_epochs", type=int, default=30)
	# Phase sweeps
	parser.add_argument("--batch_sizes", nargs='+', type=int, default=[16, 32, 64, 128])
	parser.add_argument("--lrs", nargs='+', type=float, default=[1e-4, 1e-3, 1e-2, 1e-1])
	parser.add_argument("--dropouts", nargs='+', type=float, default=[0.0, 0.2, 0.3, 0.5, 0.7])
	parser.add_argument("--weight_decays", nargs='+', type=float, default=[0.0, 1e-3, 1e-2, 1e-1])
	parser.add_argument("--patiences", nargs='+', type=int, default=[5, 10, 15, 20])
	parser.add_argument("--epoch_values", nargs='+', type=int, default=[15, 30, 45])
	parser.add_argument("--train_fractions", nargs='+', type=float, default=[0.25, 0.5, 0.75, 1.0])
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	npy_dir = Path(args.npy_dir)
	device_str = resolved_device_str(args.device)
	combined_dir = Path("results/plots/combined")
	combined_dir.mkdir(parents=True, exist_ok=True)

	train_loader, _, _ = make_dataloaders_from_npy(npy_dir, batch_size=32, train_fraction=1.0, seed=args.seed)
	show_samples_from_loader(train_loader, [class_names_map()[i] for i in range(10)], save_path=Path("results/plots/samples_processed.png"), num_per_class=3)

	def cfg(bs, lr, dr, wd, ep, pa) -> TrainConfig:
		return TrainConfig(
			batch_size=bs,
			lr=lr,
			dropout=dr,
			weight_decay=wd,
			epochs=ep,
			patience=pa,
			device=device_str,
			seed=args.seed,
		)

	all_rows: List[Dict[str, float]] = []

	# Baseline: force plots even in light mode
	baseline_cfg = cfg(args.baseline_bs, args.baseline_lr, args.baseline_dropout, args.baseline_weight_decay, args.baseline_epochs, args.baseline_patience)
	all_rows.append(run_single_experiment(npy_dir, baseline_cfg, run_name="baseline", train_fraction=1.0, seed=args.seed, light_plots=args.light_plots, force_plots=True))

	bs_rows = []
	for bs in args.batch_sizes:
		c = cfg(bs, args.baseline_lr, args.baseline_dropout, args.baseline_weight_decay, args.baseline_epochs, args.baseline_patience)
		bs_rows.append(run_single_experiment(npy_dir, c, f"bs_{bs}", 1.0, args.seed, args.light_plots))
	all_rows.extend(bs_rows)
	lr_rows = []
	for lr in args.lrs:
		c = cfg(args.baseline_bs, lr, args.baseline_dropout, args.baseline_weight_decay, args.baseline_epochs, args.baseline_patience)
		lr_rows.append(run_single_experiment(npy_dir, c, f"lr_{lr}", 1.0, args.seed, args.light_plots))
	all_rows.extend(lr_rows)
	do_rows = []
	for dr in args.dropouts:
		c = cfg(args.baseline_bs, args.baseline_lr, dr, args.baseline_weight_decay, args.baseline_epochs, args.baseline_patience)
		do_rows.append(run_single_experiment(npy_dir, c, f"dropout_{dr}", 1.0, args.seed, args.light_plots))
	all_rows.extend(do_rows)
	wd_rows = []
	for wd in args.weight_decays:
		c = cfg(args.baseline_bs, args.baseline_lr, args.baseline_dropout, wd, args.baseline_epochs, args.baseline_patience)
		wd_rows.append(run_single_experiment(npy_dir, c, f"l2_{wd}", 1.0, args.seed, args.light_plots))
	all_rows.extend(wd_rows)
	pa_rows = []
	for pa in args.patiences:
		c = cfg(args.baseline_bs, args.baseline_lr, args.baseline_dropout, args.baseline_weight_decay, args.baseline_epochs, pa)
		pa_rows.append(run_single_experiment(npy_dir, c, f"patience_{pa}", 1.0, args.seed, args.light_plots))
	all_rows.extend(pa_rows)
	ep_rows = []
	for ep in args.epoch_values:
		c = cfg(args.baseline_bs, args.baseline_lr, args.baseline_dropout, args.baseline_weight_decay, ep, args.baseline_patience)
		ep_rows.append(run_single_experiment(npy_dir, c, f"epochs_{ep}", 1.0, args.seed, args.light_plots))
	all_rows.extend(ep_rows)

	def best_value(rows: List[Dict[str, float]], val_key: str) -> float:
		if not rows:
			return None
		best = max(rows, key=lambda r: r.get("val_best_acc", 0.0))
		return float(best[val_key])

	best_bs = best_value(bs_rows, "batch_size") or args.baseline_bs
	best_lr = best_value(lr_rows, "lr") or args.baseline_lr
	best_do = best_value(do_rows, "dropout") or args.baseline_dropout
	best_wd = best_value(wd_rows, "weight_decay") or args.baseline_weight_decay
	best_pa = best_value(pa_rows, "patience") or args.baseline_patience
	best_ep = best_value(ep_rows, "epochs") or args.baseline_epochs

	combined_cfg = cfg(int(best_bs), float(best_lr), float(best_do), float(best_wd), int(best_ep), int(best_pa))
	# Combined-best: also force plots
	all_rows.append(run_single_experiment(npy_dir, combined_cfg, "combined_best", 1.0, args.seed, args.light_plots, force_plots=True))

	dat_rows = []
	for frac in args.train_fractions:
		dat_rows.append(run_single_experiment(npy_dir, combined_cfg, f"data_frac_{frac}", frac, args.seed, args.light_plots))
	all_rows.extend(dat_rows)

	metrics_csv = Path("results/experiments_metrics.csv")
	metrics_csv.parent.mkdir(parents=True, exist_ok=True)
	all_keys: List[str] = sorted({k for r in all_rows for k in r.keys()})
	with metrics_csv.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=all_keys)
		w.writeheader()
		for r in all_rows:
			w.writerow(r)
	print(f"Saved all experiment metrics to {metrics_csv}")

	# Combined comparison figures: for each category, a 1x2 panel (Accuracy vs param, Efficiency vs param)
	df = pd.DataFrame(all_rows)
	def two_panel_plot(sub: pd.DataFrame, x: str, name: str, title_x: str):
		fig, axes = plt.subplots(1, 2, figsize=(12, 4))
		sub_sorted = sub.sort_values(x)
		sns.barplot(data=sub_sorted, x=x, y="accuracy", ax=axes[0], color="#4c72b0")
		axes[0].set_title(f"Accuracy vs {title_x}")
		sns.barplot(data=sub_sorted, x=x, y="efficiency", ax=axes[1], color="#55a868")
		axes[1].set_title(f"Efficiency vs {title_x}")
		for ax in axes:
			for label in ax.get_xticklabels():
				label.set_rotation(45)
		plt.tight_layout()
		out = combined_dir / f"two_panel_{name}.png"
		plt.savefig(out, dpi=150)
		plt.close(fig)

	two_panel_plot(df[df["run_name"].str.startswith("bs_")], "batch_size", "batch_size", "Batch Size")
	two_panel_plot(df[df["run_name"].str.startswith("lr_")], "lr", "lr", "Learning Rate")
	two_panel_plot(df[df["run_name"].str.startswith("dropout_")], "dropout", "dropout", "Dropout")
	two_panel_plot(df[df["run_name"].str.startswith("l2_")], "weight_decay", "l2", "L2 (weight decay)")
	two_panel_plot(df[df["run_name"].str.startswith("patience_")], "patience", "patience", "Patience")
	two_panel_plot(df[df["run_name"].str.startswith("epochs_")], "epochs", "epochs", "Epochs")
	two_panel_plot(df[df["run_name"].str.startswith("data_frac_")], "train_fraction", "data_frac", "Train Fraction")

	# Save best config by efficiency
	best = df.sort_values(["efficiency", "accuracy"], ascending=False).iloc[0]
	best_cfg = {
		"run_name": best["run_name"],
		"batch_size": int(best["batch_size"]),
		"lr": float(best["lr"]),
		"dropout": float(best["dropout"]),
		"weight_decay": float(best["weight_decay"]),
		"epochs": int(best["epochs"]),
		"patience": int(best["patience"]),
		"train_fraction": float(best["train_fraction"]),
		"accuracy": float(best["accuracy"]),
		"runtime_sec": float(best["runtime_sec"]),
		"efficiency": float(best["efficiency"]) ,
	}
	best_json = Path("results/best_config.json")
	import json
	with best_json.open("w") as f:
		json.dump(best_cfg, f, indent=2)
	print(f"Saved best config by efficiency to {best_json}")

	print("Experiment runs completed.")


if __name__ == "__main__":
	main() 