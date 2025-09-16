from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import SimpleDigitsCNN
from .utils import set_seed, get_device, plot_training_curves


@dataclass
class TrainConfig:
	batch_size: int = 32
	lr: float = 1e-3
	dropout: float = 0.3
	weight_decay: float = 1e-3
	epochs: int = 30
	patience: int = 7
	device: str = "auto"  # "cuda"/"cpu"/"auto"
	model_dir: Path = Path("results/models")
	plots_dir: Path = Path("results/plots")
	seed: int = 42


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	for images, labels in loader:
		images = images.to(device)
		labels = labels.to(device)
		optimizer.zero_grad(set_to_none=True)
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * images.size(0)
		_, preds = torch.max(outputs, 1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)
	return running_loss / total, correct / total


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)
			running_loss += loss.item() * images.size(0)
			_, preds = torch.max(outputs, 1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
	return running_loss / total, correct / total


def _sanitize_config_for_save(config: TrainConfig) -> Dict:
	cfg = dict(config.__dict__)
	# Convert Paths to str for safe serialization
	for k in ["model_dir", "plots_dir"]:
		if isinstance(cfg.get(k), Path):
			cfg[k] = str(cfg[k])
	return cfg


def fit(train_loader: DataLoader, val_loader: DataLoader, config: TrainConfig, run_name: str) -> Tuple[nn.Module, Dict[str, List[float]], Path, float]:
	set_seed(config.seed)
	device = get_device() if config.device == "auto" else torch.device(config.device)
	model = SimpleDigitsCNN(num_classes=10, dropout=config.dropout).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

	history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
	best_val_loss = float("inf")
	best_path = config.model_dir / f"best_{run_name}.pt"
	best_path.parent.mkdir(parents=True, exist_ok=True)
	patience_counter = 0
	start_time = time.perf_counter()

	for epoch in range(1, config.epochs + 1):
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)
		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)
		history["train_acc"].append(train_acc)
		history["val_acc"].append(val_acc)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			patience_counter = 0
			ckpt = {"model_state": model.state_dict(), "config": _sanitize_config_for_save(config)}
			torch.save(ckpt, best_path)
		else:
			patience_counter += 1
			if patience_counter >= config.patience:
				break

	plot_training_curves(history, title=run_name, save_path=config.plots_dir / f"curves_{run_name}.png")
	elapsed = time.perf_counter() - start_time
	ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
	model.load_state_dict(ckpt["model_state"])
	return model, history, best_path, float(elapsed)
