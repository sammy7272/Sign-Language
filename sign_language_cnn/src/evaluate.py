from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import get_device


def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	model.eval()
	all_logits = []
	all_probs = []
	all_labels = []
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			logits = model(images)
			probs = torch.softmax(logits, dim=1)
			all_logits.append(logits.cpu().numpy())
			all_probs.append(probs.cpu().numpy())
			all_labels.append(labels.numpy())
	logits_np = np.concatenate(all_logits, axis=0)
	probs_np = np.concatenate(all_probs, axis=0)
	labels_np = np.concatenate(all_labels, axis=0)
	return logits_np, probs_np, labels_np


def plot_confusion_matrix(model: nn.Module, loader: DataLoader, class_names: Dict[int, str], save_path: Path) -> np.ndarray:
	device = get_device()
	_, probs, labels = predict_logits(model, loader, device)
	preds = probs.argmax(axis=1)
	cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
	fig, ax = plt.subplots(figsize=(7, 6))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
		xticklabels=[class_names[i] for i in range(len(class_names))],
		yticklabels=[class_names[i] for i in range(len(class_names))])
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_title("Confusion Matrix")
	save_path.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(save_path, dpi=150)
	plt.close(fig)
	return cm


def compute_metrics(model: nn.Module, loader: DataLoader, class_names: Dict[int, str]) -> Dict[str, float]:
	device = get_device()
	_, probs, labels = predict_logits(model, loader, device)
	preds = probs.argmax(axis=1)
	report = classification_report(labels, preds, target_names=[class_names[i] for i in range(len(class_names))], output_dict=True, zero_division=0)
	# Flatten selected metrics
	metrics = {
		"accuracy": report["accuracy"],
	}
	for i in range(len(class_names)):
		cls = class_names[i]
		metrics[f"precision_{cls}"] = report[cls]["precision"]
		metrics[f"recall_{cls}"] = report[cls]["recall"]
		metrics[f"f1_{cls}"] = report[cls]["f1-score"]
	return metrics


def plot_roc_curves(model: nn.Module, loader: DataLoader, class_names: Dict[int, str], save_path: Path) -> Dict[str, float]:
	device = get_device()
	_, probs, labels = predict_logits(model, loader, device)
	n_classes = len(class_names)
	# One-vs-rest ROC
	auc_scores: Dict[str, float] = {}
	fig, ax = plt.subplots(figsize=(7, 6))
	for i in range(n_classes):
		binary_true = (labels == i).astype(int)
		scores = probs[:, i]
		fpr, tpr, _ = roc_curve(binary_true, scores)
		roc_auc = auc(fpr, tpr)
		auc_scores[f"AUC_{class_names[i]}"] = float(roc_auc)
		ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
	ax.plot([0, 1], [0, 1], 'k--', label="Chance")
	ax.set_xlabel("FPR")
	ax.set_ylabel("TPR")
	ax.set_title("One-vs-Rest ROC Curves")
	ax.legend(fontsize=8)
	save_path.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(save_path, dpi=150)
	plt.close(fig)
	return auc_scores
