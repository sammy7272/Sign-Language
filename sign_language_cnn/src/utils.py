import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Friend-provided mapping: class 0->9, 1->0, 2->7, 3->6, 4->1, 5->8, 6->4, 7->3, 8->2, 9->5
# We apply this mapping to labels so that label values correspond to the true digit being shown.

def get_label_map() -> np.ndarray:
	return np.array([9, 0, 7, 6, 1, 8, 4, 3, 2, 5], dtype=np.int64)


def apply_label_map(labels: np.ndarray) -> np.ndarray:
	mapping = get_label_map()
	return mapping[labels]


class NumpyDigitsDataset(Dataset):
	"""Dataset for 64x64 grayscale digit images stored as numpy arrays.

	Expects X of shape (N, 64, 64) or (N, 1, 64, 64), and y of shape (N,) or (N,C) one-hot.
	"""

	def __init__(self, X: np.ndarray, y: np.ndarray, transform: Optional[transforms.Compose] = None):
		assert len(X) == len(y), "X and y must have same length"
		if X.ndim == 3:
			X = X[:, None, :, :]
		elif X.ndim == 2:
			raise ValueError("X must be (N,64,64) or (N,1,64,64)")
		# Normalize X to [0,1]
		self.X = X.astype(np.float32) / 255.0 if X.max() > 1.0 else X.astype(np.float32)
		# Handle labels: if one-hot or nested, convert to class indices, then apply mapping
		if y.ndim > 1:
			y_idx = y.argmax(axis=-1).astype(np.int64)
		else:
			y_idx = y.astype(np.int64)
		self.y = apply_label_map(y_idx)
		self.transform = transform

	def __len__(self) -> int:
		return len(self.y)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		img = self.X[idx]
		label = int(self.y[idx])
		img_tensor = torch.from_numpy(img)  # (1,64,64)
		if self.transform is not None:
			img_tensor = self.transform(img_tensor)
		return img_tensor, label


def build_default_transform() -> transforms.Compose:
	return transforms.Compose([
		transforms.ConvertImageDtype(torch.float32),
	])


def load_from_npy(npy_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
	x_path = npy_dir / "X.npy"
	y_path = npy_dir / "Y.npy"
	if not x_path.exists() or not y_path.exists():
		raise FileNotFoundError(f"Expected X.npy and Y.npy in {npy_dir}")
	X = np.load(str(x_path))
	y = np.load(str(y_path))
	return X, y


def preprocess_and_save(raw_dir: Path, out_dir: Path) -> Dict[str, int]:
	"""Normalize X to [0,1], ensure shape (N,1,64,64), convert Y to class indices, save to out_dir.
	Returns a summary dict with counts per class and shapes.
	"""
	X, y = load_from_npy(raw_dir)
	if X.ndim == 3:
		X = X[:, None, :, :]
	elif X.ndim != 4:
		raise ValueError("X must be (N,64,64) or (N,1,64,64) or (N,C,H,W)")
	X = X.astype(np.float32)
	if X.max() > 1.0:
		X = X / 255.0
	if y.ndim > 1:
		y_idx = y.argmax(axis=-1).astype(np.int64)
	else:
		y_idx = y.astype(np.int64)
	# Save
	out_dir.mkdir(parents=True, exist_ok=True)
	np.save(str(out_dir / "X_processed.npy"), X)
	np.save(str(out_dir / "Y_processed.npy"), y_idx)
	# Summary
	unique, counts = np.unique(y_idx, return_counts=True)
	summary = {"num_samples": int(len(y_idx)), "num_classes": int(len(unique))}
	for u, c in zip(unique.tolist(), counts.tolist()):
		summary[f"class_{u}"] = int(c)
	return summary


def load_from_image_folder(root: Path) -> Dataset:
	transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=1),
		transforms.Resize((64, 64)),
		transforms.ToTensor(),  # [0,1]
	])
	return datasets.ImageFolder(root=str(root), transform=transform)


def stratified_split_indices(labels: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
	indices = np.arange(len(labels))
	train_indices, temp_indices, y_train, y_temp = train_test_split(indices, labels, test_size=(1.0 - train_ratio), stratify=labels, random_state=seed)
	relative_test_ratio = test_ratio / (val_ratio + test_ratio)
	val_indices, test_indices, _, _ = train_test_split(temp_indices, y_temp, test_size=relative_test_ratio, stratify=y_temp, random_state=seed)
	return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def make_dataloaders_from_npy(npy_dir: Path, batch_size: int, train_fraction: float = 1.0, seed: int = 42, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
	set_seed(seed)
	X, y = load_from_npy(npy_dir)
	# Convert y to class indices early, then apply mapping for correct semantics
	if y.ndim > 1:
		labels_idx = y.argmax(axis=-1)
	else:
		labels_idx = y
	labels_mapped = apply_label_map(labels_idx.astype(np.int64))
	transform = build_default_transform()
	dataset = NumpyDigitsDataset(X, labels_mapped, transform=transform)
	train_idx, val_idx, test_idx = stratified_split_indices(labels_mapped, 0.7, 0.15, 0.15, seed)
	if train_fraction < 1.0:
		train_idx, _ = train_test_split(train_idx, train_size=train_fraction, random_state=seed, stratify=labels_mapped[train_idx])
	train_set = Subset(dataset, train_idx.tolist())
	val_set = Subset(dataset, val_idx.tolist())
	test_set = Subset(dataset, test_idx.tolist())
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	return train_loader, val_loader, test_loader


def make_dataloaders_from_image_folder(root: Path, batch_size: int, train_fraction: float = 1.0, seed: int = 42, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
	set_seed(seed)
	full_dataset = load_from_image_folder(root)
	labels = np.array(full_dataset.targets)
	train_idx, val_idx, test_idx = stratified_split_indices(labels, 0.7, 0.15, 0.15, seed)
	if train_fraction < 1.0:
		train_idx, _ = train_test_split(train_idx, train_size=train_fraction, random_state=seed, stratify=labels[train_idx])
	train_set = Subset(full_dataset, train_idx.tolist())
	val_set = Subset(full_dataset, val_idx.tolist())
	test_set = Subset(full_dataset, test_idx.tolist())
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	return train_loader, val_loader, test_loader


def show_samples_from_loader(loader: DataLoader, class_names: List[str], save_path: Optional[Path] = None, num_per_class: int = 3, seed: int = 42) -> None:
	set_seed(seed)
	images_by_class: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(class_names))}
	for imgs, labels in loader:
		for img, label in zip(imgs, labels):
			label_i = int(label)
			if len(images_by_class[label_i]) < num_per_class:
				images_by_class[label_i].append(img)
			if all(len(v) >= num_per_class for v in images_by_class.values()):
				break
		else:
			continue
		break

	n_classes = len(class_names)
	fig, axes = plt.subplots(n_classes, num_per_class, figsize=(num_per_class * 2.2, n_classes * 2.2))
	for c in range(n_classes):
		for j in range(num_per_class):
			ax = axes[c, j] if n_classes > 1 else axes[j]
			if j < len(images_by_class[c]):
				img = images_by_class[c][j].squeeze(0).cpu().numpy()
				ax.imshow(img, cmap='gray')
				ax.set_title(f"{class_names[c]}")
				ax.axis('off')
			else:
				ax.axis('off')
	plt.tight_layout()
	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=150)
	plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], title: str, save_path: Optional[Path] = None) -> None:
	fig, axes = plt.subplots(1, 2, figsize=(10, 4))
	axes[0].plot(history['train_loss'], label='train')
	axes[0].plot(history['val_loss'], label='val')
	axes[0].set_title('Loss')
	axes[0].legend()
	axes[1].plot(history['train_acc'], label='train')
	axes[1].plot(history['val_acc'], label='val')
	axes[1].set_title('Accuracy')
	axes[1].legend()
	fig.suptitle(title)
	plt.tight_layout()
	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=150)
	plt.close(fig)
