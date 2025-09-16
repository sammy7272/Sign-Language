import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

from src.utils import NumpyDigitsDataset, build_default_transform, show_samples_from_loader
from torch.utils.data import DataLoader, Subset

def main():
	proc = Path('data/processed')
	X_path = proc / 'X.npy'
	Y_path = proc / 'Y.npy'
	if not X_path.exists() or not Y_path.exists():
		print('Processed X.npy/Y.npy not found at data/processed. Run preprocess first.')
		return
	X = np.load(X_path)
	y = np.load(Y_path)
	print('Shapes:', X.shape, y.shape)
	# Checks
	unique = np.unique(y)
	print('Unique labels:', unique)
	if not np.all((unique >= 0) & (unique <= 9)):
		print('WARNING: labels outside 0-9 found')
	counts = Counter(y.tolist())
	print('Class distribution:', dict(sorted(counts.items())))
	# Save class distribution plot
	fig, ax = plt.subplots(figsize=(8,4))
	ax.bar(list(counts.keys()), list(counts.values()))
	ax.set_title('Class distribution (processed)')
	ax.set_xlabel('Digit')
	ax.set_ylabel('Count')
	plt.tight_layout()
	Path('results/plots').mkdir(parents=True, exist_ok=True)
	plt.savefig('results/plots/verify_class_distribution.png', dpi=150)
	plt.close(fig)
	# Sample grid per class
	ds = NumpyDigitsDataset(X, y, transform=build_default_transform())
	idx = list(range(min(400, len(ds))))
	loader = DataLoader(Subset(ds, idx), batch_size=64, shuffle=False)
	show_samples_from_loader(loader, [str(i) for i in range(10)], save_path=Path('results/plots/verify_samples.png'), num_per_class=3)
	# Per-class mean image (visual check)
	means = []
	for c in range(10):
		cls_imgs = X[y == c]
		if cls_imgs.size == 0:
			means.append(np.zeros((64,64), dtype=np.float32))
			continue
		img = cls_imgs.mean(axis=0)
		if img.ndim == 3:  # (1,64,64)
			img = img[0]
		means.append(img)
	fig, axes = plt.subplots(2,5, figsize=(10,4))
	for c in range(10):
		ax = axes[c//5, c%5]
		ax.imshow(means[c], cmap='gray')
		ax.set_title(f'Mean {c}')
		ax.axis('off')
	plt.tight_layout()
	plt.savefig('results/plots/verify_class_means.png', dpi=150)
	plt.close(fig)
	print('Saved: verify_class_distribution.png, verify_samples.png, verify_class_means.png')

if __name__ == '__main__':
	main() 