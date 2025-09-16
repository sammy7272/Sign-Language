import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import Counter
import argparse

class SignLanguagePreprocessor:
	def __init__(self, raw_data_path, processed_data_path):
		self.raw_data_path = Path(raw_data_path)
		self.processed_data_path = Path(processed_data_path)
		self.processed_data_path.mkdir(parents=True, exist_ok=True)
		
		# Create results directory for visualization
		self.results_path = Path('results/plots')
		self.results_path.mkdir(parents=True, exist_ok=True)
		
	def load_raw_data(self):
		"""Load the raw sign language digits dataset"""
		print("Loading raw dataset...")
		
		# Load the data (assuming it's in numpy format from Kaggle)
		if (self.raw_data_path / 'X.npy').exists():
			X = np.load(self.raw_data_path / 'X.npy')
			y = np.load(self.raw_data_path / 'Y.npy')
		else:
			raise FileNotFoundError("Raw data files (X.npy, Y.npy) not found in data directory")
		
		# Fix shapes: allow X to be (N,64,64) or (N,1,64,64)
		if X.ndim == 3:
			pass
		elif X.ndim == 4 and X.shape[1] == 1:
			X = X[:, 0, :, :]
		else:
			raise ValueError(f"Unexpected X shape: {X.shape}")
		
		# Convert labels: if one-hot, argmax
		if y.ndim > 1:
			y = y.argmax(axis=-1)
		
		print(f"Dataset shape: X={X.shape}, y={y.shape}")
		print(f"Data type: X={X.dtype}, y={y.dtype}")
		print(f"Pixel value range: [{X.min()}, {X.max()}]")
		
		return X, y
	
	def analyze_dataset(self, X, y):
		"""Analyze the dataset and create visualizations"""
		print("\n=== Dataset Analysis ===")
		
		# Basic statistics
		n_samples, height, width = X.shape
		n_classes = len(np.unique(y))
		
		print(f"Total samples: {n_samples}")
		print(f"Image dimensions: {height}x{width}")
		print(f"Number of classes: {n_classes}")
		print(f"Classes: {sorted(np.unique(y))}")
		
		# Class distribution
		class_counts = Counter(y)
		print(f"Class distribution: {dict(sorted(class_counts.items()))}")
		
		# Plot class distribution
		plt.figure(figsize=(10, 6))
		classes, counts = zip(*sorted(class_counts.items()))
		plt.bar(classes, counts, color='skyblue', edgecolor='black')
		plt.title('Class Distribution in Sign Language Digits Dataset')
		plt.xlabel('Digit Class')
		plt.ylabel('Number of Samples')
		plt.xticks(classes)
		for i, count in enumerate(counts):
			plt.text(classes[i], count + 5, str(count), ha='center')
		plt.tight_layout()
		plt.savefig(self.results_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
		plt.close()
		
		# Show sample images from each class
		self.show_sample_images(X, y)
		
		return {
			'n_samples': n_samples,
			'image_shape': (height, width),
			'n_classes': n_classes,
			'class_distribution': dict(class_counts)
		}
	
	def show_sample_images(self, X, y, samples_per_class=3):
		"""Display sample images from each class"""
		n_classes = len(np.unique(y))
		fig, axes = plt.subplots(n_classes, samples_per_class, 
								figsize=(samples_per_class*2, n_classes*2))
		
		if n_classes == 1:
			axes = axes.reshape(1, -1)
		if samples_per_class == 1:
			axes = axes.reshape(-1, 1)
			
		for digit in range(n_classes):
			# Find indices for this digit
			digit_indices = np.where(y == digit)[0]
			
			# Randomly select samples
			selected_indices = np.random.choice(digit_indices, 
												min(samples_per_class, len(digit_indices)), 
											replace=False)
			
			for i, idx in enumerate(selected_indices):
				axes[digit, i].imshow(X[idx], cmap='gray')
				axes[digit, i].set_title(f'Digit {digit}')
				axes[digit, i].axis('off')
		
		plt.suptitle('Sample Images from Each Class', fontsize=16)
		plt.tight_layout()
		plt.savefig(self.results_path / 'sample_images.png', dpi=300, bbox_inches='tight')
		plt.close()
		print(f"Sample images saved to {self.results_path / 'sample_images.png'}")
	
	def normalize_data(self, X):
		"""Normalize pixel values to [0, 1] range"""
		print("Normalizing data...")
		
		# Convert to float32 if not already
		X = X.astype(np.float32)
		
		# Normalize to [0, 1]
		X_normalized = X / 255.0 if X.max() > 1.0 else X
		
		print(f"Original range: [{X.min():.2f}, {X.max():.2f}]")
		print(f"Normalized range: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
		
		return X_normalized
	
	def split_data(self, X, y, test_size=0.15, val_size=0.15, random_state=42):
		"""Split data into train, validation, and test sets"""
		print(f"Splitting data (70/15/15 train/val/test)...")
		
		# First split: separate test set
		X_temp, X_test, y_temp, y_test = train_test_split(
			X, y, test_size=test_size, random_state=random_state, stratify=y
		)
		
		# Second split: separate train and validation from remaining data
		val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
		X_train, X_val, y_train, y_val = train_test_split(
			X_temp, y_temp, test_size=val_size_adjusted, 
			random_state=random_state, stratify=y_temp
		)
		
		print(f"Train set: {len(X_train)} samples")
		print(f"Validation set: {len(X_val)} samples") 
		print(f"Test set: {len(X_test)} samples")
		
		# Verify class distribution in each split
		train_dist = Counter(y_train)
		val_dist = Counter(y_val)
		test_dist = Counter(y_test)
		
		print("Class distribution per split:")
		for digit in range(len(np.unique(y))):
			print(f"  Digit {digit}: Train={train_dist.get(digit, 0)}, "
				  f"Val={val_dist.get(digit, 0)}, Test={test_dist.get(digit, 0)}")
		
		return (X_train, y_train), (X_val, y_val), (X_test, y_test)
	
	def save_processed_data(self, train_data, val_data, test_data):
		"""Save processed data to disk, including combined processed X.npy/Y.npy"""
		print("Saving processed data...")
		
		X_train, y_train = train_data
		X_val, y_val = val_data
		X_test, y_test = test_data
		
		# Save split arrays
		np.save(self.processed_data_path / 'X_train.npy', X_train)
		np.save(self.processed_data_path / 'y_train.npy', y_train)
		np.save(self.processed_data_path / 'X_val.npy', X_val)
		np.save(self.processed_data_path / 'y_val.npy', y_val)
		np.save(self.processed_data_path / 'X_test.npy', X_test)
		np.save(self.processed_data_path / 'y_test.npy', y_test)
		
		# Also save combined processed arrays compatible with existing loaders
		X_combined = np.concatenate([X_train, X_val, X_test], axis=0)
		y_combined = np.concatenate([y_train, y_val, y_test], axis=0)
		# Ensure shape (N,1,64,64)
		X_save = X_combined[:, None, :, :]
		np.save(self.processed_data_path / 'X.npy', X_save)
		np.save(self.processed_data_path / 'Y.npy', y_combined.astype(np.int64))
		
		print(f"Processed data saved to {self.processed_data_path}")
	
	def process_dataset(self):
		"""Main preprocessing pipeline"""
		print("Starting preprocessing pipeline...")
		
		# Load raw data
		X, y = self.load_raw_data()
		
		# Analyze dataset
		dataset_info = self.analyze_dataset(X, y)
		
		# Normalize data
		X_normalized = self.normalize_data(X)
		
		# Split data
		train_data, val_data, test_data = self.split_data(X_normalized, y)
		
		# Save processed data
		self.save_processed_data(train_data, val_data, test_data)
		
		# Create summary
		summary = {
			'dataset_info': dataset_info,
			'splits': {
				'train_size': len(train_data[0]),
				'val_size': len(val_data[0]),
				'test_size': len(test_data[0])
			},
			'preprocessing_steps': [
				'Normalization to [0, 1]',
				'Stratified train/val/test split',
				'Data saved as numpy arrays (splits and combined)'
			]
		}
		
		print("\n=== Preprocessing Complete ===")
		print(f"Dataset info: {summary['dataset_info']}")
		print(f"Data splits: {summary['splits']}")
		
		return summary

# Custom Dataset class for PyTorch
class SignLanguageDataset(Dataset):
	def __init__(self, X, y, transform=None):
		self.X = X
		self.y = y
		self.transform = transform
	
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, idx):
		image = self.X[idx]
		label = self.y[idx]
		
		# Convert numpy array to PIL Image for transforms
		if self.transform:
			# Convert to PIL Image (expects 0-255 range for some transforms)
			image_pil = Image.fromarray((image * 255).astype(np.uint8))
			image = self.transform(image_pil)
		else:
			# Convert to tensor
			image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension
		
		return image, torch.LongTensor([label]).squeeze()

def get_default_transforms():
	"""Get default transforms for training and validation"""
	
	train_transform = transforms.Compose([
		transforms.ToTensor(),
		# Add data augmentation for training
		transforms.RandomRotation(degrees=10),
		transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
	])
	
	val_transform = transforms.Compose([
		transforms.ToTensor(),
	])
	
	return train_transform, val_transform

def load_processed_data(processed_data_path):
	"""Load processed data for training"""
	processed_path = Path(processed_data_path)
	
	X_train = np.load(processed_path / 'X_train.npy')
	y_train = np.load(processed_path / 'y_train.npy')
	X_val = np.load(processed_path / 'X_val.npy')
	y_val = np.load(processed_path / 'y_val.npy')
	X_test = np.load(processed_path / 'X_test.npy')
	y_test = np.load(processed_path / 'y_test.npy')
	
	return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Preprocess sign language digits dataset')
	parser.add_argument('--raw_dir', type=str, default='..\\data', help='Directory containing X.npy and Y.npy')
	parser.add_argument('--out_dir', type=str, default='data\\processed', help='Output directory for processed files')
	return parser.parse_args()

def main():
	"""Main preprocessing function"""
	# Set random seed for reproducibility
	np.random.seed(42)
	args = parse_args()
	
	# Initialize preprocessor
	raw_data_path = Path(args.raw_dir)
	processed_data_path = Path(args.out_dir)
	
	preprocessor = SignLanguagePreprocessor(raw_data_path, processed_data_path)
	
	# Run preprocessing pipeline
	summary = preprocessor.process_dataset()
	
	print("\nPreprocessing completed successfully!")
	print("You can now use the processed data for training your CNN.")

if __name__ == '__main__':
	main()