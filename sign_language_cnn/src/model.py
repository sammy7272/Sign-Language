from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDigitsCNN(nn.Module):
	"""Explainable CNN for 64x64 grayscale digits (10 classes)."""

	def __init__(self, num_classes: int = 10, dropout: float = 0.3):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),  # 32x32

			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),  # 16x16

			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),  # 8x8
		)
		self.dropout = nn.Dropout(p=dropout)
		self.classifier = nn.Sequential(
			nn.Linear(64 * 8 * 8, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(p=dropout),
			nn.Linear(128, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.dropout(x)
		logits = self.classifier(x)
		return logits
