import sys
import torch
import torchvision

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
try:
	print('cuDNN version:', torch.backends.cudnn.version())
except Exception as e:
	print('cuDNN version: N/A')
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
	print('GPU name 0:', torch.cuda.get_device_name(0))
	# Simple CUDA tensor test
	a = torch.randn(2,3, device='cuda')
	print('CUDA tensor OK:', bool(a.is_cuda))
else:
	print('GPU name 0: N/A')
	print('CUDA tensor OK: N/A (CPU)') 