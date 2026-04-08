import torch
import mediapipe
import torchvision

print('✓ torch:', torch.__version__)
print('✓ mediapipe:', mediapipe.__version__)
print('✓ torchvision:', torchvision.__version__)
print('✓ GPU available:', torch.cuda.is_available())
print()
print("All training packages ready!")
