import torch
import os

# Print environment variables
print("CUDA_HOME:", os.getenv('CUDA_HOME'))
print("LD_LIBRARY_PATH:", os.getenv('LD_LIBRARY_PATH'))

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print('CUDA_VISIBLE_DEVICES:', os.getenv('CUDA_VISIBLE_DEVICES'))

print("\n\n")

try:
    # Force CUDA initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current Device:", device)
except Exception as e:
    print("CUDA Initialization Error:", str(e))
    
    
for key, value in os.environ.items():
    print(f"{key}: {value}")

#if torch.cuda.is_available():
#  print("\nCUDA Device Details:")
#  for i in range(torch.cuda.device_count()):
#    print(f"Device {i}:")
#    print(f"  Name: {torch.cuda.get_device_name(i)}")
#    print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")