import torch, torchvision

print(torch.cuda.is_available())

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("torchvision version:", torchvision.__version__)
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))