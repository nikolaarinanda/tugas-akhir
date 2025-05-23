import torch

def check_set_gpu():
    """Memeriksa ketersediaan GPU dan mengatur perangkat untuk PyTorch."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU ditemukan: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Menggunakan CPU")
    return device

def get_device():
    """Get the device to use (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def to_device(data, device):
    """Move data to specified device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data

def prepare_model(model, device):
    """Prepare model for training on specified device"""
    model = model.to(device)
    return model

def prepare_batch(batch, device):
    """Prepare batch for training on specified device"""
    return to_device(batch, device)

def get_gpu_memory():
    """Get GPU memory usage if available"""
    if torch.cuda.is_available():
        return {
            "allocated": f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
            "cached": f"{torch.cuda.memory_reserved()/1e9:.2f} GB"
        }
    return None