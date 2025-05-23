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

# Tambahkan fungsi lain yang dibutuhkan di sini