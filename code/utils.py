import torch
import numpy as np
import torch

class EarlyStopping:
    """Hentikan training lebih awal jika validation loss tidak membaik setelah sejumlah epoch.
    
    Args:
        patience (int): Berapa lama menunggu setelah perbaikan terakhir sebelum berhenti.
        min_delta (float): Perubahan minimum pada kuantitas yang dipantau agar dianggap sebagai perbaikan.
        verbose (bool): Jika True, cetak pesan untuk setiap penurunan validation loss.
        path (str): Path untuk menyimpan checkpoint model terbaik.
    """
    def __init__(self, patience=5, min_delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss # Kita ingin meminimalkan loss, jadi skor yang lebih tinggi (kurang negatif) lebih baik

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} dari {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Menyimpan model ketika validation loss menurun.'''
        if self.verbose:
            print(f'Validation loss menurun ({self.val_loss_min:.6f} --> {val_loss:.6f}). Menyimpan model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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