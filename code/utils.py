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
