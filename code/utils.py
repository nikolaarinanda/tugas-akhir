import torch
import numpy as np
import torch
import os
from datetime import datetime

class EarlyStoppingEpochs:
    """
    Menghentikan training jika validation loss tidak membaik secara signifikan
    (lebih besar dari min_delta) dalam beberapa epoch berturut-turut (patience).
    
    Args:
        patience (int): Berapa banyak epoch berturut-turut tanpa perbaikan signifikan sebelum stop.
        min_delta (float): Perubahan minimum pada val_loss yang dianggap perbaikan signifikan.
        verbose (bool): Jika True, mencetak log setiap kali model disimpan atau counter naik.
        path (str): Path lokasi penyimpanan model terbaik (berdasarkan val_loss terendah).
    """
    def __init__(self, patience=5, min_delta=0.001, verbose=False, is_save_model=False):
        self.patience = patience                  # Batas jumlah epoch tanpa perbaikan signifikan
        self.min_delta = min_delta                # Nilai perubahan minimum yang dianggap signifikan
        self.verbose = verbose                    # Apakah mencetak log saat update checkpoint/counter
        self.is_save_model = is_save_model        # Apakah menyimpan model terbaik
        self.counter = 0                          # Counter berturut-turut tanpa peningkatan signifikan
        self.best_loss = np.Inf                   # Nilai val_loss terbaik yang pernah dicapai
        self.early_stop = False                   # Status early stop; akan True jika harus berhenti

    def __call__(self, val_loss, model):
        """
        Fungsi yang dipanggil setiap epoch. Mengecek apakah ada perbaikan signifikan.
        Jika tidak, counter bertambah. Jika counter > patience, training dihentikan.
        
        Args:
            val_loss (float): Validation loss pada epoch sekarang
            model (torch.nn.Module): Model saat ini, untuk disimpan jika val_loss membaik
        """
        loss_diff = self.best_loss - val_loss  # Selisih loss dibanding best_loss sebelumnya

        if loss_diff > self.min_delta:
            # Ada perbaikan signifikan
            self.best_loss = val_loss
            if self.is_save_model:
                self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            # Tidak ada perbaikan signifikan
            self.counter += 1
            if self.verbose:
                print(f'No significant improvement. Counter: {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def create_output_dir(base_dir):
        """Create timestamped output directory"""
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir       

    def save_checkpoint(self, val_loss, model):
        """
        Menyimpan model jika val_loss saat ini lebih baik daripada sebelumnya.
        """
        if not os.path.exists('model_outputs'):
            os.makedirs('model_outputs')
        if self.verbose:
            print(f'Validation loss improved to {val_loss:.6f}. Saving model ...')
        torch.save(model.state_dict(), f'model_outputs/{self.path}')

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        # Jumlah epoch berturut-turut tanpa peningkatan signifikan yang masih ditoleransi
        self.patience = patience
        
        # Selisih minimal antara val_loss sebelumnya dan sekarang yang dianggap sebagai perbaikan
        self.min_delta = min_delta
        
        # Counter untuk menghitung berapa kali val_loss tidak membaik secara signifikan
        self.counter = 0
        
        # Menyimpan nilai val_loss terbaik yang pernah dicapai
        self.best_loss = float('inf')  # Awalnya di-set sangat besar

    def __call__(self, val_loss):
        # Cek apakah val_loss sekarang lebih baik dari best_loss sebelumnya dengan selisih signifikan
        if self.best_loss - val_loss > self.min_delta:
            # Jika ya, anggap ini sebagai peningkatan
            self.best_loss = val_loss  # Perbarui best_loss
            self.counter = 0           # Reset counter karena ada peningkatan
        else:
            # Jika tidak ada peningkatan signifikan, tambahkan counter
            self.counter += 1

        # Jika counter melebihi atau sama dengan batas patience, kembalikan True (hentikan training)
        if self.counter >= self.patience:
            return True

        # Jika belum melebihi patience, teruskan training
        return False