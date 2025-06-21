import os
import time
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import classification_report
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging

from data_reader import CyberbullyingDataset
from model_bert import BertSentimentClassifier

# Suppress Hugging Face transformers logging
hf_logging.set_verbosity_error() # Mencegah logging verbose dari Hugging Face

# =====KONSTANTA=====
DATASET_PATH = '../dataset/Dataset-Research.csv' # Jalur ke file dataset
MODEL_OUTPUT_PATH = 'model_outputs' # Direktori untuk menyimpan output model
SEED = 29082002 # Seed untuk reproducibility

N_FOLDS = 5 # Jumlah fold untuk cross-validation
MAX_LENGTH = 125 # Panjang maksimum token dalam dataset
VOCAB_SIZE = 40000 # Ukuran kosakata maksimum (lebih relevan untuk non-BERT models)
DROPOUT_RATE = 0.1 # Tingkat dropout
BATCH_SIZE = 16 # Ukuran batch
EPOCHS = 50 # Jumlah epoch
LEARNING_RATE = 2e-5 # Tingkat pembelajaran (disesuaikan untuk fine-tuning BERT)
EMBEDDING_DIM = 128 # Dimensi embedding (untuk model CNN/LSTM/RNN)
TOKENIZER_NAME = 'indobenchmark/indobert-base-p1' # Nama tokenizer BERT
PRETRAINED_MODEL_NAME = 'indobenchmark/indobert-base-p1' # Nama model BERT pretrained
NUM_CLASSES = 2 # Jumlah kelas untuk klasifikasi
NUM_FILTERS = 100 # Jumlah filter untuk CNN
KERNEL_SIZE = [3, 4, 5] # Ukuran kernel untuk CNN
OUT_CHANNELS = 50 # Jumlah channel output untuk CNN
PADDING_IDX = 0 # Indeks padding untuk embedding

# ==================== MOCK UTILS.PY ====================
# Implementasi mock untuk fungsi-fungsi dari utils.py
# Ini agar kode train.py bisa berjalan secara mandiri.
# Dalam implementasi nyata, Anda akan mengimpornya dari file utils.py.

def get_device():
    """Mendapatkan perangkat yang sesuai (GPU jika tersedia, selain itu CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data, device):
    """Memindahkan data ke perangkat yang ditentukan."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

def prepare_model(model, device):
    """Memindahkan model ke perangkat dan mengaktifkan DataParallel jika tersedia beberapa GPU."""
    if torch.cuda.device_count() > 1:
        print(f"Menggunakan {torch.cuda.device_count()} GPU untuk DataParallel.")
        model = torch.nn.DataParallel(model)
    return to_device(model, device)

def prepare_batch(batch, device):
    """Mempersiapkan batch data dengan memindahkan tensor ke perangkat yang ditentukan."""
    prepared_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            prepared_batch[k] = to_device(v, device)
        else:
            prepared_batch[k] = v
    return prepared_batch

def get_gpu_memory():
    """Mendapatkan informasi penggunaan memori CUDA."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3) # GB
        cached = torch.cuda.memory_reserved() / (1024**3) # GB
        return {'allocated': f"{allocated:.2f} GB", 'cached': f"{cached:.2f} GB"}
    return {'allocated': 'N/A', 'cached': 'N/A'}

class EarlyStopping:
    """Menghentikan pelatihan lebih awal jika loss validasi tidak membaik setelah kesabaran tertentu."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): Berapa lama menunggu setelah loss validasi terakhir membaik.
                            Default: 7
            verbose (bool): Jika True, mencetak pesan untuk setiap peningkatan loss validasi.
                            Default: False
            delta (float): Perubahan minimum dalam kuantitas yang dipantau agar memenuhi syarat sebagai peningkatan.
                           Default: 0
            path (str): Jalur untuk checkpoint yang akan disimpan.
                        Default: 'checkpoint.pt'
            trace_func (function): fungsi cetak jejak.
                                   Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Menyimpan model ketika loss validasi menurun.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ==================== LOGIKA UTAMA TRAIN.PY ====================
def parse_args():
    """Menguraikan argumen baris perintah."""
    parser = argparse.ArgumentParser(description='Melatih berbagai model untuk analisis sentimen')

    # Seed dan reproduktibilitas
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed acak untuk reproduktibilitas')
    
    # Parameter data
    parser.add_argument('--file_path', type=str, default=DATASET_PATH, 
                        help='Jalur ke file dataset')
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH,
                        help='Panjang urutan maksimum')
    
    # Parameter model
    parser.add_argument('--model_type', type=str, default='bert',
                        choices=['bert', 'textcnn_light', 'textcnn_medium', 'textcnn_heavy', 'lstm', 'rnn'],
                        help='Jenis model yang akan dilatih')
    parser.add_argument('--tokenizer', type=str, default=TOKENIZER_NAME,
                        help='Nama tokenizer (untuk model non-BERT, atau jika BERT menggunakan tokenizer kustom)')
    parser.add_argument('--pretrained_model_name', type=str, default=PRETRAINED_MODEL_NAME,
                        help='Nama model BERT pretrained (misalnya, indobenchmark/indobert-base-p1)')
    parser.add_argument('--dropout', type=float, default=DROPOUT_RATE,
                        help='Tingkat dropout')
    parser.add_argument('--vocab_size', type=int, default=VOCAB_SIZE,
                        help='Ukuran kosakata untuk embedding (untuk model non-BERT)')
    parser.add_argument('--embed_dim', type=int, default=EMBEDDING_DIM, 
                        help='Dimensi embedding untuk CNN/LSTM/RNN')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, 
                        help='Jumlah kelas')
    parser.add_argument('--num_filters', type=int, nargs='+', default=KERNEL_SIZE, 
                        help='Ukuran kernel untuk CNN (misalnya, 3 4 5)')
    parser.add_argument('--out_channels', type=int, default=OUT_CHANNELS,
                        help='Jumlah saluran output untuk CNN (jika berlaku)')
    parser.add_argument('--padding_idx', type=int, default=PADDING_IDX,
                        help='Indeks padding untuk embedding (untuk model non-BERT)')

    # Parameter pelatihan
    parser.add_argument('--n_folds', type=int, default=N_FOLDS,
                        help='Jumlah fold untuk cross-validation')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Ukuran batch')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Jumlah epoch')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Tingkat pembelajaran')
    
    # Parameter output
    parser.add_argument('--output_model', action='store_true', default=False,
                       help='Simpan model setelah pelatihan')
    
    parser.add_argument('--output_dir', type=str, default=MODEL_OUTPUT_PATH,
                        help='Direktori untuk menyimpan output model')
    
    # Parameter Wandb
    parser.add_argument('--use_wandb', action='store_true',
                       help='Aktifkan logging Weights & Biases')
    
    # Parameter Early Stopping
    parser.add_argument('--patience', type=int, default=5,
                        help='Kesabaran untuk early stopping (epoch yang harus ditunggu setelah tidak ada peningkatan)')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Perubahan minimum pada val_loss yang dianggap sebagai peningkatan untuk early stopping')
    
    return parser.parse_args()

def set_seed(seed):
    """Mengatur seed acak untuk reproduktibilitas."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed acak diatur ke {seed}")

def create_output_dir(base_dir):
    """Membuat direktori output berstempel waktu."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Direktori output dibuat: {output_dir}")
    return output_dir

def get_dataloaders_for_fold(args, fold_idx=0):
    """Membuat dataset dan dataloader pelatihan dan validasi untuk fold yang ditentukan."""
    # Untuk cross-validation KFold yang sebenarnya, CyberbullyingDataset perlu
    # menerima fold_idx dan total n_folds untuk mempartisi data dengan benar.
    # Untuk contoh ini, kita akan menggunakan pembagian train/val sederhana di dalam dataset itu sendiri.
    # `fold_idx` sebagian besar untuk tujuan logging dalam pengaturan mock ini.
    
    print(f"Mempersiapkan data untuk fold {fold_idx + 1}...")
    train_dataset = CyberbullyingDataset(
        file_path=args.file_path,
        tokenizer_name=args.tokenizer,
        random_state=args.seed + fold_idx, # Ubah seed untuk fold berbeda jika pembagian data acak
        split="train",
        n_folds=args.n_folds,
        max_length=args.max_length
    )

    val_dataset = CyberbullyingDataset(
        file_path=args.file_path,
        tokenizer_name=args.tokenizer,
        random_state=args.seed + fold_idx,
        split="val",
        n_folds=args.n_folds,
        max_length=args.max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f"Ukuran dataset pelatihan: {len(train_dataset)}, Ukuran dataset validasi: {len(val_dataset)}")
    return train_loader, val_loader

def initialize_model(args, device):
    """Menginisialisasi model berdasarkan argumen."""
    model = None
    if args.model_type == 'bert':
        print(f"Menginisialisasi model BERT: {args.pretrained_model_name}")
        model = BertSentimentClassifier(
            pretrained_model_name=args.pretrained_model_name, 
            num_classes=args.num_classes
        )
    elif args.model_type.startswith('textcnn'):
        print(f"Menginisialisasi model TextCNN ({args.model_type})")
        if args.model_type == 'textcnn_light':
            model_class = TextCNNLight
        elif args.model_type == 'textcnn_medium':
            model_class = TextCNNMedium
        elif args.model_type == 'textcnn_heavy':
            model_class = TextCNNHeavy
        model = model_class(
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            num_filters=args.num_filters,
            kernel_sizes=args.kernel_size,
            num_classes=args.num_classes,
            dropout_rate=args.dropout,
            padding_idx=args.padding_idx
        )
    elif args.model_type == 'lstm':
        print("Menginisialisasi model SimpleLSTM")
        model = SimpleLSTM(
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            num_classes=args.num_classes
        )
    elif args.model_type == 'rnn':
        print("Menginisialisasi model SimpleRNN")
        model = SimpleRNN(
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            num_classes=args.num_classes
        )
    else:
        raise ValueError(f"Jenis model tidak dikenal: {args.model_type}")
    
    return prepare_model(model, device)

def train_model(args, fold_idx=0, output_dir="none"):
    """Melatih dan mengevaluasi model untuk satu fold."""
    print(f"\n{'='*10} Memulai Pelatihan untuk Fold {fold_idx + 1} {'='*10}")
    
    device = get_device()
    print(f"Menggunakan perangkat: {device}")
    
    train_loader, val_loader = get_dataloaders_for_fold(args, fold_idx)

    model = initialize_model(args, device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr) 
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    early_stopping = EarlyStopping(
        patience=args.patience, 
        verbose=True, 
        delta=args.min_delta, 
        path=os.path.join(output_dir, f"fold_{fold_idx+1}_checkpoint.pt") if args.output_model != "none" else 'checkpoint.pt'
    )

    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': [],
        'epoch_duration': []
    }

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Fold {fold_idx+1}) - Training", leave=False):
            batch = prepare_batch(batch, device)
            
            optimizer.zero_grad()
            if args.model_type == 'bert':
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            else:
                outputs = model(batch['input_ids']) # Untuk CNN/LSTM/RNN
            
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(batch['labels'].cpu().numpy())
            
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * np.mean(np.array(train_preds) == np.array(train_labels))
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Fold {fold_idx+1}) - Validating", leave=False):
                batch = prepare_batch(batch, device)
                
                if args.model_type == 'bert':
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                else:
                    outputs = model(batch['input_ids']) # Untuk CNN/LSTM/RNN
                
                loss = criterion(outputs, batch['labels'])
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * np.mean(np.array(val_preds) == np.array(val_labels))
        
        end_time = time.time()
        epoch_duration = end_time - start_time

        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['epoch_duration'].append(epoch_duration)
        
        if torch.cuda.is_available():
            memory_info = get_gpu_memory()
            print(f"Penggunaan Memori GPU - Dialokasikan: {memory_info['allocated']}, Di-cache: {memory_info['cached']}")
        
        print(f"Epoch {epoch+1}/{args.epochs} - Durasi: {epoch_duration:.2f}s")
        print(f"Loss Pelatihan: {avg_train_loss:.4f}, Akurasi: {train_accuracy:.2f}%")
        print(f"Loss Validasi: {avg_val_loss:.4f}, Akurasi: {val_accuracy:.2f}%")
        
        if args.use_wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1,
                "epoch_duration_s": epoch_duration
            })
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping terpicu.")
            break
    
    # Muat bobot model terbaik yang ditemukan oleh early stopping
    if args.output_model and os.path.exists(early_stopping.path):
        model.load_state_dict(torch.load(early_stopping.path))
        print(f"Model terbaik dimuat dari {early_stopping.path}")
    else:
        print("Checkpoint early stopping tidak dimuat (penyimpanan model dinonaktifkan atau tidak ada checkpoint). Menggunakan bobot epoch terakhir untuk evaluasi akhir.")

    # Evaluasi akhir pada set validasi
    print("\n--- Evaluasi Akhir pada Set Validasi ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validasi Akhir", leave=False):
            batch = prepare_batch(batch, device)
            if args.model_type == 'bert':
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            else:
                outputs = model(batch['input_ids'])
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    print("Laporan Klasifikasi:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Cyberbullying', 'Cyberbullying'], digits=4))

    if args.use_wandb:
        wandb.log({"final_classification_report": classification_report(all_labels, all_preds, output_dict=True)})

    # Membuat plot riwayat pelatihan
    if output_dir != "none":
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Loss Pelatihan')
        plt.plot(history['val_loss'], label='Loss Validasi')
        plt.title(f'Fold {fold_idx+1} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Akurasi Pelatihan')
        plt.plot(history['val_accuracy'], label='Akurasi Validasi')
        plt.title(f'Fold {fold_idx+1} Akurasi')
        plt.xlabel('Epoch')
        plt.ylabel('Akurasi (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"fold_{fold_idx+1}_training_history.png"))
        plt.close() # Tutup plot untuk mengosongkan memori

    return model, history

def main():
    args = parse_args()
    set_seed(args.seed)
    
    if args.use_wandb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb.init(
            project="cyberbullying_sentiment_analysis", # Ganti dengan nama proyek Anda
            name=f"{args.model_type}_exp_{timestamp}",
            config=vars(args), # Log hyperparameter dan metadata
        )
        print("Logging Wandb diaktifkan.")

    output_dir = "none"
    if args.output_model:
        output_dir = create_output_dir(args.output_dir)
        # Simpan argumen ke file JSON di direktori output
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        print(f"Argumen disimpan ke {os.path.join(output_dir, 'args.json')}")

    all_fold_histories = []
    final_models = []

    # Jalankan pelatihan untuk N_FOLDS
    for i in range(args.n_folds):
        trained_model, history = train_model(args, fold_idx=i, output_dir=output_dir)
        all_fold_histories.append(history)
        final_models.append(trained_model)
        
    if args.use_wandb:
        wandb.finish()
        print("Logging Wandb selesai.")

if __name__ == "__main__":
    main()
