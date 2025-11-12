import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.optim import AdamW
# from muon import SingleDeviceMuonWithAuxAdam
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

# Import local modules
from datareader import CyberbullyingDataset
from model import TextCNNLight
from utils import EarlyStopping

# =====KONSTANTA=====
DATASET_PATH = '../dataset/cyberbullying.csv'
MODEL_OUTPUT_PATH = 'model_outputs'
SEED = 29082002 # Seed untuk reproducibility

# Cross-validation parameters
N_FOLDS = 5 # Jumlah fold untuk cross-validation
MAX_LENGTH = 128 # Panjang maksimum kata dalam dataset

# Training parameters
EPOCHS = 50 # Jumlah epoch 
BATCH_SIZE = 16 # Ukuran batch
LEARNING_RATE = 5e-3 # Tingkat pembelajaran
TOKENIZER_NAME = 'indobenchmark/indobert-base-p1' # Nama tokenizer BERT

# Model parameters for CNN
DROPUOUT_RATE = 0.1 # Tingkat dropout untuk CNN
NUM_CLASSES = 2 # Jumlah kelas klasifikasi untuk CNN
EMBEDDING_DIM = 300 # Dimensi embedding untuk CNN
NUM_FILTERS = 100 # Jumlah filter untuk CNN
KERNEL_SIZE = [2, 4] # Ukuran kernel untuk CNN
OUT_CHANNELS = 50 # Jumlah channel output untuk CNN

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CNN for sentiment analysis')

    # Seed and reproducibility
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed for reproducibility')
    
    # Data parameters
    parser.add_argument('--file_path', type=str, default=DATASET_PATH, 
                        help='Path to dataset file')
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH,
                        help='Maximum sequence length')
    
    # Model parameters
    parser.add_argument('--tokenizer', type=str, default=TOKENIZER_NAME,
                        help='Tokenizer name')
    parser.add_argument('--dropout', type=float, default=DROPUOUT_RATE,
                        help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for embedding')
    parser.add_argument('--embed_dim', type=int, default=EMBEDDING_DIM, 
                        help='Embedding dimension for CNN')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, 
                        help='Number of classes')
    parser.add_argument('--num_filters', type=int, default=NUM_FILTERS, 
                        help='Number of filters for CNN')
    parser.add_argument('--kernel_size', type=int, default=KERNEL_SIZE, 
                        help='Kernel sizes for CNN')
    parser.add_argument('--out_channels', type=int, default=OUT_CHANNELS,
                        help='Number of output channels for CNN')

    # Training parameters
    parser.add_argument('--n_folds', type=int, default=N_FOLDS,
                        help='Fold number for cross-validation')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    
    # Output parameters
    parser.add_argument('--output_model', action='store_true', default=False,
                       help='Save model after training')
    
    parser.add_argument('--output_dir', type=str, default=MODEL_OUTPUT_PATH,
                        help='Directory to save model outputs')

    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    
    # Early Stopping parameters
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping (epochs to wait after no improvement)')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum change in val_loss to be considered an improvement for early stopping')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_output_dir(base_dir):
    """Create timestamped output directory"""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_dataloaders_for_fold(args):
    """Create train and validation datasets/dataloaders for the fold"""
    train_dataset = CyberbullyingDataset(
        file_path=args.file_path,
        tokenizer_name=args.tokenizer,
        random_state=args.seed,
        split="train",
        n_folds=args.n_folds,
        max_length=args.max_length,
    ) # Membuat fold train dataset

    val_dataset = CyberbullyingDataset(
        file_path=args.file_path,
        tokenizer_name=args.tokenizer,
        random_state=args.seed,
        split="val",
        n_folds=args.n_folds,
        max_length=args.max_length,
    ) # Membuat fold val dataset
    
    # Membuat DataLoader
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
    
    return train_loader, val_loader

def cnn_train_fold(args, output_dir="none"):
    print(f"\n{'='*5} Fold {args.n_folds+1} {'='*5}")
    
    # Setup device
    device = get_device()
    
    train_loader, val_loader = get_dataloaders_for_fold(args)

    model = TextCNNLight(
        vocab_size=train_loader.dataset.vocab_size,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        output_dim=args.out_channels,
        dropout_rate=args.dropout
    )

    model = model.to(device)

    # hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
    # other_params = [p for p in model.parameters() if p.ndim < 2]

    # param_groups = [
    #     {"params": hidden_weights, "use_muon": True, "lr": 0.02, "weight_decay": 0.01},
    #     {"params": other_params, "use_muon": False, "lr": 3e-4, "betas": (0.9, 0.95), "weight_decay": 0.01},
    # ]

    optimizer = AdamW(model.parameters(), lr=args.lr) 
    # optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # 🔽 Inisialisasi EarlyStopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training loop
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            # Move batch to device
            batch = to_device(batch, device)
            
            optimizer.zero_grad()
            outputs = model(batch['input_ids'])
            loss = criterion(outputs, batch['labels'])
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch['labels'].size(0)
            train_correct += (predicted == batch['labels']).sum().item()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                # Move batch to device
                batch = to_device(batch, device)
                
                outputs = model(batch['input_ids'])
                loss = criterion(outputs, batch['labels'])
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch['labels'].size(0)
                val_correct += (predicted == batch['labels']).sum().item()
                val_loss += loss.item()

        # Print GPU memory usage if available
        if torch.cuda.is_available():
            memory_info = get_gpu_memory()
            print(f"GPU Memory Usage - Allocated: {memory_info['allocated']}, Cached: {memory_info['cached']}")
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Log metrics
        if args.use_wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr'],
                # "epoch": epoch + 1
            })

        # Early stopping check
        scheduler.step()
        
        # Print results
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # 🔽 Early stopping check
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
    
    # Save model if required
    if args.output_model:
        model_save_path = os.path.join(output_dir, f"fold_{args.fold+1}_model.pth")
        torch.save(model.state_dict(), model_save_path)

    return model

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

def get_gpu_memory():
    """Get GPU memory usage if available"""
    if torch.cuda.is_available():
        return {
            "allocated": f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
            "cached": f"{torch.cuda.memory_reserved()/1e9:.2f} GB"
        }
    return None

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Cek wandb status
    if args.use_wandb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb.init(
            project="my-awesome-project",
            name=f"exp_{timestamp}",
            config=vars(args), # Track hyperparameters and metadata
        )

    # Train untuk 1 fold
    if args.output_model:
        output_dir = create_output_dir(args.output_dir)
        cnn_train_fold(args, output_dir)

    cnn_train_fold(args)
        
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()