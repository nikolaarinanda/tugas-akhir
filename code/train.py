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
from torch.utils.data import DataLoader
from torch.optim import AdamW
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
from datetime import datetime

# Import local modules
from data_reader import CyberbullyingDataset
from model import CNNSentimentClassifier, BERTSentimentClassifier, TextCNN, SimpleLightweightTextCNN
from utils import get_device, to_device, prepare_model, prepare_batch, get_gpu_memory, EarlyStopping

# =====KONSTANTA=====
DATASET_PATH = '../dataset/Dataset-Research.csv'
MODEL_OUTPUT_PATH = 'model_outputs'

SEED = 29082002 # Seed untuk reproducibility

N_FOLDS = 5 # Jumlah fold untuk cross-validation
MAX_LENGTH = 125 # Panjang maksimum kata dalam dataset
VOCAB_SIZE = 40000 # Ukuran kosakata maksimum
# [0.2, 0.5]
DROPUOUT_RATE = 0.1 # Tingkat dropout
# [8, 16, 32]
BATCH_SIZE = 16 # Ukuran batch
# [15, 30]
EPOCHS = 3 # Jumlah epoch
# [1e-4, 5e-3]
LEARNING_RATE = 1e-3 # Tingkat pembelajaran
# 64, 256
EMBEDDING_DIM = 128 # Dimensi embedding untuk CNN
TOKENIZER_NAME = 'indobenchmark/indobert-base-p1' # Nama tokenizer BERT
NUM_CLASSES = 2 # Jumlah kelas untuk klasifikasi
# 64, 128, 256
NUM_FILTERS = 100 # Jumlah filter untuk CNN
# [2, 3, 4], [3, 5, 7]
KERNEL_SIZE = [3, 4, 5] # Ukuran kernel untuk CNN
OUT_CHANNELS = 50 # Jumlah channel output untuk CNN
PADDING_IDX = 0 # Indeks padding untuk embedding

BERT_MODEL_NAME = 'bert-base-uncased' # Nama model BERT

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train BERT for sentiment analysis')

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
    parser.add_argument('--bert_model', type=str, default=BERT_MODEL_NAME,
                        help='Pre-trained BERT model name')
    parser.add_argument('--dropout', type=float, default=DROPUOUT_RATE,
                        help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=VOCAB_SIZE,
                        help='Vocabulary size for embedding')
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
    parser.add_argument('--padding_idx', type=int, default=PADDING_IDX,
                        help='Padding index for embedding')

    # Training parameters
    parser.add_argument('--n_folds', type=int, default=N_FOLDS,
                        help='Fold number for cross-validation')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
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
        max_length=args.max_length
    ) # Membuat fold train dataset

    val_dataset = CyberbullyingDataset(
        file_path=args.file_path,
        tokenizer_name=args.tokenizer,
        random_state=args.seed,
        split="val",
        n_folds=args.n_folds,
        max_length=args.max_length
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

    # Initialize model and move to device

    # model = CNNSentimentClassifier(
    #     vocab_size=args.vocab_size,
    #     embed_dim=args.embed_dim,
    #     kernel_size=args.kernel_size,
    #     num_filters=args.num_filters,
    #     dropout_rate=args.dropout,
    # )

    model = TextCNN(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        kernel_sizes=args.kernel_size,
        num_filters=args.num_filters,
        dropout_rate=args.dropout,
        num_classes=args.num_classes,
    )

    # model = SimpleLightweightTextCNN(
    #     vocab_size=args.vocab_size,
    #     embed_dim=args.embed_dim,
    #     num_classes=args.num_classes,
    #     out_channels=args.out_channels,
    #     kernel_size=args.kernel_size,
    #     dropout_rate=args.dropout,
    #     padding_idx=args.padding_idx
    # )

    model = prepare_model(model, device)

    optimizer = AdamW(model.parameters(), lr=args.lr) 
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training loop
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            # Move batch to device
            batch = prepare_batch(batch, device)
            
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
                batch = prepare_batch(batch, device)
                
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
                "epoch": epoch + 1
            })
        
        # Print results
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    
    # Save model if required
    if args.output_model:
        model_save_path = os.path.join(output_dir, f"fold_{args.fold+1}_model.pth")
        torch.save(model.state_dict(), model_save_path)

    return model

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Cek wandb status
    if args.use_wandb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = wandb.init(
            project="my-awesome-project",
            name=f"exp_{timestamp}",
            # Track hyperparameters and metadata
            config=vars(args),
            # config={
            #     "learning_rate": 0.01,
            #     "epochs": 10,
            # },
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