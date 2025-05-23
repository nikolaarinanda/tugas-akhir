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
from model import CNNSentimentClassifier, BERTSentimentClassifier, TextCNN
from utils import check_set_gpu

# =====KONSTANTA=====
DATASET_PATH = '../dataset/Dataset-Research.csv'
MODEL_OUTPUT_PATH = 'model_outputs'

SEED = 29082002 # Seed untuk reproducibility

N_FOLDS = 2 # Jumlah fold untuk cross-validation
MAX_LENGTH = 128 # Panjang maksimum kata dalam dataset
VOCAB_SIZE = 40000 # Ukuran kosakata
DROPUOUT_RATE = 0.1 # Tingkat dropout
BATCH_SIZE = 16 # Ukuran batch
EPOCHS = 30 # Jumlah epoch
LEARNING_RATE = 1e-3 # Tingkat pembelajaran
EMBEDDING_DIM = 128 # Dimensi embedding untuk CNN
TOKENIZER_NAME = 'indobenchmark/indobert-base-p1' # Nama tokenizer BERT
NUM_CLASSES = 2 # Jumlah kelas untuk klasifikasi
NUM_FILTERS = 100 # Jumlah filter untuk CNN
KERNEL_SIZES = [3, 4, 5] # Ukuran kernel untuk CNN

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
    parser.add_argument('--kernel_sizes', type=list, default=KERNEL_SIZES, 
                        help='Kernel sizes for CNN')

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

def cnn_train_fold(args, device, output_dir="none"):
    print(f"\n{'='*5} Fold {args.n_folds+1} {'='*5}")
    
    train_loader, val_loader = get_dataloaders_for_fold(args)

    # Inisialisasi model CNN saya
    # model = CNNSentimentClassifier(
    #     vocab_size=args.vocab_size,
    #     embed_dim=args.embed_dim,
    #     num_classes=args.num_classes,
    #     kernel_sizes=args.kernel_sizes,
    #     num_filters=args.num_filters,
    # )

    # Inisialisasi model CNN dari repo
    model = TextCNN(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
    )

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr) 
    criterion = torch.nn.CrossEntropyLoss() 
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device) # Untuk CNN
            optimizer.zero_grad()

            # outputs = model(batch['input_ids'], batch['attention_mask']) # Untuk BERT
            outputs = model(inputs) # Untuk CNN
            loss = criterion(outputs, batch['labels'])
            train_loss += loss.item()

            # Hitung akurasi training
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()
            
        # Average training loss and training accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0 
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                # batch = {key: val.to(device) for key, val in batch.items()} # Untuk BERT
                inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device) # Untuk CNN

                # outputs = model(batch['input_ids'], batch['attention_mask']) # Untuk BERT
                outputs = model(inputs) # Untuk CNN
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Hitung akurasi validasi
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Average validation loss and validation accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
            
        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
                f"train_loss": avg_train_loss,
                f"train_accuracy": train_accuracy,
                f"val_loss": avg_val_loss,
                f"val_accuracy": val_accuracy,
            })

        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}")

    # Saving model
    if args.output_model:
        model_save_path = os.path.join(output_dir, f"fold_{args.fold+1}_model.pth")
        torch.save(model.state_dict(), model_save_path)

def bert_train_fold(args, device, output_dir="none"):
    print(f"\n{'='*5} Fold {args.fold + 1} {'='*5}")
    
    train_loader, val_loader = get_dataloaders_for_fold(args, args.fold)
    
    # Inisialisasi model BERT
    model = BERTSentimentClassifier(
        model_name=args.model_name,
        dropout_rate=args.dropout
    )

    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            
            outputs = model(batch['input_ids'], batch['attention_mask'])
            labels = batch['labels'] # Mendefinisikan labels

            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # Hitung akurasi training
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()
            
        # Average training loss and training accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0 
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                batch = {key: val.to(device) for key, val in batch.items()}
                outputs = model(batch['input_ids'], batch['attention_mask'])
                labels = batch['labels']
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Hitung akurasi validasi
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Average validation loss and validation accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}")
    
    # Saving model
    if args.output_model:
        model_save_path = os.path.join(output_dir, f"fold_{args.fold + 1}_model.pth")
        torch.save(model.state_dict(), model_save_path)

# Tambahkan di awal main()
def check_wandb_status():
    try:
        print("Checking wandb status...")
        print(f"Wandb version: {wandb.__version__}")
        
        # Cek status login
        if wandb.api.api_key is None:
            print("Warning: Not logged in to wandb!")
            return False
        else:
            print("Successfully logged in to wandb")
            return True
            
    except Exception as e:
        print(f"Error checking wandb status: {e}")
        return False

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Cek wandb status
    if args.use_wandb:
        is_wandb_ready = check_wandb_status()
        if not is_wandb_ready:
            print("Wandb is not ready. Exiting...")
            return
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
    
    device = check_set_gpu()

    # Train untuk semua fold
    # for fold in range(5):
    #     if args.output_model:
    #         output_dir = create_output_dir(args.output_dir)
    #         cnn_train_fold(args, device, output_dir)
    #     cnn_train_fold(args, device)

    # Train untuk 1 fold
    if args.output_model:
        output_dir = create_output_dir(args.output_dir)
        cnn_train_fold(args, device, output_dir)
    else:
        cnn_train_fold(args, device)
        
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()