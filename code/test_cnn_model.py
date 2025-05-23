import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from data_reader import CyberbullyingDataset  # Ubah ini sesuai struktur proyek Anda
from code.model import CNNSentimentClassifier  # Custom model class

def evaluate_model(model_path, test_data_path, tokenizer_name, vocab_size, embed_dim, max_length, device):
    # Muat model
    model = CNNSentimentClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_classes=2,
        kernel_sizes=[3, 4, 5],
        num_filters=100
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Siapkan dataset uji
    test_dataset = CyberbullyingDataset(
        file_path=test_data_path,
        tokenizer_name=tokenizer_name,
        split='test',
        max_length=max_length
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Evaluasi kinerja
    print(classification_report(all_labels, all_preds, target_names=["Non-Cyberbullying", "Cyberbullying"]))

def evaluate_ensemble_model(model_paths, test_data_path, tokenizer_name, vocab_size, embed_dim, max_length, device):
    # Siapkan dataset uji
    test_dataset = CyberbullyingDataset(
        file_path=test_data_path,
        tokenizer_name=tokenizer_name,
        split='test',
        max_length=max_length
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    all_preds = []
    all_labels_collected = False
    all_labels = []
    
    # Simpan semua prediksi dari model-model
    ensemble_outputs = []

    # Loop untuk setiap model
    for model_path in model_paths:
        # Muat model
        model = CNNSentimentClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=2,
            kernel_sizes=[3, 4, 5],
            num_filters=100
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        model_outputs = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                # Kumpulkan semua label sekali
                if not all_labels_collected:
                    all_labels.extend(labels.cpu().numpy())

                outputs = model(inputs)
                model_outputs.append(outputs)

        # Setelah menyelesaikan loop pertama, set flag
        if not all_labels_collected:
            all_labels_collected = True

        # Simpan output dari model ini
        ensemble_outputs.append(torch.cat(model_outputs, dim=0))

    # Rata-rata semua output model (soft voting)
    ensemble_outputs = torch.mean(torch.stack(ensemble_outputs), dim=0)
    _, ensemble_preds = torch.max(ensemble_outputs, 1)

    # Evaluasi kinerja
    print(classification_report(all_labels, ensemble_preds.cpu().numpy(), target_names=["Non-Cyberbullying", "Cyberbullying"]))

# evaluate_model(
#     model_path='model_outputs/run_20250516_101113/fold_5_model.pth',
#     test_data_path='../dataset/Dataset-Research.csv',
#     tokenizer_name='indobenchmark/indobert-base-p1',
#     vocab_size=40000,
#     embed_dim=128,
#     max_length=128,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )

model_paths = [
    'model_outputs/run_20250516_101113/fold_1_model.pth',
    'model_outputs/run_20250516_101113/fold_2_model.pth',
    'model_outputs/run_20250516_101113/fold_3_model.pth',
    'model_outputs/run_20250516_101113/fold_4_model.pth',
    'model_outputs/run_20250516_101113/fold_5_model.pth'
]

evaluate_ensemble_model(
    model_paths=model_paths,
    test_data_path='../dataset/Dataset-Research.csv',
    tokenizer_name='indobenchmark/indobert-base-p1',
    vocab_size=40000,
    embed_dim=128,
    max_length=128,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)