import torch
from torch import nn # import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BERTSentimentClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout_rate=0.3):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)  # dua kelas output

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output['pooler_output']
        dropped_output = self.dropout(pooled_output)
        return self.linear(dropped_output)

class TextCNNLight(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=2):
        super(TextCNNLight, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, 64, kernel_size=4)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = F.relu(self.conv1(x)).max(dim=2)[0]
        x2 = F.relu(self.conv2(x)).max(dim=2)[0]
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        return self.fc(x)

class TextCNNMedium(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNNMedium, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, 100, kernel_size=4)
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=5)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = F.relu(self.conv1(x)).max(dim=2)[0]
        x2 = F.relu(self.conv2(x)).max(dim=2)[0]
        x3 = F.relu(self.conv3(x)).max(dim=2)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        return self.fc(x)

class TextCNNHeavy(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, num_classes=2):
        super(TextCNNHeavy, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=4),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=5),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = self.conv1(x).max(dim=2)[0]
        x2 = self.conv2(x).max(dim=2)[0]
        x3 = self.conv3(x).max(dim=2)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# class SimpleRNN(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim=128, num_classes=2):
#         super(SimpleRNN, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, num_classes)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         output, hidden = self.rnn(embedded)
#         return self.fc(hidden.squeeze(0))
        
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=2, hidden_dim=128, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden[-1, :, :])
    
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers=1, hidden_dim=128, num_classes=2, dropout_rate=0.1):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        # Ambil hidden state dari layer terakhir (hidden[-1, :, :])
        # Jika bidirectional, Anda perlu menggabungkan hidden state dari kedua arah
        # Untuk kasus ini, kita asumsikan unidirectional
        dropped_output = self.dropout(hidden[-1, :, :]) # Mengambil hidden state terakhir dari layer terakhir
        return self.fc(dropped_output)