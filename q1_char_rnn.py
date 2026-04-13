# Character-level RNN (LSTM)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

text = "hello help hello world"
chars = list(set(text))
char2idx = {ch:i for i,ch in enumerate(chars)}
idx2char = {i:ch for ch,i in char2idx.items()}
data = [char2idx[c] for c in text]

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

model = CharRNN(len(chars))
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

seq_len = 5
for epoch in range(5):
    for i in range(len(data) - seq_len):
        x = torch.tensor(data[i:i+seq_len]).unsqueeze(0)
        y = torch.tensor(data[i+1:i+seq_len+1]).unsqueeze(0)
        out, _ = model(x)
        loss = criterion(out.view(-1, len(chars)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training complete")
