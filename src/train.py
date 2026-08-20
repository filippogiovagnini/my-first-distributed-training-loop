from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch.nn.functional as F

from src.config import DEFAULT_CONFIG
from src.model import GPT

# one could also use this: dataset_name = "tiny_shakespeare"
dataset_name = "roneneldan/TinyStories"
tokenizer_name = "gpt2"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def tokenize_texts(tokenizer, texts, context_window, device):
    encoded = tokenizer(texts, truncation=True, max_length=context_window + 1, padding="max_length", return_tensors="pt")
    return encoded["input_ids"].to(device)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, context_window):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = context_window

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]["text"] if isinstance(self.texts[idx], dict) else self.texts[idx]
        tokens = tokenize_texts(self.tokenizer, [text], self.block_size, device=device).squeeze(0)
        x = tokens[:-1]
        y = tokens[1:]
        return x, y

def train(**config):
    cfg = {**DEFAULT_CONFIG, **config}
    context_window = cfg["context_window"]
    # load dataset
    train_texts = load_dataset(dataset_name, split="train")
    val_texts = load_dataset(dataset_name, split="validation")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # load model and wrap the dataset into a DataLoader
    model = GPT(vocab_size=tokenizer.vocab_size, context_window=context_window, **{k: cfg[k] for k in ("n_layer", "n_heads", "d_model", "dropout", "sliding_window", "attention_sink", "n_kv")}).to(device)
    train_dataset = TextDataset(train_texts, tokenizer, context_window)
    val_dataset = TextDataset(val_texts, tokenizer, context_window)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    val_losses = []
    train_losses = []

    for epoch in range(cfg["n_epochs"]):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits, loss, _ = model(x, targets=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if epoch % 10 == 0:
            model.eval()
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                with torch.no_grad():
                    _, val_loss, _ = model(val_x, targets=val_y)
                val_losses.append(val_loss.item())
            print("latest val_loss:", val_losses[-1])
            model.train()

    path = cfg["path"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)

if __name__ == "__main__":
    train()
