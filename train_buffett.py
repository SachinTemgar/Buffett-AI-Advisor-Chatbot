import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import os
import re
import json
from collections import Counter, defaultdict
from tqdm import tqdm

# BPE TOKENIZER 

class BPETokenizer:
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = {}  # Stores BPE merge rules
        self.vocab = {}   # Token to index
        self.inverse_vocab = {}  # Index to token
        
        # PROBLEM 3 FIX: Special tokens for stopping
        # <EOS> tells model when to stop generating
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3,  # End of sequence - FIXES RAMBLING
        }
    
    def _get_word_frequencies(self, texts):
        """Count word frequencies in corpus"""
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            for word in words:
                # Add end-of-word marker to track word boundaries
                word = ' '.join(list(word)) + ' </w>'
                word_freq[word] += 1
        return word_freq
    
    def _get_pair_frequencies(self, word_freq):
        """Count frequency of adjacent character pairs"""
        pairs = defaultdict(int)
        for word, freq in word_freq.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _merge_pair(self, pair, word_freq):
        """Merge most frequent pair in vocabulary"""
        new_word_freq = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freq.items():
            new_word = word.replace(bigram, replacement)
            new_word_freq[new_word] = freq
        
        return new_word_freq
    
    def fit(self, texts):
        """Train BPE tokenizer on texts"""
        print("Training BPE tokenizer...")
        
        # Initialize vocabulary with special tokens
        self.vocab = dict(self.special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Get word frequencies
        word_freq = self._get_word_frequencies(texts)
        
        # Add all characters to vocabulary
        all_chars = set()
        for word in word_freq.keys():
            all_chars.update(word.split())
        
        for char in all_chars:
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in tqdm(range(num_merges), desc="BPE Merges"):
            pairs = self._get_pair_frequencies(word_freq)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            word_freq = self._merge_pair(best_pair, word_freq)
            
            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[merged_token] = idx
                self.inverse_vocab[idx] = merged_token
            
            # Store merge rule
            self.merges[best_pair] = merged_token
        
        print(f"BPE Vocabulary size: {len(self.vocab)}")
    
    def _tokenize_word(self, word):
        """Tokenize a single word using learned BPE merges"""
        word = ' '.join(list(word)) + ' </w>'
        symbols = word.split()
        
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            valid_pairs = [(p, self.merges[p]) for p in pairs if p in self.merges]
            
            if not valid_pairs:
                break
            
            # Apply first valid merge
            pair_to_merge = valid_pairs[0][0]
            merged = ''.join(pair_to_merge)
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair_to_merge:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        
        return symbols
    
    def encode(self, text):
        """Encode text to token indices"""
        text = text.lower()
        words = text.split()
        tokens = []
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab["<UNK>"])
        
        return tokens
    
    def decode(self, indices):
        """Decode token indices back to text"""
        tokens = []
        for idx in indices:
            if idx in self.inverse_vocab:
                token = self.inverse_vocab[idx]
                # Skip special tokens in output
                if token not in ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]:
                    tokens.append(token)
        
        # Join and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def save(self, path):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'merges': {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.merges = {tuple(k.split('|||')): v for k, v in data['merges'].items()}
        self.vocab_size = data['vocab_size']


# DATASET 

class BuffettDataset(Dataset):
    
    def __init__(self, texts, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = self._prepare_data(texts)
    
    def _prepare_data(self, texts):
        all_tokens = []
        eos_token = self.tokenizer.vocab.get("<EOS>", 3)
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            tokens.append(eos_token)  # PROBLEM 3 FIX: Add <EOS> after each text
            all_tokens.extend(tokens)
        
        sequences = []
        for i in range(0, len(all_tokens) - self.seq_length - 1, self.seq_length // 2):
            input_seq = all_tokens[i:i + self.seq_length]
            target_seq = all_tokens[i + 1:i + self.seq_length + 1]
            sequences.append((input_seq, target_seq))
        
        print(f"Created {len(sequences)} training sequences")
        return sequences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)

# TRANSFORMER MODEL

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.qkv_linear = nn.Linear(d_model, 3 * d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.depth)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.reshape(batch_size, seq_len, self.d_model)
        
        return self.output_linear(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class BuffettGPT(nn.Module):

    
    def __init__(self, vocab_size, d_model=256, num_heads=8, d_ff=512, 
                 num_layers=6, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        
        mask = self.create_causal_mask(seq_len, x.device)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.output_linear(x)
        
        return logits
    
    
    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_length=100, temperature=0.8, 
                 top_k=50, top_p=0.9, repetition_penalty=1.2, 
                 no_repeat_ngram_size=3, device='cpu'):

        self.eval()
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens).unsqueeze(0).to(device)
        
        # Get special token IDs
        eos_token_id = tokenizer.vocab.get("<EOS>", 3)
        pad_token_id = tokenizer.vocab.get("<PAD>", 0)
        
        # Track generated tokens for n-gram blocking
        generated_ngrams = set()
        
        for step in range(max_length):
            if tokens.size(1) >= self.max_seq_len:
                tokens = tokens[:, -self.max_seq_len:]
            
            logits = self(tokens)
            logits = logits[:, -1, :]
            
    
            if repetition_penalty != 1.0:
                for token_id in set(tokens[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
            

            if no_repeat_ngram_size > 0 and tokens.size(1) >= no_repeat_ngram_size:
                # Get the last (n-1) tokens
                prev_tokens = tuple(tokens[0, -(no_repeat_ngram_size-1):].tolist())
                
                # Block any token that would create a repeated n-gram
                for token_id in range(logits.size(-1)):
                    potential_ngram = prev_tokens + (token_id,)
                    if potential_ngram in generated_ngrams:
                        logits[0, token_id] = float('-inf')
            
            # Apply temperature
            logits = logits / temperature
            

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
    
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == eos_token_id:
                break
            
            # Update n-gram tracking
            if no_repeat_ngram_size > 0 and tokens.size(1) >= no_repeat_ngram_size - 1:
                new_ngram = tuple(tokens[0, -(no_repeat_ngram_size-1):].tolist()) + (next_token.item(),)
                generated_ngrams.add(new_ngram)
            
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokenizer.decode(tokens[0].tolist())


# TRAINING


def train(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def load_letters(directory):
    """Load all text files from a directory"""
    texts = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
            print(f"Loaded: {filename}")
    return texts


def load_qa_data(filepath):
    """Load Q&A data from Excel file"""
    import pandas as pd
    
    # Read Excel file (first row is header)
    df = pd.read_excel(filepath)
    
    print(f"Columns found: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    print(f"Sample Q: {df.iloc[0]['question'][:50]}...")
    print(f"Sample A: {df.iloc[0]['answer'][:50]}...")
    
    # Format as "Question: ... Answer: ..." for training
    qa_texts = []
    for _, row in df.iterrows():
        question = str(row['question']).strip()
        answer = str(row['answer']).strip()
        
        # Skip if empty
        if not question or not answer or question == 'nan' or answer == 'nan':
            continue
        
        # Format as Q&A
        qa_text = f"Question: {question} Answer: {answer}"
        qa_texts.append(qa_text)
    
    print(f"Loaded {len(qa_texts)} valid Q&A pairs")
    return qa_texts



# MAIN


def main():

    
    config = {
        'data_dir': './buffett_letters',
        'qa_file': './warren_buffett_qa_augmented.xlsx',
        'vocab_size': 5000,
        'd_model': 384,
        'num_heads': 8,
        'd_ff': 1536,
        'num_layers': 8,
        'max_seq_len': 256,
        'dropout': 0.1,
        'seq_length': 192,
        'batch_size': 16,
        'epochs': 30,              
        'learning_rate': 3e-4,
        'warmup_steps': 1000,
        'save_dir': './checkpoints_v4'  
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load letters
    print("\n📚 Loading Buffett's letters...")
    letter_texts = load_letters(config['data_dir'])
    print(f"Loaded {len(letter_texts)} letters")
    
    # Load Q&A data
    print("\n❓ Loading Q&A data...")
    qa_texts = load_qa_data(config['qa_file'])
    print(f"Loaded {len(qa_texts)} Q&A pairs")
    
    # IMPROVEMENT: Give 3x weight to Q&A data (cleaner, more structured)
    # This helps model learn Q&A format better
    qa_texts_weighted = qa_texts * 3
    print(f"Q&A weighted 3x: {len(qa_texts_weighted)} samples")
    
    # Combine all texts (more Q&A than letters now)
    texts = letter_texts + qa_texts_weighted
    print(f"\n📊 Total training texts: {len(texts)}")
    
    print("\n🔤 Training BPE tokenizer (FIXES <UNK> PROBLEM)...")
    tokenizer = BPETokenizer(vocab_size=config['vocab_size'])
    tokenizer.fit(texts)
    tokenizer.save(os.path.join(config['save_dir'], 'tokenizer.json'))
    
    # Dataset
    print("\n📊 Creating dataset...")
    dataset = BuffettDataset(texts, tokenizer, seq_length=config['seq_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    
    # Model
    print("\n🤖 Initializing model...")
    model = BuffettGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    
    total_steps = len(dataloader) * config['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['learning_rate'], total_steps=total_steps, pct_start=0.1
    )
    
    # Training loop
    print("\n🚀 Starting training...")
    print(f"   Model improvements:")
    print(f"   - BPE Tokenizer (fixes <UNK>)")
    print(f"   - Larger model: {config['d_model']}d, {config['num_layers']} layers")
    print(f"   - <EOS> token (fixes rambling)")
    print()
    
    best_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        loss = train(model, dataloader, optimizer, scheduler, criterion, device, epoch)
        print(f"Epoch {epoch} | Average Loss: {loss:.4f}")
        
        # Save checkpoint
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pt'))
            print(f"💾 Saved best model (loss: {loss:.4f})")
        
        # Generate sample
        if epoch % 5 == 0:
            print("\n📝 Sample generations:")
            prompts = [
                "the key to investing is",
                "Question: What makes a good business? Answer:"
            ]
            for p in prompts:
                # Using improved generation with all fixes
                sample = model.generate(
                    tokenizer, p, 
                    max_length=60,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,     
                    no_repeat_ngram_size=3,      
                    device=device
                )
                print(f"Prompt: '{p}'")
                print(f"Output: '{sample}'\n")
    
    print(" Training complete!")
    print(f"\nFixes applied:")
    print(f"  BPE Tokenizer → No more <UNK>")
    print(f"  Repetition Penalty → Less loops")
    print(f"  N-gram Blocking → No repeated phrases")
    print(f"  Top-p Sampling → More diverse output")
    print(f"  <EOS> Token → Cleaner endings")
    print(f"  Larger Model → Better coherence")


if __name__ == "__main__":
    main()