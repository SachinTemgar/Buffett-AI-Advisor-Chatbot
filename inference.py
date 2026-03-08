import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math

class BPETokenizer:
    
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}
        self.vocab_size = 0
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3,
        }
    
    def load(self, path):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.inverse_vocab = {int(v) if isinstance(v, str) else v: k for k, v in self.vocab.items()}
        # Fix inverse vocab
        self.inverse_vocab = {}
        for k, v in self.vocab.items():
            self.inverse_vocab[v] = k
        self.merges = {tuple(k.split('|||')): v for k, v in data['merges'].items()}
        self.vocab_size = data['vocab_size']
    
    def _tokenize_word(self, word):
        """Tokenize a single word using learned BPE merges"""
        word = ' '.join(list(word)) + ' </w>'
        symbols = word.split()
        
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            valid_pairs = [(p, self.merges[p]) for p in pairs if p in self.merges]
            
            if not valid_pairs:
                break
            
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
                    tokens.append(self.vocab.get("<UNK>", 1))
        
        return tokens
    
    def decode(self, indices):
        """Decode token indices back to text"""
        tokens = []
        for idx in indices:
            if idx in self.inverse_vocab:
                token = self.inverse_vocab[idx]
                if token not in ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]:
                    tokens.append(token)
        
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

# MODEL CLASSES

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
                 no_repeat_ngram_size=3, min_length=10, device='cpu'):
        """
        Improved generation with all fixes applied
        """
        self.eval()
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens).unsqueeze(0).to(device)
        
        eos_token_id = tokenizer.vocab.get("<EOS>", 3)
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
                prev_tokens = tuple(tokens[0, -(no_repeat_ngram_size-1):].tolist())
                for token_id in range(logits.size(-1)):
                    potential_ngram = prev_tokens + (token_id,)
                    if potential_ngram in generated_ngrams:
                        logits[0, token_id] = float('-inf')
            
            # Temperature
            logits = logits / temperature
            
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            
            if next_token.item() == eos_token_id and step >= min_length:
                break
            
            # Update n-gram tracking
            if no_repeat_ngram_size > 0 and tokens.size(1) >= no_repeat_ngram_size - 1:
                new_ngram = tuple(tokens[0, -(no_repeat_ngram_size-1):].tolist()) + (next_token.item(),)
                generated_ngrams.add(new_ngram)
            
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokenizer.decode(tokens[0].tolist())



# BUFFETT ADVISOR

class BuffettAdvisor:
    """
    Interactive advisor using the improved model
    """
    
    def __init__(self, checkpoint_path, tokenizer_path, device='cpu'):
        self.device = torch.device(device)
        
        # Load BPE tokenizer (PROBLEM 1 FIX)
        print("Loading BPE tokenizer...")
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        
        self.model = BuffettGPT(
            vocab_size=len(self.tokenizer.vocab),
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            max_seq_len=config['max_seq_len'],
            dropout=0.0
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f" Model loaded from epoch {checkpoint['epoch']} (loss: {checkpoint['loss']:.4f})")
        print(f"   Model size: {config['d_model']}d, {config['num_layers']} layers")
        print(f"   Vocabulary: {len(self.tokenizer.vocab)} BPE tokens")
    
    def ask(self, question, max_length=100, temperature=0.5):
        """
        Ask a question - uses all generation improvements
        IMPROVEMENT: Lower temperature (0.5) for more consistent outputs
        """
        prompt = f"Question: {question} Answer:"
        
        # Generation with ALL FIXES applied
        full_response = self.model.generate(
            self.tokenizer, 
            prompt, 
            max_length=max_length,
            temperature=temperature,
            top_k=40,                    
            top_p=0.85,                  
            repetition_penalty=1.3,      
            no_repeat_ngram_size=3,
            min_length=15,               
            device=self.device
        )
        
        # Extract answer part
        if "answer:" in full_response.lower():
            parts = full_response.lower().split("answer:")
            if len(parts) > 1:
                answer = parts[-1].strip()
                if answer:
                    answer = answer[0].upper() + answer[1:]
                return answer
        
        return full_response
    
    def complete(self, prompt, max_length=80, temperature=0.7):
        """
        Text completion mode
        """
        return self.model.generate(
            self.tokenizer, 
            prompt, 
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            device=self.device
        )
    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "="*60)
        print("🎩 ASK WARREN BUFFETT (Improved Model)")
        print("="*60)
        print("\nImprovements applied:")
        print("  ✓ BPE Tokenizer (no more <UNK>)")
        print("  ✓ Repetition Penalty (no loops)")
        print("  ✓ N-gram Blocking (no repeated phrases)")
        print("  ✓ Top-p Sampling (diverse output)")
        print("  ✓ <EOS> Stopping (clean endings)")
        print("\nCommands:")
        print("  • Type a question to ask Buffett")
        print("  • Type 'complete: <text>' for text completion")
        print("  • Type 'quit' to exit")
        print()
        
        example_questions = [
            "What is value investing?",
            "How do you pick stocks?",
            "What makes a good business?",
            "Who are you?",
            "What is your advice for young investors?",
        ]
        print("Example questions:")
        for q in example_questions:
            print(f"  • {q}")
        print()
        
        while True:
            user_input = input("❓ Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n💡 Be fearful when others are greedy!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower().startswith('complete:'):
                prompt = user_input[9:].strip()
                print(f"\n🤔 Completing: \"{prompt}\"\n")
                response = self.complete(prompt)
                print(f"💬 {response}\n")
            else:
                print("\n🤔 Buffett is thinking...\n")
                answer = self.ask(user_input)
                print(f"🎩 Warren Buffett: {answer}\n")
            
            print("-"*50 + "\n")


if __name__ == "__main__":
    print("="*60)
    print("🎩 WARREN BUFFETT AI - Improved Handcoded Transformer")
    print("="*60)
    print("\nLoading model with improvements...")
    
    advisor = BuffettAdvisor(
        checkpoint_path='./checkpoints_v4/best_model.pt',
        tokenizer_path='./checkpoints_v4/tokenizer.json',
        device='cpu'
    )
    
    # Test Q&A
    print("\n" + "="*60)
    print("📝 SAMPLE Q&A (with improvements)")
    print("="*60)
    
    test_questions = [
        "Who are you?",
        "What is value investing?",
        "How do you pick stocks?",
        "What makes a good business?",
    ]
    
    for question in test_questions:
        answer = advisor.ask(question, max_length=80)
        print(f"\n❓ Question: {question}")
        print(f"🎩 Buffett: {answer}")
    
    # Test completions
    print("\n" + "="*60)
    print("📝 SAMPLE COMPLETIONS (with improvements)")
    print("="*60)
    
    completion_prompts = [
        "The key to investing is",
        "A great business has",
    ]
    
    for prompt in completion_prompts:
        response = advisor.complete(prompt, max_length=60)
        print(f"\n📊 Prompt: \"{prompt}\"")
        print(f"💬 {response}")
    
    # Interactive mode
    print("\n")
    advisor.interactive_mode()