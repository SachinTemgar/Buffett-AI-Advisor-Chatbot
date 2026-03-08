import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os


# BPE TOKENIZER

class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer
    Loads the trained tokenizer from checkpoints
    """
    
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
        """Load tokenizer from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
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


# MODEL ARCHITECTURE

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
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
    """Single Transformer block with attention and feed-forward"""
    
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
    """
    Handcoded GPT-style Transformer for Buffett text generation
    """
    
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
        """Create causal mask for autoregressive generation"""
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
    def generate(self, tokenizer, prompt, max_length=100, temperature=0.5, 
                 top_k=40, top_p=0.85, repetition_penalty=1.3, 
                 no_repeat_ngram_size=3, min_length=15, device='cpu'):
        """
        Generate text with all improvements:
        - Repetition penalty
        - N-gram blocking
        - Top-k and Top-p sampling
        - EOS stopping
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
            
            # Repetition Penalty
            if repetition_penalty != 1.0:
                for token_id in set(tokens[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
            
            # N-gram Blocking
            if no_repeat_ngram_size > 0 and tokens.size(1) >= no_repeat_ngram_size:
                prev_tokens = tuple(tokens[0, -(no_repeat_ngram_size-1):].tolist())
                for token_id in range(logits.size(-1)):
                    potential_ngram = prev_tokens + (token_id,)
                    if potential_ngram in generated_ngrams:
                        logits[0, token_id] = float('-inf')
            
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) sampling
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
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop at EOS
            if next_token.item() == eos_token_id and step >= min_length:
                break
            
            # Update n-gram tracking
            if no_repeat_ngram_size > 0 and tokens.size(1) >= no_repeat_ngram_size - 1:
                new_ngram = tuple(tokens[0, -(no_repeat_ngram_size-1):].tolist()) + (next_token.item(),)
                generated_ngrams.add(new_ngram)
            
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokenizer.decode(tokens[0].tolist())


# CHATBOT CLASS


class BuffettChatbot:
    """
    Warren Buffett AI Chatbot
    Uses the handcoded transformer model
    """
    
    def __init__(self, model_path=None, tokenizer_path=None):
        self.device = 'cpu'
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # =====================================================================
        # MODEL PATHS - Update these to match your file locations
        # =====================================================================
        if model_path is None:
            model_path = '/Users/sachintemgar/Downloads/GEN AI project/checkpoints_v4/best_model.pt'
        if tokenizer_path is None:
            tokenizer_path = '/Users/sachintemgar/Downloads/GEN AI project/checkpoints_v4/tokenizer.json'
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            # Check if files exist
            if not os.path.exists(self.model_path):
                return False, f"Model file not found: {self.model_path}"
            if not os.path.exists(self.tokenizer_path):
                return False, f"Tokenizer file not found: {self.tokenizer_path}"
            
            # Load tokenizer
            self.tokenizer = BPETokenizer()
            self.tokenizer.load(self.tokenizer_path)
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            config = checkpoint['config']
            
            # Initialize model
            self.model = BuffettGPT(
                vocab_size=len(self.tokenizer.vocab),
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                d_ff=config['d_ff'],
                num_layers=config['num_layers'],
                max_seq_len=config['max_seq_len'],
                dropout=0.0  # No dropout during inference
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_loaded = True
            
            return True, f"Model loaded (Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})"
        
        except FileNotFoundError as e:
            return False, f"Model files not found: {str(e)}. Please train the model first."
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def ask(self, question, max_length=150):
        """
        Ask a question and get Buffett's response
        
        Args:
            question: User's question
            max_length: Maximum response length
            
        Returns:
            str: Buffett-style response
        """
        if not self.is_loaded:
            return "Model not loaded. Please load the model first."
        
        # Format as Q&A prompt
        prompt = f"Question: {question} Answer:"
        
        # Generate response
        full_response = self.model.generate(
            self.tokenizer,
            prompt,
            max_length=max_length,
            temperature=0.5,
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
                    # Capitalize sentences
                    answer = '. '.join(s.strip().capitalize() for s in answer.split('. ') if s.strip())
                    return answer
        
        return full_response
    
    def get_stock_opinion(self, ticker, buffett_score, recommendation):
        """
        Generate Buffett's opinion on a stock based on analysis
        
        Args:
            ticker: Stock symbol
            buffett_score: Score from BuffettAnalyzer (0-100)
            recommendation: BUY/HOLD/AVOID
            
        Returns:
            str: Buffett-style stock opinion
        """
        if not self.is_loaded:
            return self._fallback_opinion(ticker, buffett_score, recommendation)
        
        # Create appropriate prompt based on score
        if buffett_score >= 80:
            prompt = "Question: What do you think about a company with strong fundamentals and durable competitive advantages? Answer:"
        elif buffett_score >= 60:
            prompt = "Question: What advice do you have for investing in a moderately valued stock? Answer:"
        else:
            prompt = "Question: Should I avoid companies with weak fundamentals and high debt? Answer:"
        
        response = self.model.generate(
            self.tokenizer,
            prompt,
            max_length=100,
            temperature=0.5,
            device=self.device
        )
        
        # Extract and clean response
        if "answer:" in response.lower():
            response = response.lower().split("answer:")[-1].strip()
            response = '. '.join(s.strip().capitalize() for s in response.split('. ') if s.strip())
        
        return response
    
    def _fallback_opinion(self, ticker, score, recommendation):
        """Fallback responses when model isn't loaded"""
        if score >= 80:
            return f"Based on the fundamentals, {ticker} shows characteristics I look for: durable competitive advantages and strong financials."
        elif score >= 60:
            return f"{ticker} has some positive qualities, but I'd want a margin of safety before investing."
        else:
            return f"I would be cautious with {ticker}. The fundamentals suggest it may not have a durable competitive advantage."



# SINGLETON INSTANCE FOR STREAMLIT


_chatbot_instance = None

def get_chatbot():
    """Get or create the chatbot instance (singleton pattern)"""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = BuffettChatbot()
    return _chatbot_instance



# TEST FUNCTION


if __name__ == "__main__":
    print("Testing Buffett Chatbot...")
    print("=" * 50)
    
    # Create chatbot
    chatbot = BuffettChatbot()
    
    # Load model
    success, message = chatbot.load_model()
    print(f"Load result: {message}")
    
    if success:
        # Test questions
        test_questions = [
            "Who are you?",
            "What is value investing?",
            "How do you pick stocks?",
        ]
        
        for q in test_questions:
            print(f"\n❓ Question: {q}")
            answer = chatbot.ask(q)
            print(f"🎩 Answer: {answer}")