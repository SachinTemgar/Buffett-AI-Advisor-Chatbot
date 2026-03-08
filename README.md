# Warren Buffett AI Investment Advisor

A comprehensive stock analysis platform powered by artificial intelligence that evaluates companies using Warren Buffett's investment philosophy. The system combines a custom-built transformer model with modern AI APIs to deliver investment insights.

## Live Demo

**Try it now:** [buffett-ai-advisor-chatbot.streamlit.app](https://buffett-ai-advisor-chatbot-hvc32jmofypv3qf83bbhwg.streamlit.app/)

## Project Overview

This project was developed as part of a GenAI course focused on practical applications of transformer architectures and retrieval-augmented generation in the financial domain. It demonstrates end-to-end AI system development, from data collection and model training to deployment.

## Key Features

### Stock Analysis Dashboard
- Evaluates any publicly traded company using 15 financial ratios
- Calculates a Buffett Score (0-100) based on value investing principles
- Displays income statements, balance sheets, and cash flow statements
- Provides buy/hold/avoid recommendations with detailed reasoning

### Dual AI Chat System
- **Custom Transformer**: 18M parameter GPT-style model trained from scratch on Buffett's shareholder letters
- **Llama 3.1 Integration**: Meta's 8B parameter model accessed via Groq API for high-quality responses
- Both models answer questions about value investing, financial metrics, and investment strategy

### Real-Time Financial Metrics
The analyzer calculates these key ratios:

**Income Statement (8 ratios)**
- Gross Margin (>40%)
- SG&A Expense Margin (<30%)
- R&D Expense Margin (<30%)
- Depreciation Margin (<10%)
- Interest Expense Margin (<15%)
- Income Tax Rate
- Net Margin (>20%)
- EPS Growth (Positive & Growing)

**Balance Sheet (6 ratios)**
- Cash to Debt Ratio (>1.0)
- Return on Equity (>15%)
- Adjusted Debt to Equity (<0.80)
- Preferred Stock (None)
- Retained Earnings Growth
- Treasury Stock (Should exist)

**Cash Flow (1 ratio)**
- CapEx Margin (<25%)

## Technical Architecture

### Part 1: AI Chatbot System

**Custom Transformer Model**
- Architecture: Decoder-only transformer with 8 layers
- Attention: Multi-head self-attention with 8 heads
- Parameters: ~18 million
- Vocabulary: 5,000 tokens (custom BPE tokenizer)
- Training: 30 epochs on Buffett's letters (1977-2023) and curated Q&A pairs
- Techniques: Causal masking, learned positional embeddings, repetition penalty, n-gram blocking

**Llama 3.1 Integration**
- Model: Meta's 8B parameter open-source LLM
- API: Groq for accelerated inference
- Method: System prompt engineering to create Buffett persona
- Response time: ~0.5-1 second

### Part 2: Financial Analysis Engine

**Data Processing**
- Source: yfinance API with fallback to cached data
- Analysis: Custom ratio calculator based on Buffett's criteria
- Scoring: Weighted algorithm considering all 15 metrics
- Visualization: Interactive Plotly gauges and charts

**Technology Stack**
```
Backend: Python 3.11
Deep Learning: PyTorch 2.10
UI Framework: Streamlit 1.55
Data: Pandas, NumPy
APIs: Groq, yfinance
Vector DB: FAISS (for future RAG features)
Deployment: Streamlit Cloud + Google Drive
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Local Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/buffett-ai-advisor-chatbot.git
cd buffett-ai-advisor-chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure API keys

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Get your free Groq API key at: https://console.groq.com

4. Run the application
```bash
streamlit run ui/dashboard.py
```

The app will open at `http://localhost:8501`

## Project Structure
```
buffett-ai-advisor-chatbot/
├── ui/
│   ├── dashboard.py              # Main Streamlit application
│   └── buffett.jpg               # UI assets
├── data/
│   ├── financial_data.py         # Stock data fetcher
│   └── demo_cache/               # Pre-loaded stock data
├── analysis/
│   ├── buffett_ratios.py         # Financial ratio calculator
│   ├── buffett_chatbot.py        # Custom transformer inference
│   └── llama_advisor.py          # Llama 3.1 integration
├── checkpoints_v4/
│   ├── best_model.pt             # Trained transformer weights (217MB)
│   └── tokenizer.json            # Custom BPE tokenizer
├── config.py                     # Configuration and API keys
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage Guide

### Analyzing a Stock

1. Enter a stock ticker in the sidebar (e.g., AAPL, MSFT, KO)
2. Click "Generate Analysis"
3. Review the Buffett Score and detailed ratio breakdown
4. Check the verdict: Strong Buy, Moderate/Hold, or Avoid/Risky

### Using the Chatbots

**Custom Transformer Chat**
- Best for: Concise responses in Buffett's style
- Ask about: Investment philosophy, stock selection, financial metrics
- Response style: Direct, sometimes terse, trained specifically on Buffett's writing

**Llama 3.1 Chat**
- Best for: Detailed explanations and complex questions
- Ask about: Detailed analysis, comparisons, investment strategies
- Response style: Comprehensive, articulate, context-aware

### Sample Questions

Try asking either chatbot:
- "What is an economic moat?"
- "How do you evaluate management quality?"
- "Should I invest in index funds?"
- "What makes Coca-Cola a good investment?"
- "Explain the margin of safety concept"

## Model Training Details

The custom transformer was trained using:
- Dataset: Warren Buffett's shareholder letters (1977-2023) plus curated Q&A pairs
- Tokenizer: Custom BPE with 5,000 vocabulary trained on Buffett's writing
- Optimization: AdamW with OneCycleLR scheduler
- Training time: ~8 hours on consumer GPU
- Loss function: Cross-entropy with next-token prediction
- Validation: Held-out test set from recent letters

Generation uses multiple techniques to improve quality:
- Repetition penalty to avoid loops
- N-gram blocking to prevent repeated phrases
- Top-k and top-p sampling for diversity
- Temperature control for coherence

## Deployment

The application is deployed on Streamlit Community Cloud with these considerations:

**Model Hosting**
- Large model file (217MB) hosted on Google Drive
- Downloaded on first run and cached
- Tokenizer and config files included in repository

**API Key Management**
- Stored securely in Streamlit Cloud secrets
- Never committed to version control
- Accessed via environment variables

**Performance Optimization**
- Stock data pre-loaded for common tickers
- Model loaded once and cached in session state
- Streamlit caching used for expensive operations

## Limitations

- Custom transformer produces shorter, sometimes less coherent responses compared to Llama 3.1
- Stock data limited to cached companies on cloud deployment (works with any ticker locally)
- Model file size requires Google Drive download on cloud platforms
- Response time varies based on question complexity and model choice

## Future Enhancements

Some ideas for extending this project:
- Add RAG system to retrieve specific sections from Buffett's letters
- Fine-tune larger models (70B+) on investment data
- Implement portfolio tracking and backtesting features
- Add comparative analysis across multiple stocks
- Include news sentiment analysis
- Support for international stocks and markets

## Learning Outcomes

Building this project provided hands-on experience with:
- Implementing transformer architecture from scratch
- Training custom tokenizers (BPE)
- Designing attention mechanisms and positional encodings
- Working with modern LLM APIs
- Financial data processing and analysis
- Full-stack deployment to cloud platforms
- Managing large model files in production

## Acknowledgments

This project draws inspiration from Warren Buffett's investment philosophy as documented in his annual shareholder letters. The financial metrics are based on publicly available investment analysis frameworks.

Special thanks to:
- Meta AI for releasing Llama 3.1 as open source
- Groq for providing fast inference infrastructure
- Streamlit for the excellent deployment platform
- The open-source community for PyTorch and related tools

## Technical References

Key resources used during development:
- "Attention Is All You Need" (Vaswani et al., 2017)
- Warren Buffett's Annual Letters (1977-2023)
- PyTorch documentation for transformer implementation
- Streamlit deployment guides

