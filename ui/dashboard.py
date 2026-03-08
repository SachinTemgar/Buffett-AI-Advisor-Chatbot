import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.financial_data import get_stock_financials, get_company_info
from analysis.buffett_ratios import BuffettAnalyzer
from analysis.buffett_chatbot import get_chatbot, BuffettChatbot
from analysis.llama_advisor import LlamaAdvisor

st.set_page_config(page_title="Buffett Quantitative Analyzer", layout="wide", page_icon="🏛️", initial_sidebar_state="expanded")

# Compact Header
st.markdown("""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                padding: 1.2rem 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
        <h2 style='color: white; margin: 0; font-size: 1.6rem; font-weight: 700;'>
            💼 Warren Buffett AI Investment Advisor
        </h2>
        <p style='color: rgba(255,255,255,0.85); margin-top: 0.4rem; font-size: 0.9rem; margin-bottom: 0;'>
            Stock Analysis • Custom Transformer • AI Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
# CSS STYLING
st.markdown("""
<style>
    .block-container { padding-top: 3rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 600; }

    /* --- Fix Tab Bar --- */
    [data-baseweb="tab-list"] {
        gap: 8px;
        overflow-x: auto;
    }
    [data-baseweb="tab"] {
        white-space: nowrap;
        font-size: 0.9rem;
        padding: 8px 16px;
    }

    /* --- Metric Cards (Stock Analysis) --- */
    .metric-card {
        background-color: #ffffff; border-radius: 8px; padding: 16px;
        margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-left: 5px solid #cbd5e1; transition: transform 0.2s; color: #1E293B;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .pass { border-left-color: #10B981; }
    .fail { border-left-color: #EF4444; }
    .metric-title { font-size: 1.1rem; font-weight: 700; color: #334155; }
    .metric-logic { font-size: 0.85rem; color: #64748B; font-style: italic; }
    .metric-value { font-size: 1.25rem; font-weight: 800; color: #0F172A; }
    .metric-target { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; }

    /* --- Verdict --- */
    .verdict-container {
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 25px;
    }
    .v-buy { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
    .v-hold { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); }
    .v-avoid { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); }
    .v-title { font-size: 1.8rem; font-weight: 800; }
    .v-desc { font-size: 1rem; opacity: 0.9; }

    /* --- Chat Bubbles --- */
    .chat-container {
        max-height: 500px; overflow-y: auto; padding: 10px;
        border: 1px solid #e2e8f0; border-radius: 12px;
        background: #f8fafc; margin-bottom: 15px;
    }
    .chat-msg {
        padding: 12px 16px; border-radius: 16px; margin: 8px 0;
        max-width: 85%; line-height: 1.5; font-size: 0.95rem;
    }
    .chat-user {
        background: linear-gradient(135deg, #3b82f6, #2563eb); color: white;
        margin-left: auto; text-align: right; border-bottom-right-radius: 4px;
    }
    .chat-bot-custom {
        background: linear-gradient(135deg, #1e293b, #0f172a); color: #e5e7eb;
        border-left: 3px solid #f59e0b; border-bottom-left-radius: 4px;
    }
    .chat-bot-llama {
        background: linear-gradient(135deg, #1e293b, #0f172a); color: #e5e7eb;
        border-left: 3px solid #8b5cf6; border-bottom-left-radius: 4px;
    }
    .chat-label {
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; opacity: 0.7; margin-bottom: 4px;
    }
    .chat-time {
        font-size: 0.7rem; opacity: 0.5; margin-top: 4px;
    }

    /* --- Model Info Cards --- */
    .info-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px; padding: 24px; color: #e5e7eb; margin: 10px 0;
    }
    .info-card-custom { border-top: 4px solid #f59e0b; }
    .info-card-llama { border-top: 4px solid #8b5cf6; }
    .info-card h3 { color: #ffffff; margin-top: 0; }
    .info-card .tag {
        display: inline-block; padding: 4px 10px; border-radius: 6px;
        font-size: 0.75rem; font-weight: 700; margin: 2px 4px 2px 0;
    }
    .tag-gold { background: rgba(245,158,11,0.2); color: #f59e0b; }
    .tag-purple { background: rgba(139,92,246,0.2); color: #8b5cf6; }
    .tag-green { background: rgba(16,185,129,0.2); color: #10b981; }
    .tag-blue { background: rgba(59,130,246,0.2); color: #3b82f6; }
    .tag-red { background: rgba(239,68,68,0.2); color: #ef4444; }

    /* --- Comparison Table --- */
    .cmp-table {
        width: 100%; border-collapse: separate; border-spacing: 0;
        border-radius: 12px; overflow: hidden; margin: 20px 0;
    }
    .cmp-table th {
        background: #1e293b; color: #cbd5e1; padding: 14px 18px;
        font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;
    }
    .cmp-table td {
        padding: 12px 18px; border-bottom: 1px solid #e2e8f0;
        color: #334155; font-size: 0.95rem;
    }
    .cmp-table tr:nth-child(even) td { background: #f8fafc; }
    .cmp-table tr:last-child td { border-bottom: none; }
    .cmp-table .label-col { font-weight: 700; color: #1e293b; background: #f1f5f9 !important; width: 200px; }

    /* --- Model Cards --- */
    .model-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px; padding: 20px; margin: 10px 0; color: #e5e7eb;
    }
    .custom-model { border-left: 4px solid #f59e0b; }
    .llama-model { border-left: 4px solid #8b5cf6; }
    .comparison-header {
        font-size: 0.9rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; margin-bottom: 10px;
    }
    .custom-header { color: #f59e0b; }
    .llama-header { color: #8b5cf6; }
</style>
""", unsafe_allow_html=True)

# SUGGESTED QUESTIONS
SUGGESTED_QUESTIONS = [
    "What is value investing?",
    "How do you pick stocks?",
    "What makes a good business?",
    "What is your advice for young investors?",
    "How do you think about risk?",
    "What is a margin of safety?",
    "What is an economic moat?",
    "Should I invest in index funds?",
    "How do you value a company?",
    "What is the circle of competence?",
    "When should you sell a stock?",
    "What do you think about debt?",
]


# HELPER FUNCTIONS
def create_professional_gauge(score):
    bar_color = "#10B981" if score >= 80 else "#F59E0B" if score >= 60 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Buffett Score", 'font': {'size': 20, 'color': "#cbd5e1"}},
        number={'font': {'size': 48, 'weight': 800, 'color': "#FFFFFF"}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': "white"},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "#334155", 'borderwidth': 0,
            'steps': [{'range': [0, 100], 'color': "#1e293b"}],
        }
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def format_currency(val):
    if not isinstance(val, (int, float)): return "N/A"
    if val >= 1e12: return f"${val/1e12:.2f}T"
    if val >= 1e9:  return f"${val/1e9:.2f}B"
    if val >= 1e6:  return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"

def render_chat_history(history, bot_class="chat-bot-custom", bot_icon="🎩"):
    html = '<div class="chat-container">'
    if not history:
        html += '<p style="text-align:center; color:#94a3b8; padding:40px;">No messages yet. Ask a question below!</p>'
    for msg in history:
        if msg['role'] == 'user':
            html += f'''
            <div style="display:flex; justify-content:flex-end;">
                <div class="chat-msg chat-user">
                    <div class="chat-label">🧑 You</div>
                    {msg['content']}
                    <div class="chat-time">{msg.get('time','')}</div>
                </div>
            </div>'''
        else:
            html += f'''
            <div style="display:flex; justify-content:flex-start;">
                <div class="chat-msg {bot_class}">
                    <div class="chat-label">{bot_icon} Warren Buffett</div>
                    {msg['content']}
                    <div class="chat-time">{msg.get('time','')}</div>
                </div>
            </div>'''
    html += '</div>'
    return html

# SESSION STATE

defaults = {
    'custom_chat_history': [],
    'llama_chat_history': [],
    'custom_loaded': False,
    'llama_loaded': False,
    'llama_instance': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# AUTO-LOAD MODELS ON FIRST RUN

if not st.session_state.custom_loaded:
    _chatbot = get_chatbot()
    _success, _msg = _chatbot.load_model()
    st.session_state.custom_loaded = _success
    if not _success:
        st.sidebar.error(f"Custom model failed: {_msg}")

if not st.session_state.llama_loaded:
    _llama = LlamaAdvisor()
    _success, _msg = _llama.load_model()
    if _success:
        st.session_state.llama_loaded = True
        st.session_state.llama_instance = _llama
    else:
        st.sidebar.error(f"Llama failed: {_msg}")

# SIDEBAR

with st.sidebar:
    local_image_path = os.path.join(os.path.dirname(__file__), 'buffett.jpg')
    if os.path.exists(local_image_path):
        st.image(local_image_path, use_container_width=True)

    st.markdown("### 🏛️ Buffett Logic")
    ticker = st.text_input("Stock Symbol", value="AAPL").upper()
    analyze_btn = st.button("Generate Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚡ Model Status")
    st.markdown(f"Custom Transformer: {'✅ Loaded' if st.session_state.custom_loaded else '❌ Failed'}")
    st.markdown(f"Llama 3.1 (Groq): {'✅ Loaded' if st.session_state.llama_loaded else '❌ Failed'}")

    st.markdown("---")
    st.caption("Handcoded Transformer + Llama 3.1")


# MAIN TABS
stock_tab, custom_chat_tab, llama_chat_tab, info_tab = st.tabs([
    "Analysis",
    "Custom Chat",
    "Llama Chat",
    "Model Info",
])

# TAB 1: STOCK ANALYSIS
with stock_tab:
    if analyze_btn and ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            financials = get_stock_financials(ticker)

            if not financials['success']:
                st.error(f"⚠️ Unable to retrieve data for {ticker}")
            else:
                company_info = get_company_info(ticker)
                analyzer = BuffettAnalyzer(financials)
                ratios = analyzer.calculate_all_ratios()
                score = analyzer.get_buffett_score(ratios)

                st.markdown(f"## {company_info['name']} ({ticker})")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Market Cap", format_currency(company_info['market_cap']))
                c2.metric("Sector", company_info['sector'])
                industry = company_info['industry']
                c3.metric("Industry", industry.split(' — ')[0] if ' — ' in industry else industry)
                price = company_info['current_price']
                c4.metric("Current Price", f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A")

                st.markdown("---")

                col_left, col_right = st.columns([1, 2])
                with col_left:
                    st.plotly_chart(create_professional_gauge(score), use_container_width=True)
                with col_right:
                    if score >= 80:
                        vc, vt, vd = "v-buy", "STRONG BUY", "Exhibits durable competitive advantage and stellar financials."
                    elif score >= 60:
                        vc, vt, vd = "v-hold", "MODERATE / HOLD", "Solid fundamentals but requires margin of safety."
                    else:
                        vc, vt, vd = "v-avoid", "AVOID / RISKY", "Fails on core stability and profitability metrics."
                    st.markdown(f'''
                    <div class="verdict-container {vc}">
                        <div class="v-title">{vt}</div>
                        <div class="v-desc">{vd}</div>
                    </div>''', unsafe_allow_html=True)

                t1, t2, t3 = st.tabs(["💰 Income", "🏛️ Balance Sheet", "🔄 Cash Flow"])

                def render_metric(label, data):
                    val, target, thresh = data['value'], data['rule'], data['threshold']
                    passed = False
                    if thresh is not None:
                        if "Preferred" in label: passed = (val == 0)
                        elif "Treasury" in label: passed = (val == 1)
                        elif '>' in target and isinstance(val, (int, float)): passed = val > thresh
                        elif '<' in target and isinstance(val, (int, float)): passed = val < thresh
                    status = "pass" if passed else "fail"
                    icon = "✅" if passed else "⚠️"
                    fmt_val = f"{val:.2%}" if isinstance(val, float) and abs(val) < 5 else str(val)
                    st.markdown(f'''
                    <div class="metric-card {status}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div><div class="metric-title">{label}</div><div class="metric-logic">{data['logic']}</div></div>
                            <div style="text-align:right;"><div class="metric-value">{fmt_val}</div><div class="metric-target">{target}</div></div>
                            <div style="font-size:1.5rem;">{icon}</div>
                        </div>
                    </div>''', unsafe_allow_html=True)

                with t1:
                    for name, data in ratios['income_statement'].items(): render_metric(name, data)
                with t2:
                    for name, data in ratios['balance_sheet'].items(): render_metric(name, data)
                with t3:
                    for name, data in ratios['cash_flow'].items(): render_metric(name, data)
    else:
        st.info("👈 Enter a stock ticker and click 'Generate Analysis'")


# TAB 2: CUSTOM MODEL CHATBOT
with custom_chat_tab:
    st.markdown("## 🎩 Custom Transformer Chat")
    st.caption("18M parameter transformer trained from scratch on Buffett's letters & Q&A")

    if not st.session_state.custom_loaded:
        st.error("❌ Custom model failed to load. Check model/tokenizer paths.")
    else:
        chatbot = get_chatbot()

        def send_custom_message(question):
            timestamp = time.strftime("%I:%M %p")
            st.session_state.custom_chat_history.append({
                'role': 'user', 'content': question, 'time': timestamp
            })
            with st.spinner("Custom model thinking..."):
                start = time.time()
                response = chatbot.ask(question, max_length=150)
                elapsed = time.time() - start
            st.session_state.custom_chat_history.append({
                'role': 'bot', 'content': response, 'time': f"{timestamp} · {elapsed:.1f}s"
            })

        st.markdown(render_chat_history(
            st.session_state.custom_chat_history,
            bot_class="chat-bot-custom", bot_icon="🎩"
        ), unsafe_allow_html=True)

        if not st.session_state.custom_chat_history:
            st.markdown("#### 💡 Suggested Questions")
            suggestion_cols = st.columns(4)
            for i, q in enumerate(SUGGESTED_QUESTIONS[:8]):
                col = suggestion_cols[i % 4]
                if col.button(q, key=f"cq_{i}", use_container_width=True):
                    send_custom_message(q)
                    st.rerun()

        st.markdown("---")
        col_input, col_send = st.columns([5, 1])
        with col_input:
            custom_input = st.text_input(
                "Type your question...", key="custom_chat_input",
                label_visibility="collapsed",
                placeholder="Ask Warren Buffett anything about investing..."
            )
        with col_send:
            send_custom = st.button("Send 🎩", key="send_custom", use_container_width=True, type="primary")

        if send_custom and custom_input:
            send_custom_message(custom_input)
            st.rerun()

        if st.session_state.custom_chat_history:
            if st.button("🗑️ Clear Chat", key="clear_custom"):
                st.session_state.custom_chat_history = []
                st.rerun()

# TAB 3: LLAMA CHATBOT

with llama_chat_tab:
    st.markdown("## 🦙 Llama 3.1 Chat (via Groq)")
    st.caption("Meta's 8B parameter model with Buffett persona, served via Groq")

    if not st.session_state.llama_loaded:
        st.error("❌ Llama failed to load. Check your API key in config.py")
    else:
        llama = st.session_state.llama_instance

        def send_llama_message(question):
            timestamp = time.strftime("%I:%M %p")
            st.session_state.llama_chat_history.append({
                'role': 'user', 'content': question, 'time': timestamp
            })
            with st.spinner("Llama thinking..."):
                start = time.time()
                response = llama.ask(question)
                elapsed = time.time() - start
            st.session_state.llama_chat_history.append({
                'role': 'bot', 'content': response, 'time': f"{timestamp} · {elapsed:.1f}s"
            })

        st.markdown(render_chat_history(
            st.session_state.llama_chat_history,
            bot_class="chat-bot-llama", bot_icon="🦙"
        ), unsafe_allow_html=True)

        if not st.session_state.llama_chat_history:
            st.markdown("#### 💡 Suggested Questions")
            suggestion_cols = st.columns(4)
            for i, q in enumerate(SUGGESTED_QUESTIONS[:8]):
                col = suggestion_cols[i % 4]
                if col.button(q, key=f"lq_{i}", use_container_width=True):
                    send_llama_message(q)
                    st.rerun()

        st.markdown("---")
        col_input, col_send = st.columns([5, 1])
        with col_input:
            llama_input = st.text_input(
                "Type your question...", key="llama_chat_input",
                label_visibility="collapsed",
                placeholder="Ask Warren Buffett anything about investing..."
            )
        with col_send:
            send_llama = st.button("Send 🦙", key="send_llama", use_container_width=True, type="primary")

        if send_llama and llama_input:
            send_llama_message(llama_input)
            st.rerun()

        if st.session_state.llama_chat_history:
            if st.button("🗑️ Clear Chat", key="clear_llama"):
                st.session_state.llama_chat_history = []
                st.rerun()


# TAB 4: MODEL INFO
with info_tab:
    st.markdown("## 📋 Model Comparison")
    st.markdown("*How these two AI models approach being Warren Buffett*")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('''
        <div class="info-card info-card-custom">
            <h3>🎩 Custom Transformer</h3>
            <p style="font-size:0.9rem; opacity:0.8; margin-bottom:16px;">
                A GPT-style transformer built <b>entirely from scratch</b> — every layer of code handwritten,
                from the multi-head attention mechanism to the BPE tokenizer.
            </p>
            <span class="tag tag-gold">18M Parameters</span>
            <span class="tag tag-gold">8 Attention Heads</span>
            <span class="tag tag-gold">8 Layers</span>
            <span class="tag tag-green">384 Embedding Dim</span>
            <span class="tag tag-green">1536 FFN Dim</span>
            <span class="tag tag-blue">5000 BPE Vocab</span>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div class="info-card info-card-llama">
            <h3>🦙 Llama 3.1 (8B)</h3>
            <p style="font-size:0.9rem; opacity:0.8; margin-bottom:16px;">
                Meta's state-of-the-art open-source LLM with <b>8 billion parameters</b>,
                prompted to roleplay as Warren Buffett. Served via Groq for ultra-fast inference.
            </p>
            <span class="tag tag-purple">8B Parameters</span>
            <span class="tag tag-purple">32 Attention Heads</span>
            <span class="tag tag-purple">32 Layers</span>
            <span class="tag tag-green">4096 Embedding Dim</span>
            <span class="tag tag-green">14336 FFN Dim</span>
            <span class="tag tag-blue">128K Vocab</span>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🔍 Feature-by-Feature Comparison")

    st.markdown('''
    <table class="cmp-table">
        <thead>
            <tr>
                <th>Feature</th>
                <th>🎩 Custom Transformer</th>
                <th>🦙 Llama 3.1 (8B)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="label-col">Model Type</td>
                <td>GPT-style Decoder-Only Transformer</td>
                <td>Decoder-Only Transformer (LLaMA architecture)</td>
            </tr>
            <tr>
                <td class="label-col">Parameters</td>
                <td>~18 Million</td>
                <td>~8 Billion (444× larger)</td>
            </tr>
            <tr>
                <td class="label-col">Built From</td>
                <td>100% handcoded from scratch</td>
                <td>Pre-trained by Meta AI on massive compute</td>
            </tr>
            <tr>
                <td class="label-col">Training Data</td>
                <td>Buffett's letters (1977–2023) + Q&A dataset (3× weighted)</td>
                <td>15 trillion tokens from internet, books, code</td>
            </tr>
            <tr>
                <td class="label-col">Tokenizer</td>
                <td>Custom BPE (5,000 vocab) trained on Buffett's writing</td>
                <td>SentencePiece BPE (128,000 vocab)</td>
            </tr>
            <tr>
                <td class="label-col">Architecture</td>
                <td>8 layers, 8 heads, d=384, ff=1536, seq=256</td>
                <td>32 layers, 32 heads, d=4096, ff=14336, ctx=128K</td>
            </tr>
            <tr>
                <td class="label-col">Training Approach</td>
                <td>Next-token prediction, 30 epochs, AdamW + OneCycleLR</td>
                <td>Pre-trained + RLHF. We use system-prompt engineering.</td>
            </tr>
            <tr>
                <td class="label-col">Buffett Persona</td>
                <td>Learned from reading Buffett's actual words</td>
                <td>Instructed via system prompt to roleplay</td>
            </tr>
            <tr>
                <td class="label-col">Inference</td>
                <td>Local CPU/GPU — no API needed</td>
                <td>Cloud API via Groq — requires internet</td>
            </tr>
            <tr>
                <td class="label-col">Response Quality</td>
                <td>Short, stylistic, occasionally incoherent</td>
                <td>Fluent, detailed, highly coherent</td>
            </tr>
            <tr>
                <td class="label-col">Speed</td>
                <td>~1-3 seconds (local)</td>
                <td>~0.5-1 second (Groq accelerated)</td>
            </tr>
            <tr>
                <td class="label-col">Cost</td>
                <td>Free — runs locally</td>
                <td>Free tier available, paid for heavy use</td>
            </tr>
            <tr>
                <td class="label-col">Privacy</td>
                <td>100% private — data stays local</td>
                <td>Data sent to Groq servers</td>
            </tr>
        </tbody>
    </table>
    ''', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🏗️ Architecture Deep Dive")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎩 Custom Transformer")
        st.markdown('''
        Our model is a **decoder-only transformer** built entirely from scratch using PyTorch.

        **Multi-Head Self-Attention** — Fused QKV projection, 8 heads, scaled dot-product
        attention with causal masking.

        **Feed-Forward Network** — Two linear layers with GELU activation (384 → 1536 → 384).

        **Positional Encoding** — Learned embeddings up to 256 tokens.

        **BPE Tokenizer** — Custom 5,000-vocab tokenizer trained on Buffett's writing,
        eliminating the UNK token problem.
        ''')
        st.markdown("**Generation Techniques:**")
        st.markdown('''
        - Repetition Penalty (1.3)
        - N-gram Blocking (3-gram)
        - Top-k (40) + Top-p (0.85) Sampling
        - EOS Token Stopping
        - Temperature (0.5)
        ''')

    with col2:
        st.markdown("#### 🦙 Llama 3.1")
        st.markdown('''
        Meta's **state-of-the-art LLM** representing the cutting edge of open-source AI.

        **Grouped-Query Attention (GQA)** — 32 query heads, 8 KV heads for efficiency.

        **RoPE Positional Encoding** — Rotary embeddings scaling to 128K context.

        **SwiGLU Activation** — Advanced activation for better gradient flow.

        **RMSNorm** — Pre-normalization for stable training at scale.
        ''')
        st.markdown("**How We Use It:**")
        st.markdown('''
        - Detailed Buffett persona system prompt
        - Groq LPU hardware for near-instant responses
        - Temperature (0.7) for natural responses
        - Top-p (0.9) nucleus sampling
        - Pure prompt engineering, no fine-tuning
        ''')

    st.markdown("---")

    st.markdown("### 📊 Custom Model Training Details")

    t1, t2, t3 = st.columns(3)
    t1.metric("Training Epochs", "30")
    t2.metric("Learning Rate", "3e-4")
    t3.metric("Batch Size", "16")

    t4, t5, t6 = st.columns(3)
    t4.metric("Sequence Length", "192 tokens")
    t5.metric("Optimizer", "AdamW")
    t6.metric("Scheduler", "OneCycleLR")

    st.markdown('''
    **Training Data:**
    - Buffett's annual shareholder letters from **1977 to 2023**
    - Curated Q&A dataset (**3× weighted** for emphasis)
    - Custom BPE tokenizer (5,000 merge operations)
    - Overlapping sequences with 50% stride
    ''')

    st.markdown("---")

    st.markdown("### 💡 Key Takeaways")

    st.markdown('''
    <div class="info-card" style="border-top: 4px solid #10b981;">
        <h3 style="color: #10b981;">Why Build a Custom Model?</h3>
        <p style="font-size: 0.95rem;">
            The custom transformer isn't meant to <i>beat</i> Llama 3.1 — it's 444× smaller! The purpose is to demonstrate
            <b>deep understanding of transformer architecture</b> by building every component from scratch: attention mechanisms,
            positional encodings, tokenization, training loops, and generation strategies.
        </p>
        <p style="font-size: 0.95rem;">
            Despite its small size, the model successfully learned Buffett's vocabulary, writing patterns, and key investment
            concepts — proving that even a modest transformer can capture domain-specific knowledge when trained on focused data.
        </p>
        <p style="font-size: 0.95rem;">
            Meanwhile, Llama 3.1 demonstrates how <b>prompt engineering</b> can leverage a general-purpose LLM to create
            convincing domain-specific responses — a fundamentally different but equally valid approach.
        </p>
    </div>
    ''', unsafe_allow_html=True)
