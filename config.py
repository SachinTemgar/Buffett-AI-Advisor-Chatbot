# CONFIG: All keys and paths go here

import streamlit as st
import os

# Groq API Key for Llama 3.1
try:
    # Try to get from Streamlit secrets first (deployed app)
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
except:
    # Fall back to environment variable (local dev)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# Model paths 
TOKENIZER_PATH = None