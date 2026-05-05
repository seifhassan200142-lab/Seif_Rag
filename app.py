import warnings
import os

# Suppress transformers warnings
os.environ["TRANSFORMERS_CACHE"] = ".cache"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchvision.*")

import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="Medical RAG Assistant", page_icon="🩺")

st.title("🩺 Medical RAG Chatbot")


# ✅ استخدام st.cache_resource عشان الـ engine يتعمل مرة واحدة بس مش مع كل سؤال
@st.cache_resource
def load_engine():
    return RAGEngine()

rag = load_engine()

query = st.text_input("Ask a medical question:")

if query:
    with st.spinner("Thinking..."):
        answer = rag.ask(query)

    st.write("### Answer:")
    st.write(answer)