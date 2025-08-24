import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma

# Local fetchers
from notion_fetcher import fetch_notion_docs
from fetch_gdocs import fetch_gdocs_docs

# ------------------- SETUP -------------------
load_dotenv()
st.set_page_config(page_title="Internal Docs Q&A", page_icon="📄", layout="wide")

# ------------------- CUSTOM CSS -------------------
st.markdown(
    """
    <style>
    body { font-family: 'Segoe UI', sans-serif; }
    .chat-bubble-user {
        background-color: #daf8cb;
        padding: 10px; border-radius: 12px;
        margin: 5px 0; max-width: 70%;
    }
    .chat-bubble-bot {
        background-color: #f1f0f0;
        padding: 10px; border-radius: 12px;
        margin: 5px 0; max-width: 70%;
    }
    .app-title {
        text-align: center;
        font-size: 2em;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.title("⚙️ Settings")

    theme = st.radio("Theme", ["Light", "Dark"], index=0)

    sources = ["Notion", "Google Docs"]
    selected_sources = st.multiselect("Select sources", sources, default=sources)

    st.markdown("---")
    st.subheader("Quick Prompts")
    if st.button("📌 Company Policy"):
        st.session_state.user_input = "What is our company policy?"
    if st.button("📌 Leave Process"):
        st.session_state.user_input = "How do I apply for leave?"

# ------------------- THEME TOGGLE -------------------
if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #1e1e1e; color: white; }
        .chat-bubble-user { background-color: #3a3d41; color: white; }
        .chat-bubble-bot { background-color: #2d2d2d; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------- HEADER -------------------
st.markdown('<div class="app-title">📄 Internal Docs Q&A</div>', unsafe_allow_html=True)

# ------------------- VECTORSTORE -------------------
@st.cache_resource
def load_vectorstore(source_names):
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = "./chroma_index"

    if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
        return Chroma(persist_directory=persist_dir, embedding_function=emb)

    docs = []
    if "Notion" in source_names:
        docs.extend([d["text"] for d in fetch_notion_docs()])
    if "Google Docs" in source_names:
        docs.extend([d["text"] for d in fetch_gdocs_docs()])

    if not docs:
        st.error("⚠️ No documents loaded. Please check your sources.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for d in docs:
        chunks.extend(splitter.split_text(d))

    vs = Chroma.from_texts(chunks, emb, persist_directory=persist_dir)
    vs.persist()
    return vs


vs = load_vectorstore(selected_sources)
llm = ChatOllama(model="gemma:2b-instruct-q4_0")

# ------------------- MEMORY -------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def render_typing(message: str):
    """Typing animation for assistant replies."""
    placeholder = st.empty()
    typed_text = ""
    for char in message:
        typed_text += char
        placeholder.markdown(f'<div class="chat-bubble-bot">{typed_text}</div>', unsafe_allow_html=True)
        time.sleep(0.015)


def ask(query):
    hits = vs.similarity_search(query, k=3) if vs else []
    context = "\n".join([h.page_content for h in hits])

    memory_context = "\n".join(
        [
            f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}"
            for m in st.session_state["messages"]
        ]
    )

    prompt = f"""You are an assistant that answers using company docs.

Conversation so far:
{memory_context}

Context:
{context}

Question: {query}
Answer:"""

    resp = llm.invoke(prompt)
    return resp.content

# ------------------- CHAT UI -------------------
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)

if "user_input" in st.session_state and st.session_state.user_input:
    prompt = st.session_state.user_input
    st.session_state.user_input = ""
else:
    prompt = st.chat_input("Ask me anything about company docs...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-bubble-user">{prompt}</div>', unsafe_allow_html=True)

    with st.spinner("🤔 Thinking..."):
        answer = ask(prompt)
        render_typing(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})

# ------------------- RESPONSIVENESS -------------------
st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .chat-bubble-user, .chat-bubble-bot {
            max-width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)
