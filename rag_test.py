# rag_test.py
import os
from notion_fetcher import load_notion_docs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

print("📥 Fetching docs from Notion...")
docs = load_notion_docs()

raw_texts = []
metadatas = []

for d in docs:
    text = d.get("text", "").strip()
    if not text:
        print(f"⚠️ Doc #{d.get('id', 'unknown')} was empty, skipping.")
        continue
    raw_texts.append(text)
    metadatas.append({"source": d.get("id", "unknown")})

if not raw_texts:
    raise ValueError("❌ No text found in Notion docs. Check fetch_page_content!")

# Split into chunks
print("✂️ Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

all_chunks = []
all_metadatas = []

for text, meta in zip(raw_texts, metadatas):
    chunks = text_splitter.split_text(text)
    all_chunks.extend(chunks)
    all_metadatas.extend([meta] * len(chunks))

print(f"✅ Total chunks: {len(all_chunks)}")

# Embeddings
print("🧠 Creating embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_texts(all_chunks, embedding=embedding_model, metadatas=all_metadatas)

# LLM
print("🤖 Loading Gemma 2B...")
llm = OllamaLLM(model="gemma:2b")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    print(f"\n❓ Question: {query}")
    result = qa_chain.invoke({"query": query})
    print(f"💡 Answer: {result['result']}")
    print("📚 Sources:", [d.metadata.get("source", "unknown") for d in result["source_documents"]])
