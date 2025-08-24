# 📚 Internal Docs Q&A Agent  

> An AI-powered assistant that answers company questions from **Notion docs** using **Retrieval-Augmented Generation (RAG)**.  
> Built with **Streamlit + ChromaDB + Ollama (Gemma:2B)** — completely free, private, and local.  

---

## ✨ Features
- 🔗 Fetch documentation directly from the **Notion API**  
- ✂️ Smart **chunking & embeddings** via HuggingFace  
- 💾 Store & search efficiently with **ChromaDB**  
- 🤖 Generate precise answers using **Ollama (Gemma:2B)**  
- 💬 Sleek and interactive **chat UI** powered by Streamlit  
- 🔒 100% **local & private** — no OpenAI API or external calls needed  

---

## 🛠️ Tech Stack

| Layer       | Tool / Library                     |
|-------------|------------------------------------|
| **Data Fetch**  | Notion API                       |
| **Embeddings**  | HuggingFace (`all-MiniLM-L6-v2`) |
| **Vector DB**   | ChromaDB                         |
| **AI Engine**   | Ollama (`gemma:2b`)              |
| **Backend**     | LangChain                        |
| **Frontend**    | Streamlit                        |

---

## 📦 Installation

### 1️⃣ Clone the repo

git clone https://github.com/<your-username>/internal-docs-qa-agent.git
cd internal-docs-qa-agent

text

### 2️⃣ Create a virtual environment

python -m venv .venv
Activate it:

source .venv/bin/activate # Mac/Linux
.venv\Scripts\activate # Windows

text

### 3️⃣ Install dependencies

pip install -r requirements.txt

text

### 4️⃣ Install Ollama & pull the model
- Download from: https://ollama.ai/download  
- Then pull the model:

ollama pull gemma:2b

text

### 5️⃣ Setup Environment Variables
Create a `.env` file in the project root:

NOTION_API_KEY=your_notion_api_key
NOTION_PAGE_IDS=comma,separated,page_ids

text

---

## ▶️ Run the App

streamlit run app.py

text

Now open your browser at:  
👉 [**http://localhost:8501**](http://localhost:8501) 🎉  

---

## 🤝 Contributing
1. Fork the repo  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit changes & push (`git push origin feature-name`)  
4. Submit a Pull Request 🚀  

---

## 📜 License
Licensed under the **MIT License**.  
Feel free to use, modify, and share.  

---
