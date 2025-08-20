# 🦙 Streamlit RAG LLaMA Chatbot

A **Streamlit-based Retrieval-Augmented Generation (RAG) chatbot** powered by **LLaMA** and GPT-4, designed to answer user queries from uploaded documents using both vector and keyword-based search.  

This project supports **PDF and image document indexing**, combining **semantic search** with **BM25/Tf-IDF** to provide accurate responses.  

---

## ✨ Features

- 💬 **Interactive Chat Interface** – Seamless conversation with the RAG chatbot.  
- 📄 **Document-Based QA** – Retrieves answers from uploaded PDFs and images.  
- 🔎 **Hybrid Search** – Combines semantic vector search with keyword-based search.  
- 📂 **Document Upload & Indexing** – Upload folders of PDFs/PNGs, automatically indexed.  
- 🧠 **Persistent Chat History** – Stores previous conversations locally.  
- 🖼 **OCR Support** – Extracts text from images using **Tesseract**.  

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/aneesh1110/streamlit-rag-llama.git
cd streamlit-rag-llama/Interface
```
### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure API Keys
```env
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=hf_real_token_here
PINECONE_TOKEN=pinecone_token_here
```

## ✨Usage

### 1. Run the Streamlit App
```bash
streamlit run App.py
```
### 2. Upload Documents for Indexing
- Place PDFs and PNG images in the docs/ folder.
- Text is extracted via PyMuPDF (PDFs) and Tesseract OCR (images).
- Indexed documents are stored for retrieval.

### 3. Start Chatting
- Enter queries in the input box.
- The system retrieves relevant information and generates GPT-4-powered answers.
- Chat history is persisted for later reference.

## 📂 Folder Structure
```bash
.
├── Interface/
│   ├── App.py               # Streamlit frontend
│   ├── backend.py           # Document processing and GPT-4 interaction
│   ├── data/                # Chat history storage
│   ├── docs/                # Upload PDFs/PNGs here
│   ├── __pycache__/         # Compiled Python files
├── .env                     # API keys (DO NOT SHARE)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

## 🛠️ Technologies Used
- Python
- Streamlit (UI)
- GPT-4 API (NLP)
- Hugging Face / LLaMA (RAG)
- Pinecone / FAISS (Vector Search)
- SentenceTransformers (Embeddings)
- BM25/Tf-IDF (Keyword Search)
- PyMuPDF (PDF Processing)
- Tesseract OCR (Image Text Extraction)
- Joblib (Chat History Storage)

## 🐛 Troubleshooting

### 🔑 API Key Issues

If you encounter errors related to API keys, ensure:

- Ensure .env contains the correct credentials.

- Sourced the .env variables (source .env).
```bash
source .env
```

### 📦 Pinecone Index Not Found

Ensure you have created the index before running queries:
```python
pc.create_index(name='icici-reports', dimension=384, metric='cosine')
```

### ⚡ Streamlit Not Running Properly

Try clearing cached files:
```bash
streamlit cache clear
```

## 🔮 Future Enhancements
- User Authentication
- Multi-document summarization
- Improved UI/UX with chat customization

## 📜 License
MIT License

## 👨‍💻 Author
Aneesh Vinod
