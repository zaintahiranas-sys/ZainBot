# ZainBot - Personal RAG Chatbot

**Course:** Natural Language Processing (CC438), UMT Lahore, Summer 2026
**Student:** Muhammad Zain

ZainBot is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about Muhammad Zain
(education, experience, projects, skills) using his own CV and resume as the knowledge base.

## Live demo
Deployment link: **PASTE YOUR STREAMLIT CLOUD LINK HERE**

## How it works
1. **Document loading** - `data/` contains the CV (PDF) and resume (DOCX). `rag.py` reads PDF, DOCX, TXT and MD files.
2. **Preprocessing** - whitespace cleaning and automatic redaction of phone numbers (privacy).
3. **Chunking** - lines are grouped into ~650 character chunks with overlap; each chunk keeps its source file name.
4. **Embeddings** - Google Gemini embeddings (`gemini-embedding-001`).
5. **Vector database** - FAISS (`IndexFlatIP` with normalized vectors = cosine similarity).
6. **Retrieval** - top-5 chunks for each question (previous question is added to help follow-ups).
7. **Prompt engineering** - a system prompt gives the bot its identity (ZainBot), forces answers only from context, handles conflicting CV versions, and blocks phone/address sharing.
8. **LLM generation** - Gemini (`gemini-2.5-flash`) answers using retrieved context.
9. **History** - the last 8 messages are sent to the LLM; the chat is kept in Streamlit session state.
10. **Interface** - Streamlit chat UI with sample questions and a "Retrieved context" viewer.

## Project structure
```
app.py                 Streamlit interface
rag.py                 RAG pipeline
data/                  CV (PDF) + resume (DOCX)
test_pipeline.py       Offline test (no API key needed)
requirements.txt
.streamlit/secrets.toml.example
```

## Run locally
```bash
pip install -r requirements.txt
mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# put your Gemini API key in .streamlit/secrets.toml (free key: https://aistudio.google.com/apikey)
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository (do NOT push `secrets.toml`).
2. Go to https://share.streamlit.io and click **Create app**.
3. Select the repository, branch `main`, main file `app.py`.
4. Open **Advanced settings > Secrets** and paste: `GEMINI_API_KEY = "your-key"`.
5. Click **Deploy**. Copy the app link into the "Live demo" section above.

## Update the knowledge base
Put any new PDF, DOCX, TXT or MD file in `data/` and restart the app. It is re-indexed automatically.
