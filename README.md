# Legal RAG - Indian Supreme Court Judgment Search & QA

A Retrieval-Augmented Generation (RAG) system for searching and querying Indian Supreme Court judgments using hybrid retrieval (semantic + keyword) with LLM-powered answers.

## Architecture

```
PDF Judgments  →  extract.py  →  clean_data.py  →  chunk_data.py  →  create_embeddings.py
                                                                           │
                                                                     FAISS Index + Chunks
                                                                           │
User Query  →  hybrid_retrieve.py  →  reranker.py  →  ask_llm.py (Groq LLM)  →  Answer
                   │
            ┌──────┴──────┐
      retrieve.py    bm25_retriever.py
      (FAISS/BGE)      (BM25/Keyword)
```

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `extract.py` | Extracts text from PDF judgments, parses case names and dates via regex |
| 2 | `clean_data.py` | Removes boilerplate, links, page numbers, and other noise |
| 3 | `chunk_data.py` | Splits cleaned text into 1000-char chunks with 200-char overlap |
| 4 | `create_embeddings.py` | Generates embeddings using `BAAI/bge-small-en-v1.5` and builds a FAISS index |
| 5 | `hybrid_retrieve.py` | Combines FAISS semantic search + BM25 keyword search |
| 6 | `reranker.py` | Re-ranks results with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| 7 | `ask_llm.py` | Interactive CLI that retrieves context and generates answers via Groq LLM |

## Key Features

- **Dual retrieval**: FAISS (dense embeddings) + BM25 (sparse/keyword) for robust search
- **Fuzzy case matching**: Matches user query to known case names using `rapidfuzz` before falling back to vector search
- **Cross-encoder re-ranking**: Improves result relevance beyond initial retrieval scores
- **Strict grounding**: LLM is instructed to answer only from retrieved context — no hallucination of legal facts
- **Indian Kanoon dataset**: Works with Supreme Court judgment PDFs from Indian Kanoon

## Project Structure

```
├── extract.py              # PDF text extraction & metadata parsing
├── clean_data.py           # Text cleaning & normalization
├── chunk_data.py           # Text chunking with LangChain
├── create_embeddings.py    # BGE embedding + FAISS index creation
├── retrieve.py             # FAISS vector search + fuzzy case matching
├── bm25_retriever.py       # BM25 keyword search
├── hybrid_retrieve.py      # Combined FAISS + BM25 retrieval
├── reranker.py             # Cross-encoder re-ranking
├── ask_llm.py              # Main entry point (CLI)
├── case_context.py         # Helper to filter chunks by case name
├── check.py                # Debugging script for cleaned data
├── dataset/                # PDF judgments organized by year
├── output/                 # Generated artifacts (CSV, JSON, FAISS index)
├── tests/                  # Test suite
└── requirements.txt
```

## Setup

### 1. Clone and create virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/macOS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Create a `.env` file in the project root:

```
API_KEY=your_groq_api_key_here
```

### 4. Prepare dataset

Place Supreme Court judgment PDFs in `dataset/<year>/` directories (e.g., `dataset/2025/`).

### 5. Run the pipeline

```bash
# Step 1: Extract text from PDFs
python extract.py

# Step 2: Clean extracted text
python clean_data.py

# Step 3: Chunk documents
python chunk_data.py

# Step 4: Create embeddings & FAISS index
python create_embeddings.py

# Step 5: Start the QA system
python ask_llm.py
```

## Usage

After running `python ask_llm.py`, type legal questions at the prompt:

```
Enter Legal Query (or type exit): What did the court decide about bail conditions?
```

The system will:
1. Check for fuzzy case name matches
2. Run hybrid retrieval (FAISS + BM25)
3. Re-rank results with the cross-encoder
4. Send top chunks as context to the LLM
5. Return a grounded answer

Type `exit` to quit.

## Tech Stack

- **Embeddings**: `BAAI/bge-small-en-v1.5` (SentenceTransformers)
- **Vector Search**: FAISS (L2 distance)
- **Keyword Search**: BM25 (rank_bm25)
- **Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: Groq API (`llama-3.1-8b-instant`)
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter
- **Fuzzy Matching**: rapidfuzz
- **PDF Processing**: PyMuPDF (fitz)
- **Data**: pandas, numpy

## License

This project is for educational and research purposes. The dataset is sourced from publicly available Indian Kanoon judgments.