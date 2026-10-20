# TraceRAG — Resource-Aware RAG Reliability Lab

TraceRAG is a local-first engineering project for experimenting with retrieval-augmented generation (RAG). The first milestone focuses on PDF ingestion, page-aware chunking, hybrid retrieval, and reproducible retrieval metrics.

> Research honesty: TraceRAG is a proposed portfolio project, not a claim of a novel research breakthrough. Its contribution must be established through documented experiments and comparison with existing tools.

## Current MVP
- Upload research PDFs and preserve page numbers.
- Split extracted text into overlapping chunks.
- Build local dense embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Combine dense and BM25 rankings using Reciprocal Rank Fusion (RRF).
- Search for relevant passages and return source filename + page number.
- Evaluate a labelled question set using Recall@K, MRR, and nDCG@K.
- Run locally; no paid API or hosted database is required.

## Requirements
- Python 3.11 recommended (Python 3.13 may work, but ML dependency compatibility can vary).
- 8–12 GB RAM is enough for this small CPU-based prototype with a small number of PDFs.
- Internet is needed the first time the embedding model is downloaded. After caching, embeddings can run locally.

## Windows setup (PowerShell)
```powershell
# 1. Open this folder in VS Code
cd path\to\TraceRAG

# 2. Create and activate a virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation, use:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Start the API
uvicorn app.main:app --reload

# 5. Open API docs
# http://127.0.0.1:8000/docs
```

## First use
1. Open `http://127.0.0.1:8000/docs`.
2. Use `POST /documents/upload` to upload a small research PDF.
3. Use `POST /search` with a question such as `What dataset did the authors use?`.
4. The response returns ranked passages and source page numbers.

## Evaluation
Create a JSON file based on `data/eval_sample.json`, replacing the example document and page with the page that truly supports each question. Run:
```powershell
python -m app.evaluate data/eval_sample.json
```
Metrics are meaningful only when the expected document/page labels are manually checked. Do not report fabricated benchmark scores.

## Project structure
```text
TraceRAG/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── rag.py
│   ├── metrics.py
│   └── evaluate.py
├── data/
│   └── eval_sample.json
├── tests/
│   └── test_metrics.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Planned milestones
1. **MVP:** PDF ingestion + page-aware chunks + hybrid retrieval.
2. **Benchmark:** Build a small manually labelled evaluation set; compare BM25, dense retrieval, and hybrid retrieval.
3. **Fault injection:** Create controlled cases (irrelevant distractors, missing evidence, conflicting/outdated passages).
4. **Diagnosis:** Record failure cases and test one intervention at a time.
5. **Resource-aware experiments:** Compare quality, latency, and memory cost.
6. **UI + packaging:** React dashboard, Docker, GitHub Actions, and a reproducible benchmark report.

## Limitations of this first version
- Documents are held in memory; restarting the API clears the index.
- It retrieves evidence but does not yet generate answers with an LLM.
- The evaluation CLI expects you to create and verify your own gold labels.
- PDF text extraction can fail on scanned/image-only PDFs; OCR is not included yet.
