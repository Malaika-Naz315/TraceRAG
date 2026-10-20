from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag import TraceRAGEngine

app = FastAPI(
    title="TraceRAG API",
    description="Local-first RAG retrieval and reliability experiments.",
    version="0.1.0",
)
engine = TraceRAGEngine()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(dense|bm25|hybrid)$")


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(engine.chunks), "model_loaded": engine._model is not None}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    content = await file.read()
    try:
        return engine.add_pdf_bytes(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF indexing failed: {type(exc).__name__}") from exc


@app.post("/search")
def search(request: SearchRequest):
    try:
        return engine.search(request.query, request.top_k, request.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {type(exc).__name__}") from exc
