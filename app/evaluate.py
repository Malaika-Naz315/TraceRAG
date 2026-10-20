"""Evaluate retrieval against manually verified relevant chunk IDs.

Usage:
    python -m app.evaluate data/eval_sample.json

The evaluation file must reference chunk IDs returned by /search. Labels should be
verified by a human; sample labels are intentionally placeholders.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from statistics import mean

from app.metrics import ndcg_at_k, reciprocal_rank, recall_at_k
from app.rag import TraceRAGEngine


def main(path: str) -> None:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    engine = TraceRAGEngine()
    pdf_paths = sorted(Path("data/pdfs").glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit("Put evaluation PDFs in data/pdfs/ first, then edit the gold labels.")
    for pdf_path in pdf_paths:
        engine.add_pdf_bytes(pdf_path.name, pdf_path.read_bytes())

    all_metrics = []
    for item in records:
        relevant = set(item["relevant_chunk_ids"])
        result = engine.search(item["question"], top_k=10, mode="hybrid")
        ids = [r["chunk_id"] for r in result["results"]]
        row = {
            "question": item["question"],
            "recall@5": recall_at_k(ids, relevant, 5),
            "mrr": reciprocal_rank(ids, relevant),
            "ndcg@5": ndcg_at_k(ids, relevant, 5),
            "latency_ms": result["latency_ms"],
        }
        all_metrics.append(row)
        print(json.dumps(row, ensure_ascii=False))

    if all_metrics:
        print("\nMEAN METRICS")
        for key in ("recall@5", "mrr", "ndcg@5", "latency_ms"):
            print(f"{key}: {mean(row[key] for row in all_metrics):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m app.evaluate path/to/evaluation.json")
    main(sys.argv[1])
