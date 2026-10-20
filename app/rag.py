
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
import time

import fitz
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    chunk_id: str
    document_name: str
    page_number: int
    text: str


class TraceRAGEngine:
    """In-memory hybrid retrieval engine for TraceRAG."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 900,
        overlap: int = 150,
    ):
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Use chunk_size > overlap >= 0.")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model_name = model_name
        self._model = None
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25 = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name, device="cpu"
            )
        return self._model

    def _split_page(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        # Prefer sentence boundaries, but preserve long sentences.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Split unusually long sentences into smaller pieces.
            if len(sentence) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""

                start = 0
                while start < len(sentence):
                    end = min(start + self.chunk_size, len(sentence))

                    if end < len(sentence):
                        boundary = sentence.rfind(
                            " ", start + int(self.chunk_size * 0.7), end
                        )
                        if boundary > start:
                            end = boundary

                    piece = sentence[start:end].strip()
                    if piece:
                        chunks.append(piece)

                    if end >= len(sentence):
                        break
                    start = end

                continue

            candidate = f"{current} {sentence}".strip() if current else sentence

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)

                # Keep a small overlap using complete trailing sentences.
                tail = current
                overlap_sentences = []

                for part in reversed(re.split(r"(?<=[.!?])\s+", tail)):
                    if len(" ".join([part] + overlap_sentences)) > self.overlap:
                        break
                    overlap_sentences.insert(0, part)

                current = " ".join(overlap_sentences).strip()

                if current:
                    candidate = f"{current} {sentence}".strip()
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        current = sentence
                else:
                    current = sentence

        if current:
            chunks.append(current)

        return chunks

    def add_pdf_bytes(self, filename: str, content: bytes) -> dict:
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")
        if not content:
            raise ValueError("The uploaded file is empty.")

        document = fitz.open(stream=content, filetype="pdf")
        new_chunks: list[Chunk] = []

        try:
            for page_index, page in enumerate(document):
                page_text = page.get_text("text")

                for chunk_index, piece in enumerate(
                    self._split_page(page_text)
                ):
                    new_chunks.append(
                        Chunk(
                            chunk_id=(
                                f"{filename}:p{page_index + 1}:"
                                f"c{chunk_index + 1}"
                            ),
                            document_name=filename,
                            page_number=page_index + 1,
                            text=piece,
                        )
                    )
        finally:
            document.close()

        if not new_chunks:
            raise ValueError(
                "No selectable text was found. "
                "This may be a scanned PDF; OCR is not included yet."
            )

        # Replace the previous copy of the same filename.
        self.chunks = [
            chunk for chunk in self.chunks
            if chunk.document_name != filename
        ]
        self.chunks.extend(new_chunks)
        self._rebuild_index()

        return {
            "document_name": filename,
            "pages_indexed": len(
                {chunk.page_number for chunk in new_chunks}
            ),
            "chunks_added": len(new_chunks),
            "total_chunks": len(self.chunks),
        }

    def _rebuild_index(self) -> None:
        if not self.chunks:
            self.embeddings = None
            self.bm25 = None
            return

        texts = [chunk.text for chunk in self.chunks]

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        tokenized = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b",
            text.lower(),
        )

    @staticmethod
    def _rrf(
        rankings: Iterable[list[int]], k: int = 60
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}

        for ranking in rankings:
            for rank, index in enumerate(ranking, start=1):
                scores[index] = (
                    scores.get(index, 0.0) + 1.0 / (k + rank)
                )

        return sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

    def search(
        self, query: str, top_k: int = 5, mode: str = "hybrid"
    ) -> dict:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20.")
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("mode must be dense, bm25, or hybrid.")

        if not self.chunks:
            return {
                "query": query,
                "mode": mode,
                "latency_ms": 0.0,
                "results": [],
            }

        start_time = time.perf_counter()
        count = len(self.chunks)
        limit = min(max(top_k * 3, top_k), count)

        dense_ranking: list[int] = []
        bm25_ranking: list[int] = []
        dense_scores = None
        bm25_scores = None

        if mode in {"dense", "hybrid"}:
            query_vector = self.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]

            dense_scores = self.embeddings @ query_vector
            dense_ranking = np.argsort(-dense_scores)[:limit].tolist()

        if mode in {"bm25", "hybrid"}:
            bm25_scores = self.bm25.get_scores(self._tokenize(query))
            bm25_ranking = np.argsort(-bm25_scores)[:limit].tolist()

        if mode == "dense":
            ranked = [
                (index, float(dense_scores[index]))
                for index in dense_ranking[:top_k]
            ]
        elif mode == "bm25":
            ranked = [
                (index, float(bm25_scores[index]))
                for index in bm25_ranking[:top_k]
            ]
        else:
            ranked = self._rrf(
                [dense_ranking, bm25_ranking]
            )[:top_k]

        results = []

        for rank, (index, score) in enumerate(ranked, start=1):
            chunk = self.chunks[index]
            results.append({
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "score": round(float(score), 6),
                "text": chunk.text,
            })

        return {
            "query": query,
            "mode": mode,
            "latency_ms": round(
                (time.perf_counter() - start_time) * 1000, 2
            ),
            "results": results,
        }
