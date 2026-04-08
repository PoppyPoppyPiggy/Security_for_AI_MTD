# =============================================================================
# FILE: src/rag_pipeline/indexer.py
# DESC: Document chunking + vector indexing for RAG pipeline
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: config/rag_config.yaml, sentence-transformers, faiss-cpu, rank-bm25
# =============================================================================
from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A raw document loaded from the knowledge base."""

    text: str
    source_path: str
    doc_id: str


@dataclass
class Chunk:
    """A chunked segment of a document."""

    text: str
    source_path: str
    chunk_id: str
    doc_id: str
    is_poisoned: bool = False


class DocumentIndexer:
    """Loads documents, chunks them, and builds dense/sparse indexes.

    Args:
        config_path: Path to rag_config.yaml.

    ATLAS:
        Target component for AML.T0054 (False RAG Entry Injection).
        Poisoned documents enter the pipeline through this indexer.
    """

    def __init__(self, config_path: str = "config/rag_config.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        rag_cfg = cfg["rag"]
        self.chunk_size: int = rag_cfg["chunk_size"]
        self.chunk_overlap: int = rag_cfg["chunk_overlap"]
        self.embedding_model_name: str = rag_cfg["embedding_model"]
        self._encoder: SentenceTransformer | None = None

    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy-load embedding model to avoid cost during config-only usage."""
        if self._encoder is None:
            logger.info("Loading embedding model: %s", self.embedding_model_name)
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def load_documents(self, kb_path: str) -> list[Document]:
        """Load all .txt and .json files from kb_path recursively.

        Args:
            kb_path: Root directory of the knowledge base.

        Returns:
            List of Document objects with text content and metadata.
        """
        docs: list[Document] = []
        kb_root = Path(kb_path)
        for file_path in sorted(kb_root.rglob("*")):
            if file_path.suffix == ".txt":
                text = file_path.read_text(encoding="utf-8").strip()
                if text:
                    # Split .txt by double-newline to get individual paragraphs as docs
                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                    for i, para in enumerate(paragraphs):
                        doc_id = f"{file_path.stem}_{i}"
                        docs.append(Document(text=para, source_path=str(file_path), doc_id=doc_id))
            elif file_path.suffix == ".json":
                data = json.loads(file_path.read_text(encoding="utf-8"))
                entries = data if isinstance(data, list) else [data]
                for i, entry in enumerate(entries):
                    text = entry.get("text", entry.get("content", str(entry)))
                    doc_id = f"{file_path.stem}_{i}"
                    docs.append(Document(text=text, source_path=str(file_path), doc_id=doc_id))

        logger.info("Loaded %d documents from %s", len(docs), kb_path)
        return docs

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Split documents into fixed-size character chunks with overlap.

        Args:
            docs: List of Document objects to chunk.

        Returns:
            List of Chunk objects with text segments and metadata.
        """
        chunks: list[Chunk] = []
        for doc in docs:
            text = doc.text
            if len(text) <= self.chunk_size:
                chunks.append(Chunk(
                    text=text,
                    source_path=doc.source_path,
                    chunk_id=f"{doc.doc_id}_c0",
                    doc_id=doc.doc_id,
                ))
                continue

            step = self.chunk_size - self.chunk_overlap
            start = 0
            idx = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(Chunk(
                        text=chunk_text,
                        source_path=doc.source_path,
                        chunk_id=f"{doc.doc_id}_c{idx}",
                        doc_id=doc.doc_id,
                    ))
                    idx += 1
                start += step

        logger.info("Chunked %d documents into %d chunks", len(docs), len(chunks))
        return chunks

    def build_dense_index(self, chunks: list[Chunk]) -> faiss.IndexFlatIP:
        """Build a FAISS inner-product index from chunk embeddings.

        Vectors are L2-normalized so inner product equals cosine similarity.

        Args:
            chunks: List of Chunk objects to index.

        Returns:
            FAISS IndexFlatIP with normalized embeddings.
        """
        texts = [c.text for c in chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)  # normalize so IP = cosine

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        logger.info("Built dense index: %d vectors, dim=%d", index.ntotal, dim)
        return index

    def build_sparse_index(self, chunks: list[Chunk]) -> BM25Okapi:
        """Build a BM25 sparse index from chunk texts.

        Args:
            chunks: List of Chunk objects to index.

        Returns:
            BM25Okapi index for sparse retrieval.
        """
        tokenized = [c.text.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        logger.info("Built sparse index: %d documents", len(tokenized))
        return bm25

    def save_index(
        self,
        index_path: str,
        dense_index: faiss.IndexFlatIP,
        sparse_index: BM25Okapi,
        chunks: list[Chunk],
    ) -> None:
        """Persist FAISS index and chunk metadata to disk.

        Args:
            index_path: Directory path for saving index files.
            dense_index: FAISS index to save.
            sparse_index: BM25 index to save.
            chunks: Chunk metadata list to save.
        """
        os.makedirs(index_path, exist_ok=True)
        faiss.write_index(dense_index, os.path.join(index_path, "dense.faiss"))
        with open(os.path.join(index_path, "sparse.pkl"), "wb") as f:
            pickle.dump(sparse_index, f)
        with open(os.path.join(index_path, "chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)
        logger.info("Saved index to %s", index_path)

    def load_index(self, index_path: str) -> tuple[faiss.IndexFlatIP, BM25Okapi, list[Chunk]]:
        """Reload persisted index and chunk metadata from disk.

        Args:
            index_path: Directory path containing saved index files.

        Returns:
            Tuple of (dense_index, sparse_index, chunks).
        """
        dense_index = faiss.read_index(os.path.join(index_path, "dense.faiss"))
        with open(os.path.join(index_path, "sparse.pkl"), "rb") as f:
            sparse_index = pickle.load(f)
        with open(os.path.join(index_path, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        logger.info("Loaded index from %s: %d chunks", index_path, len(chunks))
        return dense_index, sparse_index, chunks
