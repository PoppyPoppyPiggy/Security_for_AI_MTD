# =============================================================================
# FILE: tests/test_indexer.py
# DESC: Unit tests for DocumentIndexer — chunking, indexing, save/load
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/rag_pipeline/indexer.py, config/rag_config.yaml
# =============================================================================
from __future__ import annotations

import os
import tempfile

import pytest

from src.rag_pipeline.indexer import Chunk, Document, DocumentIndexer


@pytest.fixture
def indexer() -> DocumentIndexer:
    return DocumentIndexer(config_path="config/rag_config.yaml")


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(text="A " * 200, source_path="test.txt", doc_id="d0"),
        Document(text="Firewall protects networks.", source_path="test.txt", doc_id="d1"),
    ]


@pytest.fixture
def sample_chunks(indexer: DocumentIndexer, sample_docs: list[Document]) -> list[Chunk]:
    return indexer.chunk_documents(sample_docs)


class TestChunking:
    def test_short_doc_single_chunk(self, indexer: DocumentIndexer) -> None:
        """Short document should produce exactly one chunk."""
        docs = [Document(text="Short text.", source_path="t.txt", doc_id="d0")]
        chunks = indexer.chunk_documents(docs)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_long_doc_multiple_chunks(self, indexer: DocumentIndexer) -> None:
        """Document longer than chunk_size should produce multiple chunks."""
        long_text = "word " * 200  # ~1000 chars > 256 chunk_size
        docs = [Document(text=long_text.strip(), source_path="t.txt", doc_id="d0")]
        chunks = indexer.chunk_documents(docs)
        assert len(chunks) > 1

    def test_chunk_overlap(self, indexer: DocumentIndexer) -> None:
        """Adjacent chunks should share overlapping text."""
        long_text = "A" * 512  # exactly 2+ chunks
        docs = [Document(text=long_text, source_path="t.txt", doc_id="d0")]
        chunks = indexer.chunk_documents(docs)
        assert len(chunks) >= 2
        # With overlap=32, second chunk starts at position 224 (256-32)
        # so it should share characters with the first chunk
        first_end = chunks[0].text
        second_start = chunks[1].text[:indexer.chunk_overlap]
        assert first_end.endswith(second_start)

    def test_chunk_metadata(self, indexer: DocumentIndexer) -> None:
        """Chunks preserve source metadata and have unique IDs."""
        docs = [Document(text="Test content here.", source_path="src.txt", doc_id="d5")]
        chunks = indexer.chunk_documents(docs)
        assert chunks[0].source_path == "src.txt"
        assert chunks[0].doc_id == "d5"
        assert chunks[0].is_poisoned is False


class TestDenseIndex:
    def test_index_shape(self, indexer: DocumentIndexer, sample_chunks: list[Chunk]) -> None:
        """FAISS index dimension should match embedding model output."""
        index = indexer.build_dense_index(sample_chunks)
        assert index.d == indexer.encoder.get_sentence_embedding_dimension()
        assert index.ntotal == len(sample_chunks)


class TestSparseIndex:
    def test_sparse_index_builds(self, indexer: DocumentIndexer, sample_chunks: list[Chunk]) -> None:
        """BM25 index should build without error."""
        bm25 = indexer.build_sparse_index(sample_chunks)
        scores = bm25.get_scores("firewall".split())
        assert len(scores) == len(sample_chunks)


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, indexer: DocumentIndexer, sample_chunks: list[Chunk]) -> None:
        """Save then load should return identical chunk texts."""
        dense_index = indexer.build_dense_index(sample_chunks)
        sparse_index = indexer.build_sparse_index(sample_chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            indexer.save_index(tmpdir, dense_index, sparse_index, sample_chunks)
            loaded_dense, loaded_sparse, loaded_chunks = indexer.load_index(tmpdir)

        assert loaded_dense.ntotal == dense_index.ntotal
        assert len(loaded_chunks) == len(sample_chunks)
        for orig, loaded in zip(sample_chunks, loaded_chunks):
            assert orig.text == loaded.text
            assert orig.chunk_id == loaded.chunk_id


class TestLoadDocuments:
    def test_load_txt(self, indexer: DocumentIndexer) -> None:
        """Loading from kb_clean should return documents."""
        docs = indexer.load_documents("data/knowledge_bases/kb_clean/")
        assert len(docs) == 20  # 20 paragraphs in seed_docs.txt

    def test_load_json(self, indexer: DocumentIndexer) -> None:
        """Loading JSON files should parse entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import json
            data = [{"text": "doc one"}, {"text": "doc two"}]
            with open(os.path.join(tmpdir, "test.json"), "w") as f:
                json.dump(data, f)
            docs = indexer.load_documents(tmpdir)
            assert len(docs) == 2
            assert docs[0].text == "doc one"
