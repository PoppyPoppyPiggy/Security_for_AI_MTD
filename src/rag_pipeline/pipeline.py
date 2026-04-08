# =============================================================================
# FILE: src/rag_pipeline/pipeline.py
# DESC: End-to-end RAG pipeline orchestrating indexer, retriever, generator
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: config/rag_config.yaml, config/mtd_config.yaml,
#       src/rag_pipeline/indexer.py, retriever.py, generator.py
# =============================================================================
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from src.rag_pipeline.generator import Generator, GeneratorOutput
from src.rag_pipeline.indexer import Chunk, DocumentIndexer
from src.rag_pipeline.retriever import Retriever, RetrievalResult
from src.utils import safe_load_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    """Complete output from a single RAG pipeline run."""

    answer: str
    retrieved_chunks: list[RetrievalResult]
    retrieval_latency_ms: float
    generation_latency_ms: float
    active_kb: str
    active_retriever: str
    active_embedder: str


class RAGPipeline:
    """End-to-end RAG pipeline with optional MTD integration.

    Args:
        mode: Pipeline mode — 'baseline' (no MTD) or 'mtd'.
        rag_config_path: Path to rag_config.yaml.
        mtd_config_path: Path to mtd_config.yaml.
        kb_path: Override KB path (default: from config or kb_clean/).

    ATLAS:
        This pipeline is the target system defended by the ATLAS-MTD-RAG
        framework. In baseline mode, it has no MTD protection and is
        vulnerable to AML.T0054, AML.T0057, AML.T0051.
    """

    def __init__(
        self,
        mode: str = "baseline",
        rag_config_path: str = "config/rag_config.yaml",
        mtd_config_path: str = "config/mtd_config.yaml",
        kb_path: str | None = None,
    ) -> None:
        self.mode = mode
        self.rag_config_path = rag_config_path
        self.mtd_config_path = mtd_config_path

        self.rag_cfg = safe_load_config(rag_config_path)["rag"]
        self.mtd_cfg = safe_load_config(mtd_config_path)["mtd"]

        self.top_k: int = self.rag_cfg["top_k"]
        self.strategy: str = self.rag_cfg["retrieval_strategy"]
        self.embedding_model_name: str = self.rag_cfg["embedding_model"]

        # Init MTD controller for mtd mode
        self.mtd = None
        if mode == "mtd":
            from src.mtd_engine.mtd_controller import MTDController
            self.mtd = MTDController(mtd_config_path)
            mtd_cfg = self.mtd.get_active_config()
            self.strategy = mtd_cfg["retriever"]
            self.embedding_model_name = mtd_cfg["embedder"]

        # Determine KB path
        if kb_path:
            self.kb_path = kb_path
        elif mode == "baseline":
            self.kb_path = self.mtd_cfg["kb_pool"][0]
        elif self.mtd is not None:
            self.kb_path = self.mtd.get_active_config()["kb"]
        else:
            self.kb_path = self.mtd_cfg["kb_pool"][0]

        # Track current config for MTD rebuild detection
        self._current_kb = self.kb_path
        self._current_strategy = self.strategy
        self._current_embedder = self.embedding_model_name

        # Build indexes
        self.indexer = DocumentIndexer(config_path=rag_config_path)
        self._build_index(self.kb_path)

        # Init generator
        self.generator = Generator(config_path=rag_config_path)

        logger.info("RAGPipeline initialized: mode=%s, kb=%s, strategy=%s, embedder=%s",
                     mode, self.kb_path, self.strategy, self.embedding_model_name)

    def _build_index(self, kb_path: str) -> None:
        """Build dense/sparse indexes and retriever for a given KB path.

        Args:
            kb_path: Path to the knowledge base directory.
        """
        docs = self.indexer.load_documents(kb_path)
        self.chunks = self.indexer.chunk_documents(docs)
        self.dense_index = self.indexer.build_dense_index(self.chunks)
        self.sparse_index = self.indexer.build_sparse_index(self.chunks)

        self.retriever = Retriever(
            strategy=self._current_strategy,
            dense_index=self.dense_index,
            sparse_index=self.sparse_index,
            chunks=self.chunks,
            encoder=self.indexer.encoder,
            config_path=self.rag_config_path,
        )
        self._current_kb = kb_path

    def _rebuild_retriever_if_needed(self) -> None:
        """Check if MTD config changed and rebuild components accordingly."""
        if self.mtd is None:
            return

        mtd_cfg = self.mtd.get_active_config()
        new_kb = mtd_cfg["kb"]
        new_strategy = mtd_cfg["retriever"]

        needs_rebuild = False

        if new_kb != self._current_kb:
            logger.info("MTD: KB changed %s → %s, rebuilding index",
                        self._current_kb, new_kb)
            needs_rebuild = True

        if new_strategy != self._current_strategy:
            logger.info("MTD: Retriever changed %s → %s",
                        self._current_strategy, new_strategy)
            self._current_strategy = new_strategy
            needs_rebuild = True

        if needs_rebuild:
            self._build_index(new_kb)

    def run(self, query: str) -> PipelineOutput:
        """Run the full RAG pipeline for a single query.

        In MTD mode, triggers MTD step after retrieval and rebuilds
        components for the next query if configuration changed.

        Args:
            query: User query string.

        Returns:
            PipelineOutput with answer, retrieved chunks, and latency metrics.
        """
        # In MTD mode, check if we need to rebuild before this query
        self._rebuild_retriever_if_needed()

        # Retrieval phase
        retrieved, retrieval_ms = _measure_latency(
            self.retriever.retrieve, query, self.top_k
        )

        # Generation phase
        gen_output, generation_ms = _measure_latency(
            self.generator.generate, query, retrieved
        )

        # MTD step: update state for next query
        if self.mtd is not None:
            scores = [r.score for r in retrieved]
            self.mtd.step(query, scores)

        # Determine active config for output
        if self.mtd is not None:
            mtd_cfg = self.mtd.get_active_config()
            active_kb = self._current_kb
            active_retriever = self._current_strategy
            active_embedder = mtd_cfg["embedder"]
        else:
            active_kb = self.kb_path
            active_retriever = self.strategy
            active_embedder = self.embedding_model_name

        output = PipelineOutput(
            answer=gen_output.answer,
            retrieved_chunks=retrieved,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
            active_kb=active_kb,
            active_retriever=active_retriever,
            active_embedder=active_embedder,
        )

        logger.info(
            "Pipeline run: query='%s', retr_ms=%.1f, gen_ms=%.1f, kb=%s, strategy=%s",
            query[:50], retrieval_ms, generation_ms, active_kb, active_retriever,
        )
        return output

    def run_batch(self, queries: list[str]) -> list[PipelineOutput]:
        """Run the pipeline on a batch of queries sequentially.

        Args:
            queries: List of query strings.

        Returns:
            List of PipelineOutput for each query.
        """
        logger.info("Starting batch run: %d queries", len(queries))
        start = time.perf_counter()
        outputs = [self.run(q) for q in queries]
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("Batch complete: %d queries in %.1f ms", len(queries), elapsed)
        return outputs


def _measure_latency(fn: Callable, *args: Any) -> tuple[Any, float]:
    """Wrap a callable and measure wall-clock execution time.

    Args:
        fn: Callable to execute.
        *args: Arguments to pass to the callable.

    Returns:
        Tuple of (result, elapsed_ms).
    """
    start = time.perf_counter()
    result = fn(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms
