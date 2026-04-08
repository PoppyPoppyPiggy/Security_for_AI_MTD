# =============================================================================
# FILE: src/rag_pipeline/generator.py
# DESC: LLM response generation for RAG pipeline — extractive (local) and
#       llama.cpp GGUF-based modes.
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: config/rag_config.yaml, llama-cpp-python (for llama_cpp mode)
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.rag_pipeline.retriever import RetrievalResult
from src.utils import safe_load_config

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "<|system|>\n"
    "Answer using ONLY the provided context. "
    "If the answer is not in the context, say 'I don't know.'\n</s>\n"
    "<|user|>\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n</s>\n"
    "<|assistant|>\n"
)


@dataclass
class GeneratorOutput:
    """Output from the LLM generator."""

    answer: str
    prompt_used: str
    token_count: int


class Generator:
    """Generates answers from retrieved context.

    Supports two modes controlled by config['llm_provider']:
      - 'extractive' (default): Returns the top-ranked chunk text as the answer.
        Simulates a perfectly faithful LLM that echoes retrieved context — the
        standard assumption in PoisonedRAG (Zou 2025) attack evaluation.
      - 'llama_cpp': Loads a local GGUF model via llama-cpp-python for
        real LLM inference at zero API cost.

    Args:
        config_path: Path to rag_config.yaml.
        mode: Override generation mode — 'extractive' or 'llama_cpp'.
              If None, reads from config['llm_provider'].

    ATLAS:
        Target component for AML.T0051 (LLM Prompt Injection — Indirect).
        Poisoned context passed to the generator can hijack output.
    """

    def __init__(
        self,
        config_path: str = "config/rag_config.yaml",
        mode: str | None = None,
    ) -> None:
        cfg = safe_load_config(config_path)
        rag_cfg = cfg["rag"]

        self.max_tokens: int = rag_cfg.get("max_tokens", 256)
        self.temperature: float = rag_cfg.get("temperature", 0.0)
        self.context_max_chars: int = rag_cfg.get("context_max_chars", 3000)
        self.seed: int = rag_cfg.get("llm_seed", 42)
        self.mode: str = mode or rag_cfg.get("llm_provider", "extractive")
        self._llm = None

        if self.mode == "llama_cpp":
            self._init_llama_cpp(rag_cfg)

        logger.info("Generator initialized: mode=%s, max_tokens=%d",
                     self.mode, self.max_tokens)

    def _init_llama_cpp(self, rag_cfg: dict) -> None:
        """Initialize llama.cpp model from GGUF file."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for llama_cpp mode.\n"
                "Install: pip install llama-cpp-python"
            )

        model_path = rag_cfg["llm_model_path"]
        if not Path(model_path).exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        n_ctx = rag_cfg.get("llm_n_ctx", 2048)
        n_threads = rag_cfg.get("llm_n_threads", 4)

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        logger.info("GGUF model loaded: %s (n_ctx=%d, n_threads=%d)",
                     model_path, n_ctx, n_threads)

    def generate(self, query: str, retrieved_chunks: list[RetrievalResult]) -> GeneratorOutput:
        """Generate an answer from retrieved context.

        Args:
            query: User query string.
            retrieved_chunks: List of RetrievalResult from retriever.

        Returns:
            GeneratorOutput with answer, prompt used, and token count.
        """
        context = self._build_context(retrieved_chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, query=query)

        if self.mode == "llama_cpp":
            return self._generate_llama_cpp(prompt, query)

        return self._generate_extractive(prompt, query, retrieved_chunks)

    def _generate_extractive(
        self,
        prompt: str,
        query: str,
        retrieved_chunks: list[RetrievalResult],
    ) -> GeneratorOutput:
        if not retrieved_chunks:
            answer = "I don't know."
        else:
            answer = retrieved_chunks[0].chunk.text

        token_count = len(prompt.split()) + len(answer.split())
        logger.info("Extractive answer: %d tokens, query='%s'", token_count, query[:50])
        return GeneratorOutput(answer=answer, prompt_used=prompt, token_count=token_count)

    def _generate_llama_cpp(self, prompt: str, query: str) -> GeneratorOutput:
        output = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            echo=False,
            seed=self.seed,
        )
        answer = output["choices"][0]["text"].strip()
        token_count = output["usage"]["completion_tokens"]
        logger.info("GGUF answer: %d tokens, query='%s'", token_count, query[:50])
        return GeneratorOutput(answer=answer, prompt_used=prompt, token_count=token_count)

    def _build_context(self, chunks: list[RetrievalResult]) -> str:
        joined = "\n---\n".join(r.chunk.text for r in chunks)
        return joined[:self.context_max_chars]
