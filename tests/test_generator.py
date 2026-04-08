# =============================================================================
# FILE: tests/test_generator.py
# DESC: Unit tests for Generator — extractive and llama_cpp modes
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: src/rag_pipeline/generator.py
# =============================================================================
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag_pipeline.generator import Generator, GeneratorOutput, PROMPT_TEMPLATE
from src.rag_pipeline.indexer import Chunk
from src.rag_pipeline.retriever import RetrievalResult

GGUF_PATH = "models/tinyllama-1.1b-q4.gguf"
GGUF_AVAILABLE = Path(GGUF_PATH).exists()


@pytest.fixture
def mock_chunks() -> list[RetrievalResult]:
    """Create mock retrieval results for testing."""
    return [
        RetrievalResult(
            chunk=Chunk(text="A firewall monitors network traffic.", source_path="t.txt",
                        chunk_id="c0", doc_id="d0"),
            score=0.95, rank=1,
        ),
        RetrievalResult(
            chunk=Chunk(text="Encryption protects data in transit.", source_path="t.txt",
                        chunk_id="c1", doc_id="d1"),
            score=0.85, rank=2,
        ),
    ]


class TestBuildContext:
    def test_joins_chunks(self, mock_chunks: list[RetrievalResult]) -> None:
        """Context should join chunk texts with --- separator."""
        gen = Generator(mode="extractive")
        context = gen._build_context(mock_chunks)
        assert "---" in context
        assert "firewall" in context
        assert "Encryption" in context

    def test_empty_chunks(self) -> None:
        """Empty chunk list should produce empty context."""
        gen = Generator(mode="extractive")
        context = gen._build_context([])
        assert context == ""


class TestExtractiveGenerate:
    def test_answer_not_empty(self, mock_chunks: list[RetrievalResult]) -> None:
        """Generate should return a non-empty answer (extractive mode)."""
        gen = Generator(mode="extractive")
        output = gen.generate("What is a firewall?", mock_chunks)
        assert isinstance(output, GeneratorOutput)
        assert len(output.answer) > 0
        assert output.token_count > 0

    def test_context_injection(self, mock_chunks: list[RetrievalResult]) -> None:
        """Extractive answer should echo the top-ranked chunk."""
        gen = Generator(mode="extractive")
        output = gen.generate("What is a firewall?", mock_chunks)
        assert "firewall" in output.answer.lower()

    def test_unknown_answer(self) -> None:
        """Should return 'I don't know.' when no chunks are provided."""
        gen = Generator(mode="extractive")
        output = gen.generate("What is quantum computing?", [])
        assert "don't know" in output.answer.lower()

    def test_prompt_template_used(self, mock_chunks: list[RetrievalResult]) -> None:
        """The prompt_used field should contain the template structure."""
        gen = Generator(mode="extractive")
        output = gen.generate("test query", mock_chunks)
        assert "Answer using ONLY" in output.prompt_used
        assert "test query" in output.prompt_used


class TestLlamaCppGenerate:
    def test_missing_model_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError if GGUF model path does not exist."""
        import yaml
        cfg = {"rag": {
            "llm_provider": "llama_cpp",
            "llm_model_path": str(tmp_path / "nonexistent.gguf"),
            "max_tokens": 64, "temperature": 0.0,
        }}
        cfg_path = tmp_path / "rag.yaml"
        cfg_path.write_text(yaml.dump(cfg))
        with pytest.raises(FileNotFoundError, match="GGUF model not found"):
            Generator(config_path=str(cfg_path))

    @pytest.mark.skipif(not GGUF_AVAILABLE, reason="GGUF model not found")
    def test_answer_not_empty(self, mock_chunks: list[RetrievalResult]) -> None:
        """GGUF model should return a non-empty answer."""
        gen = Generator(mode="llama_cpp")
        output = gen.generate("What is a firewall?", mock_chunks)
        assert isinstance(output, GeneratorOutput)
        assert len(output.answer) > 0
        assert output.token_count > 0

    @pytest.mark.skipif(not GGUF_AVAILABLE, reason="GGUF model not found")
    def test_context_used(self, mock_chunks: list[RetrievalResult]) -> None:
        """Answer should reference content from provided context."""
        gen = Generator(mode="llama_cpp")
        output = gen.generate("What is a firewall?", mock_chunks)
        answer_lower = output.answer.lower()
        assert any(w in answer_lower for w in ["firewall", "network", "traffic", "monitor"])

    @pytest.mark.skipif(not GGUF_AVAILABLE, reason="GGUF model not found")
    def test_seed_consistent_topic(self, mock_chunks: list[RetrievalResult]) -> None:
        """Same query+context should produce topically consistent answers."""
        gen = Generator(mode="llama_cpp")
        out1 = gen.generate("What is a firewall?", mock_chunks)
        out2 = gen.generate("What is a firewall?", mock_chunks)
        # Both answers should mention firewall-related content
        for out in [out1, out2]:
            assert any(w in out.answer.lower() for w in ["firewall", "network", "traffic"])
