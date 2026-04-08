# =============================================================================
# FILE: src/attacker/corpus_poison.py
# DESC: Corpus-level poisoning — adversarial passage crafting & injection
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# ATLAS: AML.T0020 — Poison Training Data (KB equivalent)
# REF:   Zhong et al., EMNLP 2023; BadRAG (Xue et al., 2024)
# DEPS: config/attack_config.yaml, src/rag_pipeline/indexer.py
# =============================================================================
from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from src.rag_pipeline.indexer import Chunk

logger = logging.getLogger(__name__)


@dataclass
class AdversarialPassage:
    """An adversarial passage crafted for corpus poisoning."""

    text: str
    trigger_phrase: str
    similarity_to_query: float


@dataclass
class InjectionReport:
    """Summary of a corpus injection operation."""

    total_docs: int
    injected: int
    injection_pct: float


class CorpusPoisonAttacker:
    """Corpus-level poisoning attacker for RAG knowledge bases.

    Supports two attacker levels:
      - L1 (black-box): Generates passages containing target_answer,
        boosts similarity by appending synonyms of query keywords.
      - L2 (gray-box, retriever-aware): Uses embedding model to iteratively
        maximize sim(embed(passage), embed(target_query)) via gradient-free
        hill-climbing by word substitution (up to 50 iterations).

    Args:
        config_path: Path to attack_config.yaml.

    ATLAS:
        AML.T0020 — Poison Training Data (KB equivalent).
        Adversarial passages are injected into the corpus to manipulate
        retrieval results for target queries.
    """

    def __init__(self, config_path: str = "config/attack_config.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        atk_cfg = cfg["attack"]

        self.attacker_level: str = atk_cfg["attacker_level"]
        self.num_poison_docs: int = atk_cfg["num_poison_docs"]
        self._encoder: SentenceTransformer | None = None

        logger.info("CorpusPoisonAttacker initialized: level=%s", self.attacker_level)

    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._encoder

    def craft_adversarial_passage(
        self,
        target_query: str,
        target_answer: str,
        existing_chunks: list[Chunk],
        attacker_level: str = "L1",
    ) -> AdversarialPassage:
        """Craft an adversarial passage for the target query.

        Args:
            target_query: The query to attack.
            target_answer: Desired attacker answer.
            existing_chunks: Current KB chunks (used for context in L2).
            attacker_level: 'L1' (black-box) or 'L2' (gray-box).

        Returns:
            AdversarialPassage with crafted text and similarity score.
        """
        if attacker_level == "L1":
            return self._craft_l1(target_query, target_answer)
        elif attacker_level == "L2":
            return self._craft_l2(target_query, target_answer, existing_chunks)
        else:
            raise ValueError(f"Unsupported attacker level: {attacker_level}")

    def _craft_l1(
        self,
        target_query: str,
        target_answer: str,
    ) -> AdversarialPassage:
        """L1 black-box: generate passage with keyword synonym boosting."""
        stop_words = {"what", "is", "the", "a", "an", "how", "does", "do",
                      "and", "it", "of", "to", "for", "in", "why", "are"}
        keywords = [
            w.strip("?.,!") for w in target_query.lower().split()
            if w.strip("?.,!") not in stop_words and len(w) > 2
        ]
        trigger_phrase = " ".join(keywords)

        # Build passage with target answer and keyword repetitions
        text = (
            f"Regarding {trigger_phrase}: {target_answer}. "
            f"This is the verified answer about {trigger_phrase}. "
            f"{target_answer} is confirmed by authoritative sources. "
            f"Key topics: {', '.join(keywords)}."
        )

        sim = self._compute_similarity(text, target_query)
        logger.info("L1 passage: sim=%.4f, trigger='%s'", sim, trigger_phrase)

        return AdversarialPassage(
            text=text, trigger_phrase=trigger_phrase, similarity_to_query=sim,
        )

    def _craft_l2(
        self,
        target_query: str,
        target_answer: str,
        existing_chunks: list[Chunk],
    ) -> AdversarialPassage:
        """L2 gray-box: hill-climb word substitution to maximize embedding similarity."""
        # Start from L1 passage as base
        base_passage = self._craft_l1(target_query, target_answer)
        best_text = base_passage.text
        best_sim = base_passage.similarity_to_query

        # Collect vocabulary from existing chunks for substitution candidates
        vocab: list[str] = []
        for chunk in existing_chunks[:20]:  # sample from first 20 chunks
            vocab.extend(w.lower().strip(".,!?;:") for w in chunk.text.split()
                         if len(w) > 3)
        vocab = list(set(vocab))
        if not vocab:
            vocab = [target_answer.lower()]

        # Hill-climbing: try word substitutions / appends
        query_emb = self.encoder.encode([target_query], convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        for iteration in range(50):
            if best_sim >= 0.90:
                break

            # Try appending a random vocab word or query-related phrase
            if random.random() < 0.5 and vocab:
                candidate = best_text + f" {random.choice(vocab)}"
            else:
                candidate = best_text + f" {target_answer} is the answer."

            cand_emb = self.encoder.encode([candidate], convert_to_numpy=True)
            cand_emb = cand_emb / np.linalg.norm(cand_emb, axis=1, keepdims=True)
            sim = float(np.dot(query_emb, cand_emb.T)[0, 0])

            if sim > best_sim:
                best_text = candidate
                best_sim = sim

        logger.info("L2 passage: sim=%.4f after %d iterations", best_sim, iteration + 1)

        return AdversarialPassage(
            text=best_text,
            trigger_phrase=base_passage.trigger_phrase,
            similarity_to_query=best_sim,
        )

    def _compute_similarity(self, text: str, query: str) -> float:
        """Compute cosine similarity between text and query embeddings."""
        embs = self.encoder.encode([text, query], convert_to_numpy=True)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        return float(np.dot(embs[0], embs[1]))

    def inject_at_corpus_level(
        self,
        corpus_path: str,
        passages: list[AdversarialPassage],
        injection_ratio: float = 0.0004,
        output_path: str = "data/knowledge_bases/kb_poisoned/",
    ) -> InjectionReport:
        """Inject adversarial passages into a corpus copy.

        Args:
            corpus_path: Path to the clean corpus directory.
            passages: List of AdversarialPassage to inject.
            injection_ratio: Fraction of corpus to poison (default 0.04%).
            output_path: Directory for the poisoned corpus.

        Returns:
            InjectionReport with injection statistics.
        """
        src = Path(corpus_path)
        dst = Path(output_path)
        dst.mkdir(parents=True, exist_ok=True)

        # Copy original corpus
        total_docs = 0
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
                total_docs += 1

        # Determine how many docs to inject based on ratio
        # At minimum, inject all provided passages
        target_injected = max(len(passages), int(total_docs * injection_ratio))
        actual_injected = min(target_injected, len(passages))

        # Write adversarial passages
        for i, passage in enumerate(passages[:actual_injected]):
            fname = f"corpus_poison_{i}.txt"
            (dst / fname).write_text(passage.text, encoding="utf-8")
            total_docs += 1

        injection_pct = actual_injected / total_docs if total_docs > 0 else 0.0

        report = InjectionReport(
            total_docs=total_docs,
            injected=actual_injected,
            injection_pct=round(injection_pct, 6),
        )
        logger.info("Corpus injection: %d/%d docs (%.4f%%)",
                     actual_injected, total_docs, injection_pct * 100)
        return report

    def compute_injection_stats(self, report: InjectionReport) -> dict:
        """Compute injection statistics from a report.

        Args:
            report: InjectionReport from inject_at_corpus_level.

        Returns:
            Dict with total_docs, injected, injection_pct, target_coverage.
        """
        return {
            "total_docs": report.total_docs,
            "injected": report.injected,
            "injection_pct": report.injection_pct,
            "target_coverage": report.injected / max(report.total_docs, 1),
        }
