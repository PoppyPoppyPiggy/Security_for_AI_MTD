# =============================================================================
# FILE: src/atlas_mapper/ttp_definitions.py
# DESC: MITRE ATLAS TTP definitions relevant to RAG pipeline attacks
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# REF:   MITRE ATLAS — https://atlas.mitre.org
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ATLASTechnique:
    """A single MITRE ATLAS technique definition."""

    id: str
    name: str
    tactic: str
    description: str
    rag_relevant: bool
    attack_stage: str  # "ingestion" | "retrieval" | "generation"


@dataclass(frozen=True)
class ATLASTactic:
    """A MITRE ATLAS tactic grouping techniques."""

    id: str
    name: str
    techniques: tuple[ATLASTechnique, ...]


class ATLASRegistry:
    """Registry of MITRE ATLAS TTPs relevant to RAG pipeline defense.

    Provides lookup by TTP ID, attack stage, and RAG relevance.
    All techniques included are directly relevant to the ATLAS-MTD-RAG
    defense framework scope.
    """

    TECHNIQUES: dict[str, ATLASTechnique] = {
        "AML.T0054": ATLASTechnique(
            id="AML.T0054",
            name="False RAG Entry Injection",
            tactic="ML Attack Staging",
            description=(
                "Adversary injects malicious documents into a RAG "
                "knowledge base to manipulate LLM outputs."
            ),
            rag_relevant=True,
            attack_stage="ingestion",
        ),
        "AML.T0020": ATLASTechnique(
            id="AML.T0020",
            name="Poison Training Data",
            tactic="ML Attack Staging",
            description=(
                "Adversary corrupts training or knowledge corpus "
                "to alter model behavior at inference time."
            ),
            rag_relevant=True,
            attack_stage="ingestion",
        ),
        "AML.T0056": ATLASTechnique(
            id="AML.T0056",
            name="Gather RAG-Indexed Targets",
            tactic="Reconnaissance",
            description=(
                "Adversary probes retrieval system to enumerate "
                "indexed documents before launching poisoning."
            ),
            rag_relevant=True,
            attack_stage="retrieval",
        ),
        "AML.T0057": ATLASTechnique(
            id="AML.T0057",
            name="RAG Database Prompting",
            tactic="Exfiltration via ML Inference API",
            description=(
                "Adversary extracts sensitive documents by crafting "
                "queries that retrieve target content."
            ),
            rag_relevant=True,
            attack_stage="retrieval",
        ),
        "AML.T0051": ATLASTechnique(
            id="AML.T0051",
            name="LLM Prompt Injection",
            tactic="Initial Access",
            description=(
                "Adversary embeds hidden instructions inside "
                "retrieved content to hijack LLM behavior."
            ),
            rag_relevant=True,
            attack_stage="generation",
        ),
    }

    @classmethod
    def get(cls, ttp_id: str) -> ATLASTechnique | None:
        """Look up a technique by its ATLAS ID.

        Args:
            ttp_id: ATLAS technique ID (e.g. "AML.T0054").

        Returns:
            ATLASTechnique if found, None otherwise.
        """
        return cls.TECHNIQUES.get(ttp_id)

    @classmethod
    def get_by_stage(cls, stage: str) -> list[ATLASTechnique]:
        """Get all techniques targeting a specific attack stage.

        Args:
            stage: One of "ingestion", "retrieval", "generation".

        Returns:
            List of ATLASTechnique matching the stage.
        """
        return [t for t in cls.TECHNIQUES.values() if t.attack_stage == stage]

    @classmethod
    def get_rag_relevant(cls) -> list[ATLASTechnique]:
        """Get all RAG-relevant techniques.

        Returns:
            List of ATLASTechnique where rag_relevant is True.
        """
        return [t for t in cls.TECHNIQUES.values() if t.rag_relevant]

    @classmethod
    def list_all(cls) -> list[ATLASTechnique]:
        """List all registered techniques.

        Returns:
            List of all ATLASTechnique in the registry.
        """
        return list(cls.TECHNIQUES.values())
