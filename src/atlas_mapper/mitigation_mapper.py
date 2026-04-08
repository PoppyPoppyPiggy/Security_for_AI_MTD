# =============================================================================
# FILE: src/atlas_mapper/mitigation_mapper.py
# DESC: MITRE ATLAS TTP ↔ MTD mitigation mapping (Table I for paper)
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# REF:   ATLAS-MTD-RAG framework — SDR triad mapping
# DEPS: src/atlas_mapper/ttp_definitions.py
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.atlas_mapper.ttp_definitions import ATLASRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MTDMitigation:
    """Maps an ATLAS TTP to its MTD defense mechanism."""

    ttp_id: str
    mtd_strategy: str       # "Shuffling" | "Diversity" | "Redundancy"
    mtd_component: str       # implementation module name
    mitigation_description: str
    effectiveness: str       # "High" | "Medium" | "Low"
    effectiveness_rationale: str


class MitigationMapper:
    """Maps MITRE ATLAS TTPs to MTD defense mitigations.

    Provides the SDR (Shuffling–Diversity–Redundancy) triad mapping
    used in Table I of the INTCOM 2026 paper.
    """

    MAPPING: list[MTDMitigation] = [
        MTDMitigation(
            ttp_id="AML.T0054",
            mtd_strategy="Shuffling",
            mtd_component="kb_rotator",
            mitigation_description=(
                "KB rotation invalidates poisoned documents by switching "
                "to a clean KB pool member before the next query epoch."
            ),
            effectiveness="High",
            effectiveness_rationale=(
                "PoisonedRAG requires static KB; rotation reduces "
                "per-epoch exposure to ≤1/N of queries."
            ),
        ),
        MTDMitigation(
            ttp_id="AML.T0020",
            mtd_strategy="Diversity",
            mtd_component="retriever_switcher",
            mitigation_description=(
                "Switching retrieval strategy (dense→sparse→hybrid) "
                "disrupts gradient-optimized adversarial passages that "
                "target a single retriever's embedding space."
            ),
            effectiveness="High",
            effectiveness_rationale=(
                "Semantic Chameleon (2026) shows hybrid retrieval "
                "reduces ASR from 38% to 0% against dense-optimized attacks."
            ),
        ),
        MTDMitigation(
            ttp_id="AML.T0056",
            mtd_strategy="Shuffling",
            mtd_component="kb_rotator",
            mitigation_description=(
                "KB rotation changes the document surface between "
                "reconnaissance and attack, invalidating probed structure."
            ),
            effectiveness="Medium",
            effectiveness_rationale=(
                "Reconnaissance is disrupted but not eliminated; "
                "attacker must re-probe after each rotation epoch."
            ),
        ),
        MTDMitigation(
            ttp_id="AML.T0057",
            mtd_strategy="Diversity",
            mtd_component="retriever_switcher",
            mitigation_description=(
                "Retrieval strategy switching makes it harder to "
                "reliably exfiltrate target documents across queries."
            ),
            effectiveness="Medium",
            effectiveness_rationale=(
                "Exfiltration requires consistent retrieval; strategy "
                "rotation introduces query-level unpredictability."
            ),
        ),
        MTDMitigation(
            ttp_id="AML.T0051",
            mtd_strategy="Redundancy",
            mtd_component="embed_rotator",
            mitigation_description=(
                "Multi-model ensemble averaging dilutes injected "
                "instructions that were optimized for a single "
                "embedding model's retrieval surface."
            ),
            effectiveness="Medium",
            effectiveness_rationale=(
                "Ensemble reduces single-model injection success but "
                "cannot fully block model-agnostic plain-text injections."
            ),
        ),
    ]

    @classmethod
    def get_mitigations_for_ttp(cls, ttp_id: str) -> list[MTDMitigation]:
        """Get all mitigations for a specific TTP.

        Args:
            ttp_id: ATLAS technique ID.

        Returns:
            List of MTDMitigation for that TTP.
        """
        return [m for m in cls.MAPPING if m.ttp_id == ttp_id]

    @classmethod
    def get_mitigations_by_strategy(cls, strategy: str) -> list[MTDMitigation]:
        """Get all mitigations using a specific MTD strategy.

        Args:
            strategy: "Shuffling", "Diversity", or "Redundancy".

        Returns:
            List of MTDMitigation with that strategy.
        """
        return [m for m in cls.MAPPING if m.mtd_strategy == strategy]

    @classmethod
    def get_high_effectiveness(cls) -> list[MTDMitigation]:
        """Get all mitigations with High effectiveness.

        Returns:
            List of MTDMitigation with effectiveness="High".
        """
        return [m for m in cls.MAPPING if m.effectiveness == "High"]

    @classmethod
    def generate_table_I(cls, save_path: str = "docs/table_I_atlas_mapping.txt") -> str:
        """Generate paper-ready Table I as formatted ASCII.

        Produces the ATLAS TTP ↔ MTD mitigation mapping table for the
        INTCOM 2026 paper.

        Args:
            save_path: File path to save the table.

        Returns:
            Table as formatted string.
        """
        registry = ATLASRegistry

        # Column widths
        w_ttp = 13
        w_name = 30
        w_strat = 12
        w_comp = 20
        w_eff = 13

        sep = (
            f"+-{'-' * w_ttp}-+-{'-' * w_name}-+"
            f"-{'-' * w_strat}-+-{'-' * w_comp}-+-{'-' * w_eff}-+"
        )

        header = (
            f"| {'ATLAS TTP':<{w_ttp}} | {'Technique Name':<{w_name}} |"
            f" {'MTD Strategy':<{w_strat}} | {'Component':<{w_comp}} |"
            f" {'Effectiveness':<{w_eff}} |"
        )

        lines = [
            "Table I: MITRE ATLAS TTP to MTD Mitigation Mapping",
            "",
            sep,
            header,
            sep,
        ]

        for m in cls.MAPPING:
            tech = registry.get(m.ttp_id)
            name = tech.name if tech else m.ttp_id
            row = (
                f"| {m.ttp_id:<{w_ttp}} | {name:<{w_name}} |"
                f" {m.mtd_strategy:<{w_strat}} | {m.mtd_component:<{w_comp}} |"
                f" {m.effectiveness:<{w_eff}} |"
            )
            lines.append(row)

        lines.append(sep)
        lines.append("")
        lines.append("SDR Triad: Shuffling (S) — Diversity (D) — Redundancy (R)")
        lines.append("Effectiveness: High = primary mitigation, Medium = supporting defense")

        table_str = "\n".join(lines)

        # Save to file
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table_str, encoding="utf-8")
        logger.info("Table I saved to %s", save_path)

        return table_str
