# =============================================================================
# FILE: src/evaluation/comparative_table.py
# DESC: Paper Table II generator — compares ATLAS-MTD-RAG vs existing defenses
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# REF:   RobustRAG, TrustRAG, SeCon-RAG, ReliabilityRAG, RAGuard
# =============================================================================
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameworkEntry:
    """A row in the comparative Table II."""

    name: str
    venue: str
    year: int
    dynamic_defense: bool
    training_free: bool
    certified_robust: bool
    atlas_mapped: bool
    asr_reduction: str          # reported ASR or reduction, e.g. "0.02" or "varies"
    faithfulness: str           # reported faithfulness or "—"
    notes: str


# Literature-reported values for comparison frameworks
BASELINE_FRAMEWORKS: list[FrameworkEntry] = [
    FrameworkEntry(
        name="RobustRAG",
        venue="ICML 2024",
        year=2024,
        dynamic_defense=False,
        training_free=True,
        certified_robust=True,
        atlas_mapped=False,
        asr_reduction="certified",
        faithfulness="—",
        notes="Isolate-then-aggregate; provable robustness guarantee",
    ),
    FrameworkEntry(
        name="TrustRAG",
        venue="AAAI-W 2026",
        year=2026,
        dynamic_defense=False,
        training_free=True,
        certified_robust=False,
        atlas_mapped=False,
        asr_reduction="~0.10",
        faithfulness="—",
        notes="Cluster filtering + LLM self-assessment; plug-and-play",
    ),
    FrameworkEntry(
        name="SeCon-RAG",
        venue="NeurIPS 2025",
        year=2025,
        dynamic_defense=False,
        training_free=False,
        certified_robust=False,
        atlas_mapped=False,
        asr_reduction="0.02",
        faithfulness="—",
        notes="Semantic + conflict-aware filtering; entity-intent extractor",
    ),
    FrameworkEntry(
        name="ReliabilityRAG",
        venue="NeurIPS 2025",
        year=2025,
        dynamic_defense=False,
        training_free=True,
        certified_robust=True,
        atlas_mapped=False,
        asr_reduction="provable",
        faithfulness="—",
        notes="Graph-theoretic MIS on document contradiction graph",
    ),
    FrameworkEntry(
        name="RAGuard",
        venue="NeurIPS 2025",
        year=2025,
        dynamic_defense=False,
        training_free=False,
        certified_robust=False,
        atlas_mapped=False,
        asr_reduction="varies",
        faithfulness="—",
        notes="Adversarial training + zero-knowledge inference patch",
    ),
    FrameworkEntry(
        name="RAGShield",
        venue="arXiv 2026",
        year=2026,
        dynamic_defense=False,
        training_free=True,
        certified_robust=False,
        atlas_mapped=False,
        asr_reduction="0.00",
        faithfulness="—",
        notes="5-layer provenance verification; C2PA-inspired attestation",
    ),
]


def generate_table_II(
    our_results: dict | None = None,
    save_ascii: str = "docs/table_II_comparison.txt",
    save_latex: str = "docs/table_II_comparison.tex",
) -> str:
    """Generate Table II comparing our framework against baselines.

    Args:
        our_results: Dict from extended_evaluator comparison results.
            Expected keys: baseline.metrics.asr, mtd.metrics.asr,
            mtd.faithfulness.score, narr
        save_ascii: Path to save ASCII table.
        save_latex: Path to save LaTeX table.

    Returns:
        ASCII table string.
    """
    # Build our entry from results
    our_asr = "—"
    our_faith = "—"
    our_narr = "—"
    if our_results:
        mtd_metrics = our_results.get("mtd", {}).get("metrics", {})
        our_asr = f"{mtd_metrics.get('asr', 0):.2f}"
        faith = our_results.get("mtd", {}).get("faithfulness", {})
        if faith:
            our_faith = f"{faith.get('score', 0):.2f}"
        our_narr = f"{our_results.get('narr', 0):.2f}"

    our_entry = FrameworkEntry(
        name="ATLAS-MTD-RAG (Ours)",
        venue="INTCOM 2026",
        year=2026,
        dynamic_defense=True,
        training_free=True,
        certified_robust=False,
        atlas_mapped=True,
        asr_reduction=our_asr,
        faithfulness=our_faith,
        notes=f"SDR triad; KB/retriever/embed rotation; NARR={our_narr}",
    )

    all_entries = BASELINE_FRAMEWORKS + [our_entry]

    ascii_table = _render_ascii(all_entries)
    latex_table = _render_latex(all_entries)

    # Save files
    out_dir = Path(save_ascii).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(save_ascii).write_text(ascii_table, encoding="utf-8")
    Path(save_latex).write_text(latex_table, encoding="utf-8")

    logger.info("Table II saved: %s, %s", save_ascii, save_latex)
    return ascii_table


def _bool_mark(val: bool) -> str:
    """Convert bool to table marker."""
    return "Y" if val else "N"


def _render_ascii(entries: list[FrameworkEntry]) -> str:
    """Render ASCII table."""
    w = [22, 14, 7, 8, 8, 7, 10, 10]  # column widths
    sep = "+" + "+".join("-" * (wi + 2) for wi in w) + "+"

    header = (
        f"| {'Framework':<{w[0]}} | {'Venue':<{w[1]}} | {'Dyn.':<{w[2]}} | "
        f"{'TrnFree':<{w[3]}} | {'Certif.':<{w[4]}} | {'ATLAS':<{w[5]}} | "
        f"{'ASR_mtd':<{w[6]}} | {'Faith.':<{w[7]}} |"
    )

    lines = [
        "Table II: Comparison with Existing RAG Defense Frameworks",
        "",
        sep,
        header,
        sep,
    ]

    for e in entries:
        row = (
            f"| {e.name:<{w[0]}} | {e.venue:<{w[1]}} | "
            f"{_bool_mark(e.dynamic_defense):<{w[2]}} | "
            f"{_bool_mark(e.training_free):<{w[3]}} | "
            f"{_bool_mark(e.certified_robust):<{w[4]}} | "
            f"{_bool_mark(e.atlas_mapped):<{w[5]}} | "
            f"{e.asr_reduction:<{w[6]}} | "
            f"{e.faithfulness:<{w[7]}} |"
        )
        lines.append(row)

    lines.append(sep)
    lines.append("")
    lines.append("Dyn. = Dynamic Defense, TrnFree = Training-Free, Certif. = Certified Robust")
    lines.append("ATLAS = MITRE ATLAS TTP Mapped, ASR_mtd = Attack Success Rate after defense")
    lines.append("Faith. = Faithfulness Score (RAGChecker methodology)")
    return "\n".join(lines)


def _render_latex(entries: list[FrameworkEntry]) -> str:
    """Render LaTeX tabular for paper inclusion."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison with Existing RAG Defense Frameworks}",
        r"\label{tab:comparison}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Framework & Venue & Dynamic & Train-Free & Certified & ATLAS & ASR$\downarrow$ & Faithful \\",
        r"\midrule",
    ]

    for e in entries:
        ck = lambda v: r"\checkmark" if v else "—"
        name = e.name.replace("_", r"\_")
        if "(Ours)" in name:
            name = r"\textbf{" + name + "}"
        row = (
            f"  {name} & {e.venue} & {ck(e.dynamic_defense)} & "
            f"{ck(e.training_free)} & {ck(e.certified_robust)} & "
            f"{ck(e.atlas_mapped)} & {e.asr_reduction} & {e.faithfulness} \\\\"
        )
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    """Generate Table II from latest results."""
    logging.basicConfig(level=logging.INFO)

    # Try to load latest comparison results
    results_dir = Path("data/results")
    our_results = None
    if results_dir.exists():
        bench_files = sorted(results_dir.glob("bench_*_compare_*.json"), reverse=True)
        if bench_files:
            our_results = json.loads(bench_files[0].read_text())

    table = generate_table_II(our_results)
    print(table)


if __name__ == "__main__":
    main()
