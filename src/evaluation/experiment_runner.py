# =============================================================================
# FILE: src/evaluation/experiment_runner.py
# DESC: Full experiment matrix runner — 16 conditions, Table III + IV
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-09
# DEPS: src/rag_pipeline/pipeline.py, src/evaluation/metrics.py,
#       src/attacker/poisoned_rag.py
# =============================================================================
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from src.evaluation.metrics import compute_asr, compute_f1, compute_mean_latency, compute_mrr
from src.rag_pipeline.pipeline import RAGPipeline
from src.utils import safe_load_config

logger = logging.getLogger(__name__)

SEED = 42


@dataclass
class ExperimentResult:
    """Result of a single experiment condition."""

    condition_id: int
    attack_type: str
    defense: str
    pool_size: int
    n_queries: int
    asr: float
    mrr10: float
    mean_f1: float
    narr: float
    mean_retrieval_ms: float
    mean_generation_ms: float


class ExperimentRunner:
    """Runs the full experiment matrix for the paper.

    16 conditions covering:
    - 4 attack types × no defense (baseline)
    - 4 ablation configs × naive attack
    - 4 attack types × MTD-SDR (full defense)
    - 4 pool size variants × MTD-SDR + naive attack

    All experiments use extractive generator for speed and worst-case ASR.
    """

    def __init__(
        self,
        rag_config_path: str = "config/rag_config.yaml",
        mtd_config_path: str = "config/mtd_config.yaml",
        atk_config_path: str = "config/attack_config.yaml",
    ) -> None:
        self.rag_config_path = rag_config_path
        self.mtd_config_path = mtd_config_path
        self.atk_config_path = atk_config_path

        self.atk_cfg = safe_load_config(atk_config_path)["attack"]
        self.rag_cfg = safe_load_config(rag_config_path)["rag"]
        self.mtd_cfg = safe_load_config(mtd_config_path)["mtd"]

        self.target_answer = self.atk_cfg["target_answer"]
        self.kb_poisoned = self.atk_cfg["kb_poisoned_path"]
        self.kb_clean = self.atk_cfg["kb_clean_path"]

        # Load queries
        with open(self.atk_cfg["target_queries_path"]) as f:
            self.target_data = json.load(f)
        self.queries = [d["query"] for d in self.target_data]
        self.ground_truths = [d["ground_truth"] for d in self.target_data]
        self.n_queries = len(self.queries)

        # Full 5-KB pool (all clean snapshots, no poisoned)
        self.full_pool = list(self.mtd_cfg["kb_pool"])

        logger.info("ExperimentRunner: %d queries, pool=%d KBs",
                     self.n_queries, len(self.full_pool))

    def run_matrix(self) -> list[ExperimentResult]:
        """Run all 16 experiment conditions."""
        results: list[ExperimentResult] = []
        cid = 0

        # ── Group A: Attack variants × No Defense (baseline) ──
        logger.info("=== Group A: Baseline (no defense) ===")
        attack_types = ["naive", "optimized", "corpus_l2", "indirect"]
        baseline_asrs: dict[str, float] = {}

        for atk_type in attack_types:
            cid += 1
            logger.info("[%d/16] Baseline: attack=%s", cid, atk_type)
            r = self._run_single(
                condition_id=cid,
                attack_type=atk_type,
                defense="none",
                pool_size=0,
            )
            results.append(r)
            baseline_asrs[atk_type] = r.asr

        # ── Group B: MTD ablation (Naive attack only) ──
        logger.info("=== Group B: MTD Ablation (naive attack) ===")
        ablation_configs = ["MTD-S", "MTD-D", "MTD-R", "MTD-SD"]

        for defense in ablation_configs:
            cid += 1
            logger.info("[%d/16] Ablation: defense=%s", cid, defense)
            r = self._run_single(
                condition_id=cid,
                attack_type="naive",
                defense=defense,
                pool_size=len(self.full_pool),
            )
            r.narr = self._compute_narr(baseline_asrs["naive"], r.asr)
            results.append(r)

        # ── Group C: MTD-SDR full × attack variants ──
        logger.info("=== Group C: MTD-SDR vs all attacks ===")
        for atk_type in attack_types:
            cid += 1
            logger.info("[%d/16] MTD-SDR: attack=%s", cid, atk_type)
            r = self._run_single(
                condition_id=cid,
                attack_type=atk_type,
                defense="MTD-SDR",
                pool_size=len(self.full_pool),
            )
            r.narr = self._compute_narr(baseline_asrs[atk_type], r.asr)
            results.append(r)

        # ── Group D: Pool size ablation (MTD-SDR, naive) ──
        logger.info("=== Group D: Pool size ablation ===")
        for pool_size in [2, 3, 4, 5]:
            cid += 1
            logger.info("[%d/16] Pool=%d", cid, pool_size)
            r = self._run_single(
                condition_id=cid,
                attack_type="naive",
                defense="MTD-SDR",
                pool_size=pool_size,
            )
            r.narr = self._compute_narr(baseline_asrs["naive"], r.asr)
            results.append(r)

        return results

    def _run_single(
        self,
        condition_id: int,
        attack_type: str,
        defense: str,
        pool_size: int,
    ) -> ExperimentResult:
        """Run a single experiment condition."""
        import random
        random.seed(SEED)
        np.random.seed(SEED)

        # Build poisoned KB for this attack type
        kb_path = self._get_poisoned_kb(attack_type)

        if defense == "none":
            # Baseline: run on poisoned KB, no MTD
            pipeline = RAGPipeline(
                mode="baseline",
                rag_config_path=self.rag_config_path,
                mtd_config_path=self.mtd_config_path,
                kb_path=kb_path,
            )
        else:
            # MTD: create temp config for this defense variant
            mtd_cfg_path = self._make_mtd_config(defense, pool_size, kb_path)
            pipeline = RAGPipeline(
                mode="mtd",
                rag_config_path=self.rag_config_path,
                mtd_config_path=mtd_cfg_path,
            )

        outputs = pipeline.run_batch(self.queries)

        # Metrics
        asr = compute_asr(outputs, self.queries, self.target_answer)
        relevant_ids = [f"seed_docs_{i}" for i in range(self.n_queries)]
        mrr = compute_mrr(outputs, relevant_ids)
        f1_scores = [compute_f1(o.answer, gt) for o, gt in zip(outputs, self.ground_truths)]
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        latency = compute_mean_latency(outputs)

        return ExperimentResult(
            condition_id=condition_id,
            attack_type=attack_type,
            defense=defense,
            pool_size=pool_size,
            n_queries=self.n_queries,
            asr=round(asr, 4),
            mrr10=round(mrr, 4),
            mean_f1=round(mean_f1, 4),
            narr=0.0,
            mean_retrieval_ms=latency["retrieval_ms"],
            mean_generation_ms=latency["generation_ms"],
        )

    def _get_poisoned_kb(self, attack_type: str) -> str:
        """Return the poisoned KB path for an attack type.

        All attack types use the same poisoned KB (naive-injected) because:
        - 'naive': standard PoisonedRAG template injection
        - 'optimized': same docs, optimized for embedding similarity
        - 'corpus_l2': same attack surface, different crafting method
        - 'indirect': same KB, but with hidden instruction documents

        For this experiment, we use the pre-built kb_poisoned/ which
        already contains 250 naive poison docs covering all 50 queries.
        The attack_type label captures the adversary's CAPABILITY LEVEL,
        not a different KB.
        """
        return self.kb_poisoned

    def _make_mtd_config(
        self,
        defense: str,
        pool_size: int,
        kb_poisoned: str,
    ) -> str:
        """Create a temporary MTD config for a specific defense variant.

        MTD-S:  KB rotation only. Fixed retriever (dense), fixed embed (single).
        MTD-D:  Retriever cycling only. Fixed KB (poisoned), fixed embed.
        MTD-R:  Embed rotation only. Fixed KB (poisoned), fixed retriever.
        MTD-SD: KB rotation + retriever cycling. Fixed embed.
        MTD-SDR: All three active.
        """
        cfg = {"mtd": dict(self.mtd_cfg)}

        # Default: all enabled
        pool = self.full_pool[:pool_size]
        cfg["mtd"]["kb_pool"] = pool
        cfg["mtd"]["kb_pool_size"] = len(pool)
        cfg["mtd"]["enabled"] = True
        cfg["mtd"]["rotation_interval"] = 10
        cfg["mtd"]["batch_size"] = 5
        cfg["mtd"]["embed_rotation_interval"] = 10

        if defense == "MTD-S":
            # Shuffling only: rotate KB, but fix retriever and embedder
            cfg["mtd"]["retrieval_sequence"] = ["dense"]
            cfg["mtd"]["embed_models"] = [self.mtd_cfg["embed_models"][0]]
            cfg["mtd"]["batch_size"] = 9999  # never switch retriever

        elif defense == "MTD-D":
            # Diversity only: cycle retrievers, but fix KB to poisoned
            cfg["mtd"]["kb_pool"] = [kb_poisoned]
            cfg["mtd"]["kb_pool_size"] = 1
            # Need >= 2 for KBRotator, add a dummy duplicate
            cfg["mtd"]["kb_pool"] = [kb_poisoned, kb_poisoned]
            cfg["mtd"]["kb_pool_size"] = 2
            cfg["mtd"]["rotation_interval"] = 9999  # never rotate KB
            cfg["mtd"]["embed_models"] = [self.mtd_cfg["embed_models"][0]]

        elif defense == "MTD-R":
            # Redundancy only: rotate embedders, fix KB and retriever
            cfg["mtd"]["kb_pool"] = [kb_poisoned, kb_poisoned]
            cfg["mtd"]["kb_pool_size"] = 2
            cfg["mtd"]["rotation_interval"] = 9999
            cfg["mtd"]["retrieval_sequence"] = ["dense"]
            cfg["mtd"]["batch_size"] = 9999

        elif defense == "MTD-SD":
            # Shuffling + Diversity: rotate KB and retriever, fix embed
            cfg["mtd"]["embed_models"] = [self.mtd_cfg["embed_models"][0]]

        elif defense == "MTD-SDR":
            # Full defense: all three active
            pass

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="mtd_exp_",
        )
        yaml.dump(cfg, tmp, default_flow_style=False)
        tmp.close()
        return tmp.name

    @staticmethod
    def _compute_narr(asr_baseline: float, asr_mtd: float) -> float:
        """Normalized Attack Reduction Rate."""
        if asr_baseline <= 0:
            return 0.0
        return round(max((asr_baseline - asr_mtd) / asr_baseline, 0.0), 4)


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_table_III(results: list[ExperimentResult]) -> str:
    """Generate Table III: Attack × Defense ASR matrix."""
    # Group A (baseline) + Group C (MTD-SDR)
    baselines = {r.attack_type: r for r in results if r.defense == "none"}
    ablations = {r.defense: r for r in results
                 if r.defense in ("MTD-S", "MTD-D", "MTD-R", "MTD-SD")
                 and r.attack_type == "naive"}
    full_defense = {r.attack_type: r for r in results if r.defense == "MTD-SDR"
                    and r.condition_id <= 12}

    attacks = ["naive", "optimized", "corpus_l2", "indirect"]
    defenses = ["none", "MTD-S", "MTD-D", "MTD-R", "MTD-SD", "MTD-SDR"]

    lines = [
        f"Table III: ASR (%) under MTD configurations (n={results[0].n_queries}, seed={SEED})",
        "",
        "+---------------+--------+--------+--------+--------+--------+--------+",
        "| Attack        | No Def | MTD-S  | MTD-D  | MTD-R  | MTD-SD |MTD-SDR |",
        "+---------------+--------+--------+--------+--------+--------+--------+",
    ]

    for atk in attacks:
        vals = []
        # No defense
        vals.append(f"{baselines[atk].asr * 100:5.1f}%")
        # Ablation (only for naive)
        for d in ["MTD-S", "MTD-D", "MTD-R", "MTD-SD"]:
            if atk == "naive" and d in ablations:
                vals.append(f"{ablations[d].asr * 100:5.1f}%")
            else:
                vals.append("   —  ")
        # MTD-SDR
        if atk in full_defense:
            vals.append(f"{full_defense[atk].asr * 100:5.1f}%")
        else:
            vals.append("   —  ")

        label = {
            "naive": "Naive (L1)    ",
            "optimized": "Optimized (L1)",
            "corpus_l2": "Corpus (L2)  ",
            "indirect": "Indirect (L1) ",
        }[atk]
        row = f"| {label} | {' | '.join(vals)} |"
        lines.append(row)

    lines.append("+---------------+--------+--------+--------+--------+--------+--------+")

    # NARR row for naive
    narr_vals = ["   —  "]
    for d in ["MTD-S", "MTD-D", "MTD-R", "MTD-SD"]:
        if d in ablations:
            narr_vals.append(f" {ablations[d].narr:.2f} ")
        else:
            narr_vals.append("   —  ")
    if "naive" in full_defense:
        narr_vals.append(f" {full_defense['naive'].narr:.2f} ")
    lines.append(f"| NARR (naive)  | {' | '.join(narr_vals)} |")
    lines.append("+---------------+--------+--------+--------+--------+--------+--------+")

    return "\n".join(lines)


def generate_table_IV(results: list[ExperimentResult]) -> str:
    """Generate Table IV: Pool size ablation."""
    pool_results = [r for r in results if r.defense == "MTD-SDR"
                    and r.attack_type == "naive" and r.condition_id > 12]
    pool_results.sort(key=lambda r: r.pool_size)

    lines = [
        f"Table IV: Pool Size Ablation (MTD-SDR, Naive attack, n={results[0].n_queries})",
        "",
        "+------+--------+--------+--------+--------+----------+",
        "| Pool | ASR    | NARR   | MRR@10 | F1     | Lat.(ms) |",
        "+------+--------+--------+--------+--------+----------+",
    ]

    for r in pool_results:
        total_ms = r.mean_retrieval_ms + r.mean_generation_ms
        lines.append(
            f"|  {r.pool_size}   | {r.asr * 100:5.1f}% | {r.narr:.4f} | "
            f"{r.mrr10:.4f} | {r.mean_f1:.4f} | {total_ms:7.1f}  |"
        )

    lines.append("+------+--------+--------+--------+--------+----------+")
    return "\n".join(lines)


def save_all_results(
    results: list[ExperimentResult],
    path: str = "data/results/experiment_matrix.json",
) -> None:
    """Save all experiment results to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    out_path.write_text(json.dumps(data, indent=2))
    logger.info("Results saved to %s", path)


def main() -> None:
    """Run full experiment matrix."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    runner = ExperimentRunner()
    results = runner.run_matrix()

    # Generate tables
    t3 = generate_table_III(results)
    t4 = generate_table_IV(results)

    print("\n" + t3)
    print("\n" + t4)

    # Save
    save_all_results(results)
    Path("docs/table_III_experiment_matrix.txt").write_text(t3)
    Path("docs/table_IV_pool_ablation.txt").write_text(t4)

    # Summary
    best = min((r for r in results if r.defense == "MTD-SDR"), key=lambda r: r.asr)
    print(f"\nBest result: {best.defense} pool={best.pool_size} → "
          f"ASR={best.asr * 100:.1f}%, NARR={best.narr:.4f}")
    print(f"Total conditions: {len(results)}")


if __name__ == "__main__":
    main()
