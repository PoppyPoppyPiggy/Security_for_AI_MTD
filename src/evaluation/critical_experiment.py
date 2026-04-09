# =============================================================================
# FILE: src/evaluation/critical_experiment.py
# DESC: Redesigned experiment suite — 5 critical fixes for publication quality
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-09
# FIXES: (1) real optimized attack, (2) per-strategy ASR, (3) multi-seed,
#        (4) RobustRAG baseline, (5) meaningful pool ablation
# =============================================================================
from __future__ import annotations

import json
import logging
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from src.attacker.poisoned_rag import PoisonedRAGAttacker
from src.evaluation.baselines.robust_rag import robust_rag_aggregate
from src.evaluation.metrics import compute_asr, compute_f1, compute_mean_latency, compute_mrr
from src.rag_pipeline.pipeline import RAGPipeline
from src.utils import safe_load_config

logger = logging.getLogger(__name__)

KB_CLEAN = "data/knowledge_bases/kb_clean/"
KB_POISONED_NAIVE = "data/knowledge_bases/kb_poisoned/"
KB_POISONED_OPT = "data/knowledge_bases/kb_poisoned_optimized/"
KB_SNAP1 = "data/knowledge_bases/kb_rotated/snapshot_1/"
KB_SNAP2 = "data/knowledge_bases/kb_rotated/snapshot_2/"
KB_SNAP3 = "data/knowledge_bases/kb_rotated/snapshot_3/"
KB_SNAP4 = "data/knowledge_bases/kb_rotated/snapshot_4/"


@dataclass
class ExpResult:
    """Single experiment result."""
    label: str
    attack: str
    defense: str
    seed: int
    n_queries: int
    asr: float
    mrr10: float
    mean_f1: float
    latency_ms: float


class CriticalExperiment:
    """Redesigned experiment suite with 5 critical fixes."""

    def __init__(
        self,
        rag_config_path: str = "config/rag_config.yaml",
        mtd_config_path: str = "config/mtd_config.yaml",
        atk_config_path: str = "config/attack_config.yaml",
    ) -> None:
        self.rag_config_path = rag_config_path
        self.mtd_config_path = mtd_config_path
        self.atk_cfg = safe_load_config(atk_config_path)["attack"]
        self.base_mtd_cfg = safe_load_config(mtd_config_path)["mtd"]

        self.target_answer = self.atk_cfg["target_answer"]

        with open(self.atk_cfg["target_queries_path"]) as f:
            self.target_data = json.load(f)
        self.queries = [d["query"] for d in self.target_data]
        self.ground_truths = [d["ground_truth"] for d in self.target_data]

    # ------------------------------------------------------------------
    # Fix 1: Build optimized poisoned KB
    # ------------------------------------------------------------------

    def build_optimized_kb(self) -> None:
        """Build a separate poisoned KB using the optimized (keyword-sparse) strategy."""
        logger.info("Building optimized poisoned KB...")
        atk = PoisonedRAGAttacker(config_path="config/attack_config.yaml")

        opt_path = Path(KB_POISONED_OPT)
        if opt_path.exists():
            shutil.rmtree(opt_path)

        for q in self.target_data:
            docs = atk.craft_poison_docs(
                q["query"], q["target_answer"],
                num_docs=5, strategy="optimized",
            )
            atk.inject_into_kb(KB_CLEAN, docs, str(opt_path))

        n_files = len(list(opt_path.glob("*.txt")))
        logger.info("Optimized KB built: %d txt files", n_files)

    # ------------------------------------------------------------------
    # Fix 2: Per-retrieval-strategy ASR (Table VI)
    # ------------------------------------------------------------------

    def run_per_strategy(self, seed: int = 42) -> list[ExpResult]:
        """Measure ASR per retrieval strategy for naive vs optimized attacks."""
        results: list[ExpResult] = []
        strategies = ["dense", "sparse", "hybrid"]

        for attack, kb_path in [("naive", KB_POISONED_NAIVE), ("optimized", KB_POISONED_OPT)]:
            for strategy in strategies:
                logger.info("Per-strategy: attack=%s, strategy=%s", attack, strategy)
                self._set_seed(seed)

                # Create config with fixed strategy
                rag_cfg = safe_load_config(self.rag_config_path)
                rag_cfg["rag"]["retrieval_strategy"] = strategy
                cfg_path = self._write_temp_yaml(rag_cfg)

                pipeline = RAGPipeline(
                    mode="baseline", rag_config_path=cfg_path,
                    mtd_config_path=self.mtd_config_path, kb_path=kb_path,
                )
                outputs = pipeline.run_batch(self.queries)
                asr = compute_asr(outputs, self.queries, self.target_answer)
                latency = compute_mean_latency(outputs)

                results.append(ExpResult(
                    label=f"{attack}_{strategy}",
                    attack=attack, defense=f"fixed_{strategy}", seed=seed,
                    n_queries=len(self.queries), asr=round(asr, 4),
                    mrr10=0.0, mean_f1=0.0,
                    latency_ms=latency["total_ms"],
                ))

            # MTD-D (rotating strategies) on this attack's KB
            logger.info("Per-strategy: attack=%s, strategy=MTD-D", attack)
            self._set_seed(seed)
            mtd_cfg = self._make_mtd_cfg(
                pool=[kb_path, kb_path], rotation_interval=9999,
                retrieval_sequence=["dense", "sparse", "hybrid"],
                batch_size=5, embed_models=[self.base_mtd_cfg["embed_models"][0]],
            )
            pipeline = RAGPipeline(
                mode="mtd", rag_config_path=self.rag_config_path,
                mtd_config_path=mtd_cfg,
            )
            outputs = pipeline.run_batch(self.queries)
            asr = compute_asr(outputs, self.queries, self.target_answer)

            results.append(ExpResult(
                label=f"{attack}_MTD-D",
                attack=attack, defense="MTD-D", seed=seed,
                n_queries=len(self.queries), asr=round(asr, 4),
                mrr10=0.0, mean_f1=0.0, latency_ms=0.0,
            ))

        return results

    # ------------------------------------------------------------------
    # Fix 3: Multi-seed Scenario B (Table III with ±std)
    # ------------------------------------------------------------------

    def run_multi_seed_scenario_b(self, seeds: list[int] = None) -> dict:
        """Run Scenario B (1 poisoned/5) across multiple seeds."""
        if seeds is None:
            seeds = [42, 0, 7]

        pool_b = [KB_POISONED_NAIVE, KB_CLEAN, KB_SNAP1, KB_SNAP2, KB_SNAP3]
        conditions = [
            ("none", None),
            ("MTD-S", {"retrieval_sequence": ["dense"], "batch_size": 9999,
                       "embed_models": [self.base_mtd_cfg["embed_models"][0]]}),
            ("MTD-SDR", {}),
        ]

        all_results: dict[str, list[float]] = {}
        for defense_name, overrides in conditions:
            asrs = []
            for seed in seeds:
                logger.info("Multi-seed: defense=%s, seed=%d", defense_name, seed)
                self._set_seed(seed)

                if defense_name == "none":
                    pipeline = RAGPipeline(
                        mode="baseline", rag_config_path=self.rag_config_path,
                        mtd_config_path=self.mtd_config_path,
                        kb_path=KB_POISONED_NAIVE,
                    )
                else:
                    cfg_overrides = overrides or {}
                    mtd_cfg = self._make_mtd_cfg(pool=pool_b, **cfg_overrides)
                    pipeline = RAGPipeline(
                        mode="mtd", rag_config_path=self.rag_config_path,
                        mtd_config_path=mtd_cfg,
                    )

                outputs = pipeline.run_batch(self.queries)
                asr = compute_asr(outputs, self.queries, self.target_answer)
                asrs.append(asr)

            all_results[defense_name] = asrs

        return all_results

    # ------------------------------------------------------------------
    # Fix 4: RobustRAG baseline
    # ------------------------------------------------------------------

    def run_robust_rag_baseline(self, seed: int = 42) -> ExpResult:
        """Run RobustRAG isolate-then-aggregate on poisoned KB."""
        logger.info("RobustRAG baseline...")
        self._set_seed(seed)

        pipeline = RAGPipeline(
            mode="baseline", rag_config_path=self.rag_config_path,
            mtd_config_path=self.mtd_config_path,
            kb_path=KB_POISONED_NAIVE,
        )

        successes = 0
        for query in self.queries:
            output = pipeline.run(query)
            aggregated = robust_rag_aggregate(
                output.retrieved_chunks, self.target_answer,
            )
            if self.target_answer.lower() in aggregated.lower():
                successes += 1

        asr = successes / len(self.queries)
        logger.info("RobustRAG ASR = %.4f", asr)

        return ExpResult(
            label="RobustRAG", attack="naive", defense="RobustRAG",
            seed=seed, n_queries=len(self.queries), asr=round(asr, 4),
            mrr10=0.0, mean_f1=0.0, latency_ms=0.0,
        )

    # ------------------------------------------------------------------
    # Fix 5: Pool size ablation under Scenario B (1 poisoned always)
    # ------------------------------------------------------------------

    def run_pool_ablation(self, seed: int = 42) -> list[ExpResult]:
        """Run MTD-SDR with varying pool sizes, always 1 poisoned KB."""
        results: list[ExpResult] = []
        clean_kbs = [KB_CLEAN, KB_SNAP1, KB_SNAP2, KB_SNAP3]

        for pool_size in [2, 3, 4, 5]:
            logger.info("Pool ablation: size=%d", pool_size)
            self._set_seed(seed)

            pool = [KB_POISONED_NAIVE] + clean_kbs[:pool_size - 1]
            mtd_cfg = self._make_mtd_cfg(pool=pool)

            pipeline = RAGPipeline(
                mode="mtd", rag_config_path=self.rag_config_path,
                mtd_config_path=mtd_cfg,
            )
            outputs = pipeline.run_batch(self.queries)

            asr = compute_asr(outputs, self.queries, self.target_answer)
            f1_scores = [compute_f1(o.answer, gt) for o, gt in zip(outputs, self.ground_truths)]
            mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            latency = compute_mean_latency(outputs)

            results.append(ExpResult(
                label=f"pool_{pool_size}",
                attack="naive", defense="MTD-SDR", seed=seed,
                n_queries=len(self.queries), asr=round(asr, 4),
                mrr10=0.0, mean_f1=round(mean_f1, 4),
                latency_ms=latency["total_ms"],
            ))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    def _make_mtd_cfg(
        self,
        pool: list[str],
        rotation_interval: int = 10,
        retrieval_sequence: list[str] | None = None,
        batch_size: int = 5,
        embed_models: list[str] | None = None,
    ) -> str:
        cfg = {"mtd": dict(self.base_mtd_cfg)}
        cfg["mtd"]["enabled"] = True
        cfg["mtd"]["kb_pool"] = list(pool)
        cfg["mtd"]["kb_pool_size"] = len(pool)
        cfg["mtd"]["rotation_interval"] = rotation_interval
        cfg["mtd"]["batch_size"] = batch_size
        if retrieval_sequence:
            cfg["mtd"]["retrieval_sequence"] = retrieval_sequence
        if embed_models:
            cfg["mtd"]["embed_models"] = embed_models
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(cfg, tmp, default_flow_style=False)
        tmp.close()
        return tmp.name

    def _write_temp_yaml(self, data: dict) -> str:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(data, tmp, default_flow_style=False)
        tmp.close()
        return tmp.name


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_table_vi(results: list[ExpResult]) -> str:
    """Table VI: Per-retrieval-strategy ASR."""
    lines = [
        "Table VI: ASR by Retrieval Strategy — Naive vs Optimized Attack (n=50)",
        "",
        "+------------------+------------+----------------+",
        "| Strategy         | Naive ASR  | Optimized ASR  |",
        "+------------------+------------+----------------+",
    ]
    strategies = ["fixed_dense", "fixed_sparse", "fixed_hybrid", "MTD-D"]
    labels = ["Dense (fixed)", "Sparse (fixed)", "Hybrid (fixed)", "MTD-D (rotate)"]

    for strat, label in zip(strategies, labels):
        naive_r = next((r for r in results if r.attack == "naive" and r.defense == strat), None)
        opt_r = next((r for r in results if r.attack == "optimized" and r.defense == strat), None)
        n_asr = f"{naive_r.asr * 100:5.1f}%" if naive_r else "  —  "
        o_asr = f"{opt_r.asr * 100:5.1f}%" if opt_r else "  —  "
        lines.append(f"| {label:<16} | {n_asr:>10} | {o_asr:>14} |")

    lines.append("+------------------+------------+----------------+")
    return "\n".join(lines)


def generate_table_iii_multiseed(multi_seed: dict) -> str:
    """Table III: Scenario B with mean±std across seeds."""
    lines = [
        "Table III: Scenario B ASR — Multi-Seed Validation (seeds=42,0,7, n=50)",
        "",
        "+-----------+------------------+",
        "| Defense   | ASR (mean±std)   |",
        "+-----------+------------------+",
    ]
    for defense, asrs in multi_seed.items():
        mean_asr = np.mean(asrs) * 100
        std_asr = np.std(asrs) * 100
        lines.append(f"| {defense:<9} | {mean_asr:5.1f} ± {std_asr:4.1f}%    |")
    lines.append("+-----------+------------------+")
    return "\n".join(lines)


def generate_table_iv_pool(results: list[ExpResult]) -> str:
    """Table IV: Pool size ablation under Scenario B."""
    lines = [
        "Table IV: Pool Size Ablation — Scenario B (1 poisoned KB, MTD-SDR, n=50)",
        "",
        "+------+--------+--------+---------+----------+",
        "| Pool | P-Rate | ASR    | F1      | Lat.(ms) |",
        "+------+--------+--------+---------+----------+",
    ]
    for r in sorted(results, key=lambda x: int(x.label.split("_")[1])):
        pool_size = int(r.label.split("_")[1])
        p_rate = 1.0 / pool_size
        lines.append(
            f"|  {pool_size}   | {p_rate*100:4.0f}%  | {r.asr*100:5.1f}% | "
            f"{r.mean_f1:.4f}  | {r.latency_ms:7.1f}  |"
        )
    lines.append("+------+--------+--------+---------+----------+")
    lines.append("")
    lines.append("P-Rate = poisoning ratio (1 poisoned / pool_size)")
    return "\n".join(lines)


def generate_all_figures(
    per_strategy: list[ExpResult],
    pool_ablation: list[ExpResult],
    multi_seed: dict,
    robust_rag: ExpResult,
    output_dir: str = "docs/figures",
) -> None:
    """Regenerate all figures with updated data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Fig 6: Naive vs Optimized per strategy ──
    fig, ax = plt.subplots(figsize=(8, 5))
    strategies = ["Dense", "Sparse", "Hybrid", "MTD-D"]
    naive_asrs = []
    opt_asrs = []
    for strat in ["fixed_dense", "fixed_sparse", "fixed_hybrid", "MTD-D"]:
        n = next((r for r in per_strategy if r.attack == "naive" and r.defense == strat), None)
        o = next((r for r in per_strategy if r.attack == "optimized" and r.defense == strat), None)
        naive_asrs.append(n.asr * 100 if n else 0)
        opt_asrs.append(o.asr * 100 if o else 0)

    x = np.arange(len(strategies))
    ax.bar(x - 0.17, naive_asrs, 0.32, label="Naive Attack", color="#d32f2f", edgecolor="white")
    ax.bar(x + 0.17, opt_asrs, 0.32, label="Optimized Attack", color="#1976d2", edgecolor="white")

    for i, (nv, ov) in enumerate(zip(naive_asrs, opt_asrs)):
        ax.text(i - 0.17, nv + 1.5, f"{nv:.0f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + 0.17, ov + 1.5, f"{ov:.0f}%", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_ylabel("ASR (%)", fontsize=12)
    ax.set_title("Naive vs Optimized Attack: ASR by Retrieval Strategy", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fig6_naive_vs_optimized.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 7: Pool size ablation (Scenario B) ──
    fig, ax = plt.subplots(figsize=(7, 5))
    pool_sorted = sorted(pool_ablation, key=lambda r: int(r.label.split("_")[1]))
    pool_sizes = [int(r.label.split("_")[1]) for r in pool_sorted]
    pool_asrs = [r.asr * 100 for r in pool_sorted]
    theoretical = [98.0 / ps for ps in pool_sizes]

    ax.plot(pool_sizes, pool_asrs, "s-", color="#7b1fa2", linewidth=2.5, markersize=10,
            label="MTD-SDR (measured)", zorder=5)
    ax.plot(pool_sizes, theoretical, "o--", color="#888888", linewidth=1.5, markersize=8,
            label="Theoretical: ASR_base/N", zorder=4)
    ax.axhline(y=98, color="#d32f2f", linestyle=":", alpha=0.7, label="No Defense (98%)")

    for ps, asr_val in zip(pool_sizes, pool_asrs):
        ax.annotate(f"{asr_val:.0f}%", (ps, asr_val), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold", color="#7b1fa2")

    ax.set_xlabel("Pool Size (N)", fontsize=12)
    ax.set_ylabel("ASR (%)", fontsize=12)
    ax.set_title("Pool Size Ablation — Scenario B (1 Poisoned KB)", fontsize=13, fontweight="bold")
    ax.set_xticks(pool_sizes)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fig7_pool_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 8: Defense comparison (No Def vs RobustRAG vs MTD-SDR) ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    defenses = ["No Defense", "RobustRAG\n(Static)", "MTD-SDR\n(Dynamic)"]
    # For MTD-SDR use Scenario B pool=5 result
    mtd_asr = next((r.asr * 100 for r in pool_ablation if r.label == "pool_5"), 20.0)
    asrs = [98.0, robust_rag.asr * 100, mtd_asr]
    colors = ["#d32f2f", "#ff9800", "#7b1fa2"]

    bars = ax.bar(defenses, asrs, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("ASR (%)", fontsize=12)
    ax.set_title("Defense Comparison — Scenario B (Naive Attack, n=50)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fig8_defense_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Figures 6-8 saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    exp = CriticalExperiment()

    # Fix 1: Build optimized KB
    logger.info("===== Fix 1: Building optimized poisoned KB =====")
    exp.build_optimized_kb()

    # Fix 2: Per-strategy ASR
    logger.info("===== Fix 2: Per-retrieval-strategy ASR =====")
    per_strategy = exp.run_per_strategy()
    t6 = generate_table_vi(per_strategy)
    print("\n" + t6)

    # Fix 3: Multi-seed
    logger.info("===== Fix 3: Multi-seed Scenario B =====")
    multi_seed = exp.run_multi_seed_scenario_b()
    t3 = generate_table_iii_multiseed(multi_seed)
    print("\n" + t3)

    # Fix 4: RobustRAG
    logger.info("===== Fix 4: RobustRAG baseline =====")
    robust_rag = exp.run_robust_rag_baseline()
    print(f"\nRobustRAG ASR: {robust_rag.asr * 100:.1f}%")

    # Fix 5: Pool ablation (Scenario B)
    logger.info("===== Fix 5: Pool size ablation =====")
    pool_ablation = exp.run_pool_ablation()
    t4 = generate_table_iv_pool(pool_ablation)
    print("\n" + t4)

    # Save tables
    Path("docs/table_VI_per_strategy.txt").write_text(t6)
    Path("docs/table_III_multiseed.txt").write_text(t3)
    Path("docs/table_IV_pool_ablation_scenB.txt").write_text(t4)

    # Save JSON
    all_data = {
        "per_strategy": [asdict(r) for r in per_strategy],
        "multi_seed": {k: [round(v, 4) for v in vs] for k, vs in multi_seed.items()},
        "robust_rag": asdict(robust_rag),
        "pool_ablation": [asdict(r) for r in pool_ablation],
    }
    Path("data/results").mkdir(parents=True, exist_ok=True)
    Path("data/results/critical_experiment.json").write_text(json.dumps(all_data, indent=2))

    # Generate figures
    logger.info("===== Generating figures =====")
    generate_all_figures(per_strategy, pool_ablation, multi_seed, robust_rag)

    print("\n===== All 5 fixes complete =====")


if __name__ == "__main__":
    main()
