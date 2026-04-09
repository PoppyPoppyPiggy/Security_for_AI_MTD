# =============================================================================
# FILE: src/evaluation/realistic_experiment.py
# DESC: Realistic experiment runner with mixed KB pools (Scenario A/B/C)
#       + visualization of all results
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-09
# DEPS: src/evaluation/experiment_runner.py, matplotlib
# =============================================================================
from __future__ import annotations

import json
import logging
import random
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

# KB paths
KB_CLEAN = "data/knowledge_bases/kb_clean/"
KB_POISONED = "data/knowledge_bases/kb_poisoned/"
KB_SNAP1 = "data/knowledge_bases/kb_rotated/snapshot_1/"
KB_SNAP2 = "data/knowledge_bases/kb_rotated/snapshot_2/"
KB_SNAP3 = "data/knowledge_bases/kb_rotated/snapshot_3/"
KB_SNAP4 = "data/knowledge_bases/kb_rotated/snapshot_4/"

SCENARIOS = {
    "A": {
        "name": "Clean Pool (upper bound)",
        "pool": [KB_CLEAN, KB_SNAP1, KB_SNAP2, KB_SNAP3, KB_SNAP4],
        "n_poisoned": 0,
    },
    "B": {
        "name": "Realistic (1 poisoned / 5)",
        "pool": [KB_POISONED, KB_CLEAN, KB_SNAP1, KB_SNAP2, KB_SNAP3],
        "n_poisoned": 1,
    },
    "C": {
        "name": "Severe (2 poisoned / 5)",
        "pool": [KB_POISONED, KB_CLEAN, KB_POISONED, KB_SNAP1, KB_SNAP2],
        "n_poisoned": 2,
    },
}

DEFENSES = ["none", "MTD-S", "MTD-D", "MTD-SDR"]


@dataclass
class ScenarioResult:
    """Result of a single scenario × defense experiment."""

    scenario: str
    scenario_name: str
    defense: str
    n_queries: int
    n_poisoned_in_pool: int
    pool_size: int
    asr: float
    narr: float
    mrr10: float
    mean_f1: float
    mean_retrieval_ms: float
    mean_generation_ms: float


class RealisticExperimentRunner:
    """Runs experiments with mixed KB pools (clean + poisoned).

    Three scenarios:
      A: Pool = [clean × 5]            — upper bound, trivial defense
      B: Pool = [poisoned × 1, clean × 4] — realistic deployment
      C: Pool = [poisoned × 2, clean × 3] — severe compromise

    Four defenses per scenario: none, MTD-S, MTD-D, MTD-SDR
    Total: 3 × 4 = 12 experiments
    """

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
        self.kb_poisoned = self.atk_cfg["kb_poisoned_path"]

        with open(self.atk_cfg["target_queries_path"]) as f:
            self.target_data = json.load(f)
        self.queries = [d["query"] for d in self.target_data]
        self.ground_truths = [d["ground_truth"] for d in self.target_data]

        logger.info("RealisticExperimentRunner: %d queries", len(self.queries))

    def run_all_scenarios(self) -> list[ScenarioResult]:
        """Run all 12 experiments (3 scenarios × 4 defenses)."""
        results: list[ScenarioResult] = []
        baseline_asrs: dict[str, float] = {}

        total = len(SCENARIOS) * len(DEFENSES)
        idx = 0

        for scenario_id, scenario_cfg in SCENARIOS.items():
            # First run baseline for this scenario
            for defense in DEFENSES:
                idx += 1
                logger.info("[%d/%d] Scenario %s (%s) × %s",
                            idx, total, scenario_id, scenario_cfg["name"], defense)

                r = self._run_single(scenario_id, scenario_cfg, defense)
                results.append(r)

                if defense == "none":
                    baseline_asrs[scenario_id] = r.asr

            # Compute NARR for this scenario's MTD results
            for r in results:
                if r.scenario == scenario_id and r.defense != "none":
                    base_asr = baseline_asrs.get(scenario_id, 0.0)
                    if base_asr > 0:
                        r.narr = round(max((base_asr - r.asr) / base_asr, 0.0), 4)

        return results

    def _run_single(
        self,
        scenario_id: str,
        scenario_cfg: dict,
        defense: str,
    ) -> ScenarioResult:
        """Run a single experiment."""
        random.seed(SEED)
        np.random.seed(SEED)

        pool = scenario_cfg["pool"]

        if defense == "none":
            pipeline = RAGPipeline(
                mode="baseline",
                rag_config_path=self.rag_config_path,
                mtd_config_path=self.mtd_config_path,
                kb_path=self.kb_poisoned,
            )
        else:
            mtd_cfg_path = self._make_config(defense, pool)
            pipeline = RAGPipeline(
                mode="mtd",
                rag_config_path=self.rag_config_path,
                mtd_config_path=mtd_cfg_path,
            )

        outputs = pipeline.run_batch(self.queries)

        asr = compute_asr(outputs, self.queries, self.target_answer)
        relevant_ids = [f"seed_docs_{i}" for i in range(len(self.queries))]
        mrr = compute_mrr(outputs, relevant_ids)
        f1_scores = [compute_f1(o.answer, gt) for o, gt in zip(outputs, self.ground_truths)]
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        latency = compute_mean_latency(outputs)

        return ScenarioResult(
            scenario=scenario_id,
            scenario_name=scenario_cfg["name"],
            defense=defense,
            n_queries=len(self.queries),
            n_poisoned_in_pool=scenario_cfg["n_poisoned"],
            pool_size=len(pool),
            asr=round(asr, 4),
            narr=0.0,
            mrr10=round(mrr, 4),
            mean_f1=round(mean_f1, 4),
            mean_retrieval_ms=latency["retrieval_ms"],
            mean_generation_ms=latency["generation_ms"],
        )

    def _make_config(self, defense: str, pool: list[str]) -> str:
        """Create temp MTD config for given defense and pool."""
        cfg = {"mtd": dict(self.base_mtd_cfg)}
        cfg["mtd"]["enabled"] = True
        cfg["mtd"]["kb_pool"] = list(pool)
        cfg["mtd"]["kb_pool_size"] = len(pool)
        cfg["mtd"]["rotation_interval"] = 10
        cfg["mtd"]["batch_size"] = 5
        cfg["mtd"]["embed_rotation_interval"] = 10

        if defense == "MTD-S":
            cfg["mtd"]["retrieval_sequence"] = ["dense"]
            cfg["mtd"]["embed_models"] = [self.base_mtd_cfg["embed_models"][0]]
            cfg["mtd"]["batch_size"] = 9999

        elif defense == "MTD-D":
            # Fix KB to poisoned (always rotate within same KB)
            cfg["mtd"]["kb_pool"] = [self.kb_poisoned, self.kb_poisoned]
            cfg["mtd"]["kb_pool_size"] = 2
            cfg["mtd"]["rotation_interval"] = 9999
            cfg["mtd"]["embed_models"] = [self.base_mtd_cfg["embed_models"][0]]

        elif defense == "MTD-SDR":
            pass  # all active with the mixed pool

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="mtd_real_",
        )
        yaml.dump(cfg, tmp, default_flow_style=False)
        tmp.close()
        return tmp.name


# ---------------------------------------------------------------------------
# Table + Visualization
# ---------------------------------------------------------------------------

def generate_scenario_table(results: list[ScenarioResult]) -> str:
    """Generate the combined scenario table."""
    lines = [
        "Table V: ASR (%) under Realistic KB Pool Scenarios (n=50, seed=42)",
        "",
        "+-----------+--------+--------+--------+--------+",
        "| Scenario  | No Def | MTD-S  | MTD-D  |MTD-SDR |",
        "+-----------+--------+--------+--------+--------+",
    ]

    for sid in ["A", "B", "C"]:
        scenario_results = {r.defense: r for r in results if r.scenario == sid}
        name = SCENARIOS[sid]["name"]
        n_poison = SCENARIOS[sid]["n_poisoned"]

        vals = []
        for d in ["none", "MTD-S", "MTD-D", "MTD-SDR"]:
            if d in scenario_results:
                vals.append(f"{scenario_results[d].asr * 100:5.1f}%")
            else:
                vals.append("   —  ")

        label = f"{sid}(p={n_poison})"
        lines.append(f"| {label:<9} | {' | '.join(vals)} |")

    lines.append("+-----------+--------+--------+--------+--------+")
    lines.append("")
    lines.append("A = Clean pool (0 poisoned), B = Realistic (1 poisoned/5)")
    lines.append("C = Severe (2 poisoned/5), p = # poisoned KBs in pool")

    # Add NARR rows
    lines.append("")
    lines.append("NARR (Normalized Attack Reduction Rate):")
    for sid in ["A", "B", "C"]:
        scenario_results = {r.defense: r for r in results if r.scenario == sid}
        for d in ["MTD-S", "MTD-D", "MTD-SDR"]:
            if d in scenario_results:
                r = scenario_results[d]
                lines.append(f"  Scenario {sid} × {d:<7}: NARR = {r.narr:.4f}")

    return "\n".join(lines)


def generate_visualizations(results: list[ScenarioResult], output_dir: str = "docs/figures") -> list[str]:
    """Generate all paper figures as PNG files.

    Returns list of saved file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # Color scheme
    colors = {
        "none": "#d32f2f",      # red
        "MTD-S": "#1976d2",     # blue
        "MTD-D": "#388e3c",     # green
        "MTD-SDR": "#7b1fa2",   # purple
    }

    # ── Figure 1: ASR Bar Chart (Scenario × Defense) ──
    fig, ax = plt.subplots(figsize=(10, 5))

    scenarios = ["A", "B", "C"]
    scenario_labels = [
        "Scenario A\n(Clean Pool)",
        "Scenario B\n(1 Poisoned/5)",
        "Scenario C\n(2 Poisoned/5)",
    ]
    defenses = ["none", "MTD-S", "MTD-D", "MTD-SDR"]
    defense_labels = ["No Defense", "MTD-S (Shuffling)", "MTD-D (Diversity)", "MTD-SDR (Full)"]

    x = np.arange(len(scenarios))
    bar_width = 0.18

    for i, (d, dlabel) in enumerate(zip(defenses, defense_labels)):
        asr_vals = []
        for sid in scenarios:
            r = next((r for r in results if r.scenario == sid and r.defense == d), None)
            asr_vals.append(r.asr * 100 if r else 0)
        offset = (i - 1.5) * bar_width
        bars = ax.bar(x + offset, asr_vals, bar_width, label=dlabel, color=colors[d], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, asr_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("KB Pool Composition", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("ASR under Different MTD Configurations and KB Pool Scenarios", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = f"{output_dir}/fig1_asr_scenarios.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    logger.info("Saved %s", path)

    # ── Figure 2: NARR Comparison (Scenario B & C) ──
    fig, ax = plt.subplots(figsize=(8, 5))

    mtd_defenses = ["MTD-S", "MTD-D", "MTD-SDR"]
    mtd_labels = ["MTD-S\n(Shuffling)", "MTD-D\n(Diversity)", "MTD-SDR\n(Full)"]
    x = np.arange(len(mtd_defenses))

    for i, sid in enumerate(["B", "C"]):
        narr_vals = []
        for d in mtd_defenses:
            r = next((r for r in results if r.scenario == sid and r.defense == d), None)
            narr_vals.append(r.narr if r else 0)
        label = f"Scenario {sid} ({SCENARIOS[sid]['n_poisoned']} poisoned)"
        color = "#1976d2" if sid == "B" else "#d32f2f"
        offset = (i - 0.5) * 0.3
        ax.bar(x + offset, narr_vals, 0.28, label=label, color=color, alpha=0.85, edgecolor="white")

    ax.set_xlabel("MTD Defense Configuration", fontsize=12)
    ax.set_ylabel("NARR (Normalized Attack Reduction Rate)", fontsize=12)
    ax.set_title("Defense Effectiveness: NARR by Scenario", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(mtd_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect defense (NARR=1.0)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = f"{output_dir}/fig2_narr_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    logger.info("Saved %s", path)

    # ── Figure 3: MTD-SDR ASR vs Pool Poisoning Rate ──
    fig, ax = plt.subplots(figsize=(7, 5))

    poison_rates = []
    asr_none = []
    asr_sdr = []
    for sid in ["A", "B", "C"]:
        n_p = SCENARIOS[sid]["n_poisoned"]
        rate = n_p / len(SCENARIOS[sid]["pool"])
        r_none = next((r for r in results if r.scenario == sid and r.defense == "none"), None)
        r_sdr = next((r for r in results if r.scenario == sid and r.defense == "MTD-SDR"), None)
        poison_rates.append(rate * 100)
        asr_none.append(r_none.asr * 100 if r_none else 0)
        asr_sdr.append(r_sdr.asr * 100 if r_sdr else 0)

    ax.plot(poison_rates, asr_none, "o-", color="#d32f2f", linewidth=2, markersize=10, label="No Defense", zorder=5)
    ax.plot(poison_rates, asr_sdr, "s-", color="#7b1fa2", linewidth=2, markersize=10, label="MTD-SDR (Full)", zorder=5)

    # Fill between to show defense gap
    ax.fill_between(poison_rates, asr_sdr, asr_none, alpha=0.15, color="#7b1fa2")

    # Annotate defense gap
    for i, (pr, a_n, a_s) in enumerate(zip(poison_rates, asr_none, asr_sdr)):
        gap = a_n - a_s
        if gap > 0:
            mid_y = (a_n + a_s) / 2
            ax.annotate(f"ASR-{gap:.0f}pp", xy=(pr, mid_y),
                        fontsize=9, ha="center", color="#7b1fa2", fontweight="bold")

    ax.set_xlabel("Pool Poisoning Rate (%)", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("Defense Effectiveness vs. Pool Contamination Level", fontsize=13, fontweight="bold")
    ax.set_xlim(-5, 45)
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = f"{output_dir}/fig3_asr_vs_poisoning.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    logger.info("Saved %s", path)

    # ── Figure 4: SDR Triad Contribution (Scenario B) ──
    fig, ax = plt.subplots(figsize=(7, 5))

    b_results = {r.defense: r for r in results if r.scenario == "B"}
    defenses_ordered = ["none", "MTD-S", "MTD-D", "MTD-SDR"]
    labels = ["No Defense", "Shuffling\n(KB Rotation)", "Diversity\n(Retriever Switch)", "SDR Full\n(All Combined)"]
    asr_vals = [b_results[d].asr * 100 if d in b_results else 0 for d in defenses_ordered]
    bar_colors = [colors[d] for d in defenses_ordered]

    bars = ax.barh(range(len(labels)), asr_vals, color=bar_colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, asr_vals):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("Scenario B: SDR Triad Component Contribution", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(asr_vals) * 1.2 + 5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = f"{output_dir}/fig4_sdr_contribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    logger.info("Saved %s", path)

    # ── Figure 5: ATLAS TTP → MTD Mapping Heatmap ──
    fig, ax = plt.subplots(figsize=(8, 4))

    ttps = ["AML.T0054\n(KB Injection)", "AML.T0020\n(Corpus Poison)", "AML.T0056\n(Recon)",
            "AML.T0057\n(DB Prompting)", "AML.T0051\n(Prompt Inject)"]
    strategies = ["Shuffling (S)", "Diversity (D)", "Redundancy (R)"]

    # Effectiveness matrix: 0=none, 1=medium, 2=high
    matrix = np.array([
        [2, 0, 0],   # T0054 → S=High
        [0, 2, 0],   # T0020 → D=High
        [1, 0, 0],   # T0056 → S=Medium
        [0, 1, 0],   # T0057 → D=Medium
        [0, 0, 1],   # T0051 → R=Medium
    ])

    cmap = plt.cm.YlOrRd
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=2)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_yticks(range(len(ttps)))
    ax.set_yticklabels(ttps, fontsize=9)

    # Add text annotations
    labels_map = {0: "—", 1: "Medium", 2: "High"}
    for i in range(len(ttps)):
        for j in range(len(strategies)):
            val = matrix[i, j]
            color = "white" if val >= 2 else "black"
            ax.text(j, i, labels_map[val], ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_title("MITRE ATLAS TTP → MTD Strategy Mapping (SDR Triad)", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.8)
    cbar.ax.set_yticklabels(["None", "Medium", "High"])

    path = f"{output_dir}/fig5_atlas_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    logger.info("Saved %s", path)

    return saved


def main() -> None:
    """Run realistic experiments + generate visualizations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    runner = RealisticExperimentRunner()
    results = runner.run_all_scenarios()

    # Generate table
    table = generate_scenario_table(results)
    print("\n" + table)

    # Save table
    Path("docs/table_V_realistic_scenarios.txt").write_text(table)

    # Save JSON
    Path("data/results").mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    Path("data/results/realistic_experiment.json").write_text(json.dumps(data, indent=2))

    # Generate visualizations
    figures = generate_visualizations(results)
    print(f"\nFigures saved: {len(figures)}")
    for f in figures:
        print(f"  {f}")

    # Summary
    best_b = next((r for r in results if r.scenario == "B" and r.defense == "MTD-SDR"), None)
    best_c = next((r for r in results if r.scenario == "C" and r.defense == "MTD-SDR"), None)
    if best_b:
        print(f"\nScenario B (realistic): MTD-SDR ASR={best_b.asr*100:.1f}%, NARR={best_b.narr:.4f}")
    if best_c:
        print(f"Scenario C (severe):    MTD-SDR ASR={best_c.asr*100:.1f}%, NARR={best_c.narr:.4f}")


if __name__ == "__main__":
    main()
