# =============================================================================
# FILE: src/evaluation/evaluator.py
# DESC: Experiment runner — baseline vs MTD comparison
# AUTHOR: DS Lab / Minseong
# CREATED: 2026-04-08
# DEPS: config/rag_config.yaml, config/attack_config.yaml,
#       src/rag_pipeline/pipeline.py, src/evaluation/metrics.py
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.evaluation.metrics import summarize
from src.rag_pipeline.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# EXP: reproducibility
SEED: int = 42


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_target_queries(path: str) -> list[dict]:
    """Load target queries from JSON file.

    Args:
        path: Path to target_queries.json.

    Returns:
        List of dicts with keys: query, target_answer, ground_truth.
    """
    with open(path) as f:
        return json.load(f)


def run_evaluation(
    mode: str,
    attack_config_path: str = "config/attack_config.yaml",
    rag_config_path: str = "config/rag_config.yaml",
    mtd_config_path: str = "config/mtd_config.yaml",
    seed: int = SEED,
    kb_path: str | None = None,
) -> dict:
    """Run a full evaluation experiment.

    Args:
        mode: Pipeline mode — 'baseline' or 'mtd'.
        attack_config_path: Path to attack config.
        rag_config_path: Path to RAG config.
        mtd_config_path: Path to MTD config.
        seed: Random seed for reproducibility.
        kb_path: Override KB path (defaults to attack config's kb_path for baseline).

    Returns:
        Full results dict with metrics and configuration.
    """
    set_seed(seed)

    with open(attack_config_path) as f:
        atk_cfg = yaml.safe_load(f)["attack"]

    target_data = load_target_queries(atk_cfg["target_queries_path"])
    target_queries = [d["query"] for d in target_data]
    target_answer = atk_cfg["target_answer"]
    ground_truths = [d["ground_truth"] for d in target_data]

    # Use poisoned KB for baseline attack measurement, clean KB otherwise
    if kb_path is None:
        if mode == "baseline":
            kb_path = atk_cfg["kb_path"]
        else:
            kb_path = None  # let pipeline decide from mtd_config

    # Init pipeline
    pipeline = RAGPipeline(
        mode=mode,
        rag_config_path=rag_config_path,
        mtd_config_path=mtd_config_path,
        kb_path=kb_path,
    )

    # Run all target queries
    outputs = pipeline.run_batch(target_queries)

    # For MRR: use doc_id "seed_docs_0" as relevant for first query, etc.
    # In real experiments, relevant_doc_ids would come from ground truth annotations
    relevant_doc_ids = [f"seed_docs_{i}" for i in range(len(target_queries))]

    # Compute metrics
    metrics = summarize(outputs, target_queries, target_answer, relevant_doc_ids, ground_truths)

    # Build results record
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results = {
        "mode": mode,
        "attack_type": atk_cfg["type"],
        "attacker_level": atk_cfg["attacker_level"],
        "seed": seed,
        "timestamp": timestamp,
        "kb_path": kb_path,
        "metrics": metrics,
        "individual_outputs": [
            {
                "query": q,
                "answer": o.answer,
                "retrieval_latency_ms": round(o.retrieval_latency_ms, 2),
                "generation_latency_ms": round(o.generation_latency_ms, 2),
                "top_chunks": [r.chunk.text[:100] for r in o.retrieved_chunks[:3]],
            }
            for q, o in zip(target_queries, outputs)
        ],
    }

    # Save results
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{mode}_{atk_cfg['type']}_{atk_cfg['attacker_level']}_seed{seed}_{timestamp}.json"
    results_path = results_dir / filename
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", results_path)
    return results


def print_summary(results: dict) -> None:
    """Print formatted summary table to stdout."""
    m = results["metrics"]
    lat = m["latency"]
    print("\n┌─────────────────────────────────────────────────┐")
    print(f"│  Mode     : {results['mode']:<36}│")
    print(f"│  Attack   : {results['attack_type']} / {results['attacker_level']:<24}│")
    print(f"│  Seed     : {results['seed']:<36}│")
    print(f"│  ASR      : {m['asr'] * 100:>5.1f}%{' ← target ≈ 90–97%' if results['mode'] == 'baseline' else '':<30}│")
    print(f"│  MRR@10   : {m['mrr10']:.4f}{'':<33}│")
    print(f"│  Mean F1  : {m['mean_f1']:.4f}{'':<33}│")
    print(f"│  Retr Lat : {lat['retrieval_ms']:.0f} ms{'':<34}│")
    print(f"│  Gen Lat  : {lat['generation_ms']:.0f} ms{'':<34}│")
    print("└─────────────────────────────────────────────────┘\n")


def main() -> None:
    """CLI entry point for evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ATLAS-MTD-RAG Evaluator")
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "mtd"],
                        help="Pipeline mode: baseline or mtd")
    parser.add_argument("--config", type=str, default="config/attack_config.yaml",
                        help="Path to attack config")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--kb-path", type=str, default=None,
                        help="Override KB path")
    args = parser.parse_args()

    results = run_evaluation(
        mode=args.mode,
        attack_config_path=args.config,
        seed=args.seed,
        kb_path=args.kb_path,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
