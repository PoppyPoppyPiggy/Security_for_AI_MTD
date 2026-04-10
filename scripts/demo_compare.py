#!/usr/bin/env python3
"""
DEMO: Comparison Terminal — baseline vs MTD side by side.

Usage:
  python scripts/demo_compare.py

Shows:
  Each query answered by both baseline (poisoned) and MTD (defended),
  with color-coded status showing which attacks were blocked.
"""
from __future__ import annotations

import json
import time

from src.rag_pipeline.pipeline import RAGPipeline
from src.utils import safe_load_config

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner(text: str, color: str = CYAN) -> None:
    width = 64
    print(f"\n{color}{BOLD}{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}{RESET}\n")


def kb_short(path: str) -> str:
    parts = path.rstrip("/").split("/")
    return parts[-1] if parts else path


def main() -> None:
    atk_cfg = safe_load_config("config/attack_config.yaml")["attack"]

    target_answer = atk_cfg["target_answer"]
    kb_poisoned = atk_cfg["kb_poisoned_path"]

    with open(atk_cfg["target_queries_path"]) as f:
        target_data = json.load(f)

    banner("COMPARISON: Baseline (No Defense) vs MTD Defense", YELLOW)

    # Init both pipelines
    print(f"  {DIM}Initializing baseline pipeline (poisoned KB)...{RESET}")
    baseline = RAGPipeline(mode="baseline", kb_path=kb_poisoned)

    print(f"  {DIM}Initializing MTD pipeline (rotating KB pool)...{RESET}")
    mtd = RAGPipeline(mode="mtd")
    print()

    n_queries = min(15, len(target_data))
    queries = target_data[:n_queries]

    base_hits = 0
    mtd_hits = 0

    for i, qd in enumerate(queries, 1):
        q = qd["query"]
        gt = qd["ground_truth"]

        out_base = baseline.run(q)
        out_mtd = mtd.run(q)

        base_poisoned = target_answer.lower() in out_base.answer.lower()
        mtd_poisoned = target_answer.lower() in out_mtd.answer.lower()

        if base_poisoned:
            base_hits += 1
        if mtd_poisoned:
            mtd_hits += 1

        print(f"  {BOLD}Query {i:2d}:{RESET} {q[:58]}")
        print()

        # Baseline answer
        base_icon = f"{RED}POISONED{RESET}" if base_poisoned else f"{GREEN}CLEAN{RESET}"
        print(f"    {RED}Baseline{RESET} [{base_icon}]")
        print(f"    {DIM}Answer:{RESET} {out_base.answer[:80]}")
        print()

        # MTD answer
        mtd_icon = f"{RED}POISONED{RESET}" if mtd_poisoned else f"{GREEN}BLOCKED{RESET}"
        mtd_kb = kb_short(out_mtd.active_kb)
        print(f"    {GREEN}MTD-SDR{RESET}  [{mtd_icon}]  KB={mtd_kb}  Retr={out_mtd.active_retriever}")
        print(f"    {DIM}Answer:{RESET} {out_mtd.answer[:80]}")
        print()

        # Ground truth
        print(f"    {BLUE}Truth{RESET}    {gt[:80]}")
        print(f"  {'─' * 64}")
        print()

        time.sleep(0.1)

    # ── Summary ──
    base_asr = base_hits / n_queries * 100
    mtd_asr = mtd_hits / n_queries * 100
    reduction = base_asr - mtd_asr

    banner("FINAL COMPARISON", YELLOW)
    print(f"  {'Metric':<25} {'Baseline':>12} {'MTD-SDR':>12} {'Delta':>10}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 12} {'─' * 10}")
    print(f"  {'ASR':<25} {RED}{base_asr:>11.1f}%{RESET} {GREEN}{mtd_asr:>11.1f}%{RESET} {CYAN}{-reduction:>+9.1f}pp{RESET}")
    print(f"  {'Attacks succeeded':<25} {RED}{base_hits:>12}{RESET} {GREEN}{mtd_hits:>12}{RESET}")
    print(f"  {'Attacks blocked':<25} {DIM}{0:>12}{RESET} {GREEN}{base_hits - mtd_hits:>12}{RESET}")

    if base_asr > 0:
        narr = (base_asr - mtd_asr) / base_asr
        print(f"  {'NARR':<25} {'—':>12} {BOLD}{narr:>11.2f}{RESET}")

    print()
    print(f"  {GREEN}{BOLD}MTD reduced attack success by {reduction:.0f} percentage points.{RESET}")
    print()


if __name__ == "__main__":
    main()
