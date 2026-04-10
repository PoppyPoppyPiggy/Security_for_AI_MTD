#!/usr/bin/env python3
"""
DEMO: Defense Terminal — watch MTD rotate KB and block attacks.

Usage:
  python scripts/demo_defense.py

Shows:
  1. MTD controller initializing with KB pool
  2. Per-query: which KB is active, which retriever, whether attack was blocked
  3. Real-time KB rotation at interval boundaries
  4. Final ASR comparison vs baseline
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
    """Extract short KB name from path."""
    parts = path.rstrip("/").split("/")
    return parts[-1] if parts else path


def main() -> None:
    atk_cfg = safe_load_config("config/attack_config.yaml")["attack"]
    mtd_cfg = safe_load_config("config/mtd_config.yaml")["mtd"]

    target_answer = atk_cfg["target_answer"]
    kb_poisoned = atk_cfg["kb_poisoned_path"]

    with open(atk_cfg["target_queries_path"]) as f:
        target_data = json.load(f)

    banner("MTD DEFENSE TERMINAL — Moving Target Defense Active", GREEN)

    # ── Show MTD configuration ──
    print(f"  {BOLD}MTD Configuration:{RESET}")
    print(f"    Policy          : {CYAN}{mtd_cfg['rotation_policy']}{RESET}")
    print(f"    Rotation interval: every {mtd_cfg['rotation_interval']} queries")
    print(f"    Retriever cycle : {mtd_cfg['retrieval_sequence']}")
    print(f"    Embed models    : {len(mtd_cfg['embed_models'])} models")
    print()
    print(f"  {BOLD}KB Pool ({len(mtd_cfg['kb_pool'])} KBs):{RESET}")
    for kb in mtd_cfg["kb_pool"]:
        is_poisoned = "poisoned" in kb.lower()
        icon = f"{RED}[POISONED]{RESET}" if is_poisoned else f"{GREEN}[CLEAN]{RESET}"
        print(f"    {icon} {kb}")
    print()

    time.sleep(1)

    # ── Initialize MTD pipeline ──
    print(f"  {YELLOW}Initializing MTD pipeline...{RESET}")
    pipeline = RAGPipeline(mode="mtd")
    print(f"  {GREEN}Pipeline ready.{RESET}\n")

    time.sleep(0.5)

    # ── Run queries ──
    n_queries = min(20, len(target_data))
    queries = target_data[:n_queries]

    print(f"  {BOLD}Running {n_queries} queries with MTD defense...{RESET}\n")

    print(f"  {'#':>3}  {'Status':<12} {'KB':<16} {'Retriever':<8} {'Query':<42}")
    print(f"  {'─' * 3}  {'─' * 12} {'─' * 16} {'─' * 8} {'─' * 42}")

    successes = 0
    prev_kb = ""
    for i, qd in enumerate(queries, 1):
        out = pipeline.run(qd["query"])
        hit = target_answer.lower() in out.answer.lower()
        if hit:
            successes += 1

        curr_kb = kb_short(out.active_kb)
        is_poisoned_kb = "poisoned" in curr_kb.lower()

        # Show rotation event
        if curr_kb != prev_kb and prev_kb:
            print(f"  {MAGENTA}{BOLD}  ↻ KB ROTATED: {prev_kb} → {curr_kb}{RESET}")
        prev_kb = curr_kb

        if hit:
            status = f"{RED}{'POISONED':12}{RESET}"
        else:
            status = f"{GREEN}{'BLOCKED':12}{RESET}"

        kb_display = f"{RED}{curr_kb:<16}{RESET}" if is_poisoned_kb else f"{GREEN}{curr_kb:<16}{RESET}"
        query_short = qd["query"][:42]

        print(f"  {i:3d}  {status} {kb_display} {out.active_retriever:<8} {query_short}")

        time.sleep(0.1)

    # ── Summary ──
    asr = successes / n_queries * 100
    blocked = n_queries - successes
    narr = (98.0 - asr) / 98.0 if asr < 98 else 0.0

    print()
    banner("MTD DEFENSE RESULTS", GREEN)
    print(f"  Total queries    : {n_queries}")
    print(f"  {RED}Attacks succeeded : {successes}{RESET}")
    print(f"  {GREEN}Attacks blocked   : {blocked}{RESET}")
    print(f"  ASR (with MTD)   : {BOLD}{asr:.1f}%{RESET}")
    print(f"  ASR (no defense) : {DIM}98.0%{RESET}")
    print(f"  NARR             : {BOLD}{narr:.2f}{RESET} ({narr*100:.0f}% of attacks eliminated)")
    print()
    print(f"  {DIM}The MTD engine rotated through {len(mtd_cfg['kb_pool'])} KBs.")
    print(f"  Queries hitting clean KBs were fully protected.{RESET}")
    print()


if __name__ == "__main__":
    main()
