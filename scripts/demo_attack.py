#!/usr/bin/env python3
"""
DEMO: Attacker Terminal — watch poison docs get crafted and injected.

Usage:
  python scripts/demo_attack.py

Shows:
  1. Loading target queries from config
  2. Crafting poison documents (naive strategy)
  3. Injecting into KB
  4. Running baseline pipeline — all queries return INJECTED_ANSWER
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from src.attacker.poisoned_rag import PoisonedRAGAttacker
from src.rag_pipeline.pipeline import RAGPipeline
from src.utils import safe_load_config

# ── Colors ──
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


def step(num: int, text: str) -> None:
    print(f"{YELLOW}{BOLD}[Step {num}]{RESET} {text}")


def pause(seconds: float = 1.0) -> None:
    time.sleep(seconds)


def main() -> None:
    atk_cfg = safe_load_config("config/attack_config.yaml")["attack"]
    rag_cfg = safe_load_config("config/rag_config.yaml")["rag"]

    target_answer = atk_cfg["target_answer"]
    queries_path = atk_cfg["target_queries_path"]
    kb_clean = atk_cfg["kb_clean_path"]
    kb_poisoned = atk_cfg["kb_poisoned_path"]
    num_poison = atk_cfg["num_poison_docs"]

    banner("ATTACKER TERMINAL — Knowledge Base Poisoning", RED)

    # ── Step 1: Load target queries ──
    step(1, "Loading target queries...")
    pause()
    with open(queries_path) as f:
        target_data = json.load(f)
    print(f"  {DIM}Source: {queries_path}{RESET}")
    print(f"  {BOLD}Total queries: {len(target_data)}{RESET}")
    print(f"  {BOLD}Target answer: {RED}{target_answer}{RESET}")
    print()

    demo_queries = target_data[:5]
    for i, q in enumerate(demo_queries):
        print(f"  {DIM}[{i+1}]{RESET} {q['query'][:60]}")
    if len(target_data) > 5:
        print(f"  {DIM}... and {len(target_data) - 5} more{RESET}")
    pause()

    # ── Step 2: Craft poison documents ──
    step(2, f"Crafting {num_poison} poison docs per query (naive strategy)...")
    pause()
    atk = PoisonedRAGAttacker()

    sample_docs = atk.craft_poison_docs(
        demo_queries[0]["query"], target_answer,
        num_docs=num_poison, strategy="naive",
    )
    print(f"\n  {MAGENTA}Example poison doc for: \"{demo_queries[0]['query'][:50]}...\"{RESET}")
    print()
    for i, doc in enumerate(sample_docs[:3]):
        text_preview = doc.text[:100].replace("\n", " ")
        print(f"    {RED}poison_{i}:{RESET} {text_preview}...")
    print(f"    {DIM}({num_poison} docs total per query){RESET}")
    pause()

    # ── Step 3: Inject into KB ──
    step(3, f"Injecting into knowledge base...")
    pause()

    # Clear and rebuild
    poisoned_path = Path(kb_poisoned)
    if poisoned_path.exists():
        shutil.rmtree(poisoned_path)

    total_injected = 0
    for q in target_data:
        docs = atk.craft_poison_docs(q["query"], target_answer, num_docs=num_poison, strategy="naive")
        atk.inject_into_kb(kb_clean, docs, kb_poisoned)
        total_injected += len(docs)

    n_txt = len(list(poisoned_path.glob("*.txt")))
    n_json = len(list(poisoned_path.glob("*.json")))
    print(f"  {DIM}Clean KB: {kb_clean} (original docs){RESET}")
    print(f"  {RED}Poisoned KB: {kb_poisoned}{RESET}")
    print(f"  {BOLD}Injected: {total_injected} poison docs ({n_txt} txt + {n_json} json){RESET}")
    pause()

    # ── Step 4: Verify attack ──
    step(4, "Verifying attack — running baseline pipeline on poisoned KB...")
    pause()

    pipeline = RAGPipeline(mode="baseline", kb_path=kb_poisoned)

    print()
    successes = 0
    test_queries = demo_queries[:5]
    for i, qd in enumerate(test_queries, 1):
        out = pipeline.run(qd["query"])
        hit = target_answer.lower() in out.answer.lower()
        if hit:
            successes += 1

        status_icon = f"{RED}POISONED{RESET}" if hit else f"{GREEN}CLEAN{RESET}"
        print(f"  {BOLD}Q{i}:{RESET} {qd['query'][:55]}")
        print(f"     {DIM}Answer:{RESET} {out.answer[:90]}...")
        print(f"     Status: [{status_icon}]  Score: {out.retrieved_chunks[0].score:.3f}")
        print()

    asr = successes / len(test_queries) * 100
    print(f"  {RED}{BOLD}Attack Success Rate: {asr:.0f}% ({successes}/{len(test_queries)} queries poisoned){RESET}")

    banner("ATTACK COMPLETE — KB is compromised", RED)
    print(f"  {DIM}The poisoned KB now contains {total_injected} fake documents.")
    print(f"  Any query to this KB will likely return INJECTED_ANSWER.")
    print(f"  Next: run {CYAN}python scripts/demo_defense.py{RESET}{DIM} to see MTD defense.{RESET}")
    print()


if __name__ == "__main__":
    main()
