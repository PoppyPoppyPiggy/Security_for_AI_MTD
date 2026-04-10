#!/usr/bin/env python3
"""
DEMO: LLM Terminal — watch how TinyLlama handles poison documents.

Usage:
  python scripts/demo_llm.py

Shows:
  For each query:
    1. Retrieved chunks (showing which are poison vs clean)
    2. Extractive answer (worst-case: echoes top chunk)
    3. GGUF LLM answer (realistic: model generates own answer)
    4. Whether attack succeeded in each mode
"""
from __future__ import annotations

import json
import time

from src.rag_pipeline.generator import Generator
from src.rag_pipeline.indexer import DocumentIndexer
from src.rag_pipeline.retriever import Retriever
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


def main() -> None:
    atk_cfg = safe_load_config("config/attack_config.yaml")["attack"]
    rag_cfg = safe_load_config("config/rag_config.yaml")["rag"]

    target_answer = atk_cfg["target_answer"]
    kb_poisoned = atk_cfg["kb_poisoned_path"]

    with open(atk_cfg["target_queries_path"]) as f:
        target_data = json.load(f)

    banner("LLM TERMINAL — Extractive vs GGUF on Poison Docs", MAGENTA)

    # Build index from poisoned KB
    print(f"  {DIM}Building index from poisoned KB: {kb_poisoned}{RESET}")
    indexer = DocumentIndexer()
    docs = indexer.load_documents(kb_poisoned)
    chunks = indexer.chunk_documents(docs)
    dense_idx = indexer.build_dense_index(chunks)
    sparse_idx = indexer.build_sparse_index(chunks)

    retriever = Retriever(
        strategy=rag_cfg["retrieval_strategy"],
        dense_index=dense_idx, sparse_index=sparse_idx,
        chunks=chunks, encoder=indexer.encoder,
    )

    # Init generators
    print(f"  {DIM}Loading extractive generator...{RESET}")
    ext_gen = Generator(mode="extractive")

    print(f"  {DIM}Loading GGUF LLM ({rag_cfg['llm_model_path']})...{RESET}")
    llm_gen = Generator(mode="llama_cpp")
    print(f"  {GREEN}Ready.{RESET}\n")

    # Demo queries
    demo_queries = target_data[:3]
    ext_hits = 0
    llm_hits = 0

    for qi, qd in enumerate(demo_queries, 1):
        query = qd["query"]
        ground_truth = qd["ground_truth"]

        print(f"  {BOLD}{CYAN}━━━ Query {qi} ━━━{RESET}")
        print(f"  {BOLD}Q:{RESET} {query}")
        print()

        # Retrieve
        retrieved = retriever.retrieve(query, top_k=5)

        print(f"  {YELLOW}Retrieved chunks (top 5):{RESET}")
        for ri, r in enumerate(retrieved, 1):
            is_poison = target_answer in r.chunk.text
            tag = f"{RED}POISON{RESET}" if is_poison else f"{GREEN}CLEAN{RESET}"
            text_preview = r.chunk.text[:70].replace("\n", " ")
            print(f"    #{ri} [{tag}] sim={r.score:.3f} | {text_preview}...")
        print()

        # Extractive answer
        ext_out = ext_gen.generate(query, retrieved)
        ext_hit = target_answer.lower() in ext_out.answer.lower()
        if ext_hit:
            ext_hits += 1

        ext_status = f"{RED}ATTACK SUCCEEDED{RESET}" if ext_hit else f"{GREEN}ATTACK FAILED{RESET}"
        print(f"  {YELLOW}Extractive (worst-case):{RESET}")
        print(f"    {DIM}Answer:{RESET} {ext_out.answer[:120]}")
        print(f"    Result: [{ext_status}]")
        print()

        # GGUF answer
        print(f"  {MAGENTA}GGUF LLM (TinyLlama — generating...){RESET}")
        llm_out = llm_gen.generate(query, retrieved)
        llm_hit = target_answer.lower() in llm_out.answer.lower()
        if llm_hit:
            llm_hits += 1

        llm_status = f"{RED}ATTACK SUCCEEDED{RESET}" if llm_hit else f"{GREEN}ATTACK FAILED{RESET}"
        print(f"    {DIM}Answer:{RESET} {llm_out.answer[:200]}")
        print(f"    Tokens: {llm_out.token_count}")
        print(f"    Result: [{llm_status}]")
        print()

        # Ground truth
        print(f"  {BLUE}Ground truth:{RESET}")
        print(f"    {ground_truth[:120]}")
        print()
        print(f"  {'─' * 64}")
        print()

    # ── Summary ──
    n = len(demo_queries)
    banner("LLM COMPARISON SUMMARY", MAGENTA)
    print(f"  {'Mode':<25} {'ASR':>10} {'Explanation':<30}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 30}")
    print(f"  {RED}{'Extractive (worst)':<25}{RESET} {ext_hits}/{n} ({ext_hits/n*100:.0f}%)   Echoes top chunk verbatim")
    print(f"  {GREEN}{'GGUF LLM (realistic)':<25}{RESET} {llm_hits}/{n} ({llm_hits/n*100:.0f}%)   LLM paraphrases, ignores poison")
    print()
    print(f"  {DIM}Key insight: even with poison docs retrieved as #1,")
    print(f"  a real LLM often generates its own coherent answer")
    print(f"  rather than echoing the injected text verbatim.{RESET}")
    print()


if __name__ == "__main__":
    main()
