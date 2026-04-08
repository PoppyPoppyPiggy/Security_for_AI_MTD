# ATLAS-MTD-RAG: Moving Target Defense for RAG Pipelines

**A MITRE ATLAS-Driven Defense Framework against Knowledge Base Poisoning**

Paper: *"Moving Target Defense for RAG Pipelines: A MITRE ATLAS-Driven Defense Framework against Knowledge Base Poisoning"*
Target venue: INTCOM 2026 (IEEE, 6-page format)

---

## Overview

RAG (Retrieval-Augmented Generation) pipelines are vulnerable to **knowledge base poisoning attacks**. An attacker injects adversarial documents into the KB, causing the retriever to surface poisoned content and the LLM to generate attacker-controlled answers. PoisonedRAG (Zou et al., USENIX Security 2025) demonstrated **97% attack success rates** with just 5 injected documents.

This project implements a **Moving Target Defense (MTD)** framework that disrupts these attacks by dynamically rotating the attack surface:

| Defense Layer | MTD Strategy | What Rotates | ATLAS TTP Mitigated |
|--------------|-------------|-------------|-------------------|
| Knowledge Base | **Shuffling** | Active KB from a pool of snapshots | AML.T0054 — False RAG Entry Injection |
| Retriever | **Diversity** | Dense / Sparse / Hybrid strategy | AML.T0020 — Poison Training Data |
| Embedding Model | **Redundancy** | Single model / Multi-model ensemble | AML.T0051 — LLM Prompt Injection |

The attacker optimizes their poison for one specific configuration. By the time their poisoned query arrives, the configuration has changed.

### Key Results

| Metric | Baseline (No Defense) | With MTD | Improvement |
|--------|---------------------|----------|------------|
| **ASR** (extractive) | 100.0% | 40.0% | -60.0 pp |
| **ASR** (GGUF LLM) | 0.0% | 20.0% | — |
| **MRR@10** | 0.1667 | 0.6000 | +0.4333 |
| **Mean F1** | 0.1585 | 0.3610 | +0.2025 |

> ASR = Attack Success Rate (lower is better). Extractive mode models worst-case where LLM echoes top chunk verbatim. GGUF mode uses TinyLlama 1.1B for real inference.

---

## Project Structure

```
security-for-ai/
├── config/
│   ├── rag_config.yaml            # RAG pipeline (chunking, retrieval, LLM)
│   ├── mtd_config.yaml            # MTD engine (rotation policies, thresholds)
│   └── attack_config.yaml         # Attacker (poison docs, similarity thresholds)
│
├── src/
│   ├── utils.py                   # safe_load_config() — shared config loader
│   │
│   ├── rag_pipeline/              # Phase 1 — Target system
│   │   ├── indexer.py             #   Document loading, chunking, FAISS+BM25 indexing
│   │   ├── retriever.py           #   Dense / Sparse / Hybrid retrieval
│   │   ├── generator.py           #   Extractive or GGUF LLM generation
│   │   └── pipeline.py            #   End-to-end orchestrator with MTD hooks
│   │
│   ├── attacker/                  # Phase 2 — Attack modules (research testbed)
│   │   ├── poisoned_rag.py        #   AML.T0054: naive + optimized KB poisoning
│   │   ├── corpus_poison.py       #   AML.T0020: L1/L2 adversarial passages
│   │   └── indirect_injection.py  #   AML.T0051: hidden instruction injection
│   │
│   ├── mtd_engine/                # Phase 3 — Defense engine
│   │   ├── kb_rotator.py          #   KB pool rotation (round-robin/random/adaptive)
│   │   ├── retriever_switcher.py  #   Retrieval strategy cycling
│   │   ├── embed_rotator.py       #   Embedding model rotation + ensemble
│   │   └── mtd_controller.py      #   Unified orchestrator (state machine)
│   │
│   ├── evaluation/                # Phase 4 — Metrics & benchmarks
│   │   ├── metrics.py             #   ASR, MRR@10, F1, latency (pure functions)
│   │   ├── evaluator.py           #   CLI experiment runner
│   │   ├── extended_evaluator.py  #   Benchmark evaluation (NARR, coverage, faithfulness)
│   │   ├── benchmark_adapter.py   #   RAGSecBench / SafeRAG / RAGChecker adapters
│   │   └── comparative_table.py   #   Paper Table II generator (ASCII + LaTeX)
│   │
│   └── atlas_mapper/              # Phase 4 — MITRE ATLAS mapping
│       ├── ttp_definitions.py     #   5 RAG-relevant ATLAS TTPs
│       └── mitigation_mapper.py   #   TTP-to-MTD mapping + Table I generator
│
├── data/
│   ├── attack_corpus/
│   │   └── target_queries.json    # 5 cybersecurity Q&A pairs with ground truth
│   ├── knowledge_bases/
│   │   ├── kb_clean/              # 20 cybersecurity paragraphs (seed KB)
│   │   ├── kb_poisoned/           # Generated at runtime by attacker modules
│   │   └── kb_rotated/            # Shuffled snapshots for MTD rotation pool
│   └── results/                   # JSON experiment output files
│
├── models/
│   └── tinyllama-1.1b-q4.gguf    # 638MB local LLM (not tracked in git)
│
├── docs/
│   ├── ARCHITECTURE.md            # Detailed system design & data flow
│   ├── EXPERIMENTS.md             # Experiment reproduction guide
│   ├── table_I_atlas_mapping.txt  # Paper Table I: ATLAS TTP ↔ MTD mapping
│   ├── table_II_comparison.txt    # Paper Table II: Framework comparison
│   └── table_II_comparison.tex    # LaTeX version of Table II
│
└── tests/                         # 145 tests across 18 test files
```

---

## Installation

### Prerequisites

- Python 3.10+
- ~2GB disk (for embedding models + GGUF LLM)

### Setup

```bash
# Clone
git clone https://github.com/PoppyPoppyPiggy/Security_for_AI_MTD.git
cd Security_for_AI_MTD

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Dependencies
pip install torch sentence-transformers faiss-cpu rank-bm25 pyyaml numpy llama-cpp-python httpx pytest

# Download TinyLlama GGUF model (638MB)
mkdir -p models
wget -O models/tinyllama-1.1b-q4.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### Verify Installation

```bash
# Run all 145 tests
pytest tests/ -v

# Quick smoke test
python -c "
from src.rag_pipeline.pipeline import RAGPipeline
p = RAGPipeline(mode='baseline', kb_path='data/knowledge_bases/kb_clean/')
out = p.run('What is a firewall?')
print('Answer:', out.answer[:100])
"
```

---

## Quick Start

### 1. Build Poisoned KB

```bash
python -c "
from src.attacker.poisoned_rag import PoisonedRAGAttacker
import json, pathlib, shutil

atk = PoisonedRAGAttacker()
queries = json.loads(pathlib.Path('data/attack_corpus/target_queries.json').read_text())

# Clear and rebuild poisoned KB
shutil.rmtree('data/knowledge_bases/kb_poisoned/', ignore_errors=True)
for q in queries:
    docs = atk.craft_poison_docs(q['query'], q['target_answer'], num_docs=5, strategy='naive')
    atk.inject_into_kb('data/knowledge_bases/kb_clean', docs, 'data/knowledge_bases/kb_poisoned')
print('Poisoned KB built.')
"
```

### 2. Run Baseline vs MTD Comparison

```bash
# Internal queries (5 queries, fast)
python -m src.evaluation.evaluator --mode baseline --seed 42
python -m src.evaluation.evaluator --mode mtd --seed 42
```

### 3. Run Extended Benchmark Evaluation

```bash
# Compare baseline vs MTD on RAGSecBench (39 queries, 13 attack categories)
python -m src.evaluation.extended_evaluator --benchmark ragsecbench --mode compare

# All benchmarks combined (64 queries)
python -m src.evaluation.extended_evaluator --benchmark all --mode compare
```

### 4. Generate Paper Tables

```bash
# Table I: ATLAS TTP ↔ MTD mapping
python -c "from src.atlas_mapper.mitigation_mapper import MitigationMapper; print(MitigationMapper.generate_table_I())"

# Table II: Framework comparison (ASCII + LaTeX)
python -m src.evaluation.comparative_table
```

---

## How the Defense Works

### The Attack (Baseline — No Defense)

```
Attacker injects 5 poison docs into KB
  ↓
User query: "What is a firewall?"
  ↓
Retriever: ranks poison doc #1 (optimized for this query)
  ↓
Generator: outputs attacker's answer → ASR = 100%
```

### The Defense (MTD — Shuffling + Diversity + Redundancy)

```
Query 1-2:  KB=clean, Retriever=dense    → poison docs not in KB → ASR = 0%
Query 3-4:  KB=poisoned, Retriever=sparse → poison docs exist but sparse
                                            ranking differs → ASR varies
Query 5-6:  KB=snapshot_1, Retriever=hybrid → poison docs not in KB → ASR = 0%
Query 7-8:  KB=snapshot_2, Retriever=dense  → clean KB → ASR = 0%
  ↓
MTD Controller rotates after every N queries
Anomaly detection triggers early rotation on suspicious scores
Embedding model rotates to change the retrieval surface
  ↓
Overall ASR drops from 100% → 40% (extractive) or 20% (real LLM)
```

### MTD State Machine

```
IDLE ──→ ROTATING ──→ ACTIVE ──→ MONITORING ──→ (loop)
  │         │            │            │
  │    KB rotated    New config    Anomaly check
  │    Epoch++       applied       Score updated
  │                               Retriever may switch
  │                               Embed model may rotate
  └── (passthrough if mtd.enabled=false)
```

---

## Evaluation Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **ASR** | `#(target_answer in output) / #queries` | Attack success rate (lower = better defense) |
| **MRR@10** | `mean(1/rank of first relevant doc)` | Retrieval quality (higher = better) |
| **F1** | Token-level precision × recall | Answer quality vs ground truth |
| **NARR** | `(ASR_base - ASR_mtd) / ASR_base` | Normalized attack reduction (higher = better) |
| **Defense Coverage** | `#categories with ASR<0.5 / #categories` | Breadth across attack types |
| **Faithfulness** | `supported_claims / total_claims` | Grounded in retrieved context |
| **Hallucination Rate** | `hallucinated_claims / total_claims` | Unsupported claims |

---

## MITRE ATLAS Mapping

This framework maps 5 ATLAS techniques to 3 MTD strategies (the SDR Triad):

| ATLAS TTP | Technique | MTD Strategy | Component | Effectiveness |
|-----------|-----------|-------------|-----------|--------------|
| AML.T0054 | False RAG Entry Injection | Shuffling | kb_rotator | High |
| AML.T0020 | Poison Training Data | Diversity | retriever_switcher | High |
| AML.T0056 | Gather RAG-Indexed Targets | Shuffling | kb_rotator | Medium |
| AML.T0057 | RAG Database Prompting | Diversity | retriever_switcher | Medium |
| AML.T0051 | LLM Prompt Injection | Redundancy | embed_rotator | Medium |

---

## Benchmark Integration

The framework supports evaluation against community benchmarks with offline synthetic fallback:

| Benchmark | Source | Queries | Attack Categories |
|-----------|--------|---------|-------------------|
| **Internal** | Our target_queries.json | 5 | naive_poisoning |
| **RAGSecBench** | arXiv 2505.18543 | 39 (synthetic) | 13 categories |
| **SafeRAG** | ACL 2025 | 20 (synthetic) | 4 dimensions |
| **RAGChecker** | NeurIPS 2024 | — (metric only) | Faithfulness evaluation |

When real benchmark datasets are available, place them at the paths specified in `config/attack_config.yaml` under `benchmarks:`. The adapters automatically switch from synthetic to real data.

---

## Comparison with Existing Defenses

| Framework | Venue | Dynamic | Train-Free | Certified | ATLAS | ASR |
|-----------|-------|---------|-----------|-----------|-------|-----|
| RobustRAG | ICML 2024 | No | Yes | **Yes** | No | certified |
| TrustRAG | AAAI-W 2026 | No | Yes | No | No | ~0.10 |
| SeCon-RAG | NeurIPS 2025 | No | No | No | No | 0.02 |
| ReliabilityRAG | NeurIPS 2025 | No | Yes | **Yes** | No | provable |
| RAGuard | NeurIPS 2025 | No | No | No | No | varies |
| RAGShield | arXiv 2026 | No | Yes | No | No | 0.00 |
| **ATLAS-MTD-RAG (Ours)** | INTCOM 2026 | **Yes** | **Yes** | No | **Yes** | 0.20-0.40 |

**Our unique contribution:** First framework combining dynamic MTD defense + training-free deployment + MITRE ATLAS TTP mapping for RAG pipelines.

---

## Configuration

All parameters are config-driven (zero hardcoding in source):

**`config/rag_config.yaml`** — RAG pipeline
```yaml
rag:
  chunk_size: 256              # characters per chunk
  top_k: 5                     # retrieved chunks per query
  retrieval_strategy: dense    # dense | sparse | hybrid
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  llm_provider: llama_cpp      # extractive | llama_cpp
  llm_model_path: models/tinyllama-1.1b-q4.gguf
```

**`config/mtd_config.yaml`** — MTD defense
```yaml
mtd:
  enabled: true
  rotation_policy: round-robin  # round-robin | random | adaptive
  rotation_interval: 10         # rotate KB every N queries
  kb_pool:                      # KB rotation pool
    - data/knowledge_bases/kb_clean/
    - data/knowledge_bases/kb_rotated/snapshot_1/
    - data/knowledge_bases/kb_rotated/snapshot_2/
  retrieval_sequence: [dense, sparse, hybrid]
  embed_models:
    - sentence-transformers/all-MiniLM-L6-v2
    - BAAI/bge-small-en-v1.5
```

**`config/attack_config.yaml`** — Attack parameters
```yaml
attack:
  type: poisoned_rag
  num_poison_docs: 5
  target_answer: "INJECTED_ANSWER"
  attacker_level: L1            # L1 (black-box) | L2 (gray-box)
```

---

## Testing

```bash
# Run all 145 tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_kb_rotator.py -v          # MTD rotation
pytest tests/test_poisoned_rag.py -v        # Attack modules
pytest tests/test_benchmark_adapters.py -v  # Benchmark integration
```

---

## Citation

```bibtex
@inproceedings{kim2026atlas-mtd-rag,
  title={Moving Target Defense for RAG Pipelines: A MITRE ATLAS-Driven Defense Framework against Knowledge Base Poisoning},
  author={Kim, Minseong and Kim, Do-hoon},
  booktitle={IEEE International Conference on Information Technology (INTCOM)},
  year={2026}
}
```

---

## References

- Zou et al., "PoisonedRAG: Knowledge Corruption Attacks to RAG", USENIX Security 2025
- Xie et al., "BadRAG: Identifying Vulnerabilities in RAG", 2024
- Thornton, "Semantic Chameleon: Corpus-Dependent Poisoning", 2026
- Xie et al., "RobustRAG: Certifiably Robust RAG", ICML 2024
- Zhou et al., "TrustRAG: Enhancing Robustness and Trustworthiness", AAAI 2026
- Lan et al., "SeCon-RAG: Two-Stage Semantic Filtering", NeurIPS 2025
- MITRE ATLAS: https://atlas.mitre.org

---

## License

Research use only. DS Lab, Kyonggi University.
