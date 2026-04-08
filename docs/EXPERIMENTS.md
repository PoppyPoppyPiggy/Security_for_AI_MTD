# Experiment Reproduction Guide

This document describes how to reproduce all experiments in the paper.

---

## Prerequisites

```bash
# Ensure virtual environment is active
source .venv/bin/activate

# Verify all tests pass (confirms correct installation)
pytest tests/ -v
# Expected: 145 passed

# Verify GGUF model exists
ls -lh models/tinyllama-1.1b-q4.gguf
# Expected: ~638MB file
```

---

## Experiment 1: Baseline Attack Verification

**Goal:** Verify that PoisonedRAG achieves ASR >= 80% on the undefended pipeline.

### Step 1: Build Poisoned Knowledge Base

```bash
python -c "
from src.attacker.poisoned_rag import PoisonedRAGAttacker
import json, pathlib, shutil

atk = PoisonedRAGAttacker()
queries = json.loads(pathlib.Path('data/attack_corpus/target_queries.json').read_text())

shutil.rmtree('data/knowledge_bases/kb_poisoned/', ignore_errors=True)
for q in queries:
    docs = atk.craft_poison_docs(q['query'], q['target_answer'], num_docs=5, strategy='naive')
    atk.inject_into_kb('data/knowledge_bases/kb_clean', docs, 'data/knowledge_bases/kb_poisoned')

print(f'Poisoned KB: {len(list(pathlib.Path(\"data/knowledge_bases/kb_poisoned\").iterdir()))} files')
"
```

### Step 2: Run Baseline Evaluation

```bash
# Extractive mode (worst-case — LLM echoes top chunk)
python -m src.evaluation.evaluator --mode baseline --seed 42

# Expected output:
# ASR      : 100.0%  ← poison docs ranked #1 for all 5 queries
# MRR@10   : 0.1667
# Mean F1  : 0.1585
```

### Step 3: Verify with GGUF LLM

Edit `config/rag_config.yaml` to set `llm_provider: llama_cpp`, then:

```bash
python -m src.evaluation.evaluator --mode baseline --seed 42

# Expected output:
# ASR      :   0.0%  ← LLM paraphrases instead of echoing "INJECTED_ANSWER"
# Mean F1  : 0.4225  ← higher because LLM generates coherent answers
```

> **Note:** ASR is 0% with GGUF because the LLM doesn't output the literal string "INJECTED_ANSWER". The poison docs ARE retrieved (check `top_chunks` in the JSON results), but the LLM interprets them rather than echoing verbatim. This demonstrates the difference between worst-case (extractive) and realistic (LLM) attack models.

---

## Experiment 2: MTD Defense Verification

**Goal:** Verify that MTD reduces ASR below 40%.

### Step 1: Configure MTD

Ensure `config/mtd_config.yaml` has:
```yaml
mtd:
  enabled: true
  rotation_policy: round-robin
  rotation_interval: 10
  kb_pool:
    - data/knowledge_bases/kb_clean/
    - data/knowledge_bases/kb_rotated/snapshot_1/
    - data/knowledge_bases/kb_rotated/snapshot_2/
```

### Step 2: Run MTD Evaluation

```bash
# Set rag_config.yaml back to llm_provider: extractive for comparable ASR
python -m src.evaluation.evaluator --mode mtd --seed 42

# Expected output:
# ASR      :  40.0%  ← reduced from 100% baseline
# MRR@10   : 0.6000  ← improved (clean KB returns correct docs)
# Mean F1  : 0.3610  ← improved
```

### Step 3: Verify Per-Query Behavior

```bash
python -c "
import json, glob
f = sorted(glob.glob('data/results/mtd_*.json'))[-1]
r = json.load(open(f))
for o in r['individual_outputs']:
    hit = 'INJECTED_ANSWER' in o['answer']
    print(f\"ASR hit={hit} | {o['query'][:50]}\")
"
```

Expected: queries served from clean KBs show `hit=False`, queries served from poisoned KB (if in pool) show `hit=True`.

---

## Experiment 3: Benchmark Evaluation

**Goal:** Evaluate defense across standardized attack categories.

### RAGSecBench (13 attack categories, 39 queries)

```bash
python -m src.evaluation.extended_evaluator --benchmark ragsecbench --mode compare

# Output: comparison table with ASR, NARR, Defense Coverage for both modes
```

### SafeRAG (4 dimensions, 20 queries)

```bash
python -m src.evaluation.extended_evaluator --benchmark saferag --mode compare
```

### All Benchmarks Combined (64 queries)

```bash
python -m src.evaluation.extended_evaluator --benchmark all --mode compare
```

### Interpreting Results

- **NARR** (Normalized Attack Reduction Rate): `(ASR_base - ASR_mtd) / ASR_base`. Values close to 1.0 mean nearly all attacks are mitigated. Values close to 0.0 mean MTD has little effect.
- **Defense Coverage Rate**: fraction of attack categories where ASR < 50%. Coverage of 1.0 means MTD reduces attacks in every category.
- **Faithfulness Score**: fraction of answer claims supported by retrieved context. Higher is better.

---

## Experiment 4: Generate Paper Tables

### Table I: ATLAS TTP to MTD Mapping

```bash
python -c "
from src.atlas_mapper.mitigation_mapper import MitigationMapper
print(MitigationMapper.generate_table_I())
"
# Output saved to: docs/table_I_atlas_mapping.txt
```

### Table II: Framework Comparison

```bash
# Without experiment data (uses placeholder for our row)
python -m src.evaluation.comparative_table

# With experiment data (auto-loads latest results)
python -c "
from src.evaluation.extended_evaluator import run_comparison
from src.evaluation.comparative_table import generate_table_II
results = run_comparison(benchmark='internal')
print(generate_table_II(our_results=results))
"
# Output saved to: docs/table_II_comparison.txt and docs/table_II_comparison.tex
```

---

## Experiment 5: Attack Variant Comparison

### Naive vs Optimized Poisoning

```bash
python -c "
from src.attacker.poisoned_rag import PoisonedRAGAttacker
atk = PoisonedRAGAttacker()

# Naive: template-based injection
naive_docs = atk.craft_poison_docs('What is a firewall?', 'INJECTED_ANSWER', 5, 'naive')

# Optimized: embedding-similarity maximized
opt_docs = atk.craft_poison_docs('What is a firewall?', 'INJECTED_ANSWER', 5, 'optimized')

for i, (n, o) in enumerate(zip(naive_docs, opt_docs)):
    print(f'Doc {i}: naive={len(n.text)} chars, optimized={len(o.text)} chars')
"
```

### Corpus Poisoning (L1 vs L2)

```bash
python -c "
from src.attacker.corpus_poison import CorpusPoisonAttacker
from src.rag_pipeline.indexer import Chunk

atk = CorpusPoisonAttacker()
chunks = [Chunk(text='Firewalls filter traffic.', source_path='t', chunk_id='c0', doc_id='d0')]

l1 = atk.craft_adversarial_passage('What is a firewall?', 'INJECTED', chunks, 'L1')
l2 = atk.craft_adversarial_passage('What is a firewall?', 'INJECTED', chunks, 'L2')

print(f'L1 similarity: {l1.similarity_to_query:.4f}')
print(f'L2 similarity: {l2.similarity_to_query:.4f}')
print(f'L2 >= L1: {l2.similarity_to_query >= l1.similarity_to_query}')
"
```

### Indirect Injection Styles

```bash
python -c "
from src.attacker.indirect_injection import IndirectInjectionAttacker
atk = IndirectInjectionAttacker()

for style in ['override', 'role_play', 'data_exfil']:
    doc = atk.craft_injection_document('INJECTED_ANSWER', style)
    print(f'{style}: {doc.hidden_instruction[:60]}...')
"
```

---

## Experiment Parameters

All experiments use these defaults (configurable in YAML):

| Parameter | Value | Config Key |
|-----------|-------|-----------|
| Random seed | 42 | CLI `--seed` |
| Top-k retrieval | 5 | `rag.top_k` |
| Chunk size | 256 chars | `rag.chunk_size` |
| Embedding model | all-MiniLM-L6-v2 | `rag.embedding_model` |
| LLM (GGUF) | TinyLlama 1.1B Q4 | `rag.llm_model_path` |
| Poison docs per query | 5 | `attack.num_poison_docs` |
| KB rotation interval | 10 queries | `mtd.rotation_interval` |
| Retriever batch size | 5 queries | `mtd.batch_size` |
| Rotation policy | round-robin | `mtd.rotation_policy` |

---

## Output Files

All experiment results are saved as JSON in `data/results/`:

```
data/results/
├── baseline_poisoned_rag_L1_seed42_YYYYMMDD_HHMMSS.json
├── mtd_poisoned_rag_L1_seed42_YYYYMMDD_HHMMSS.json
├── bench_ragsecbench_baseline_seed42_YYYYMMDD_HHMMSS.json
├── bench_ragsecbench_mtd_seed42_YYYYMMDD_HHMMSS.json
├── bench_saferag_baseline_seed42_YYYYMMDD_HHMMSS.json
└── bench_all_mtd_seed42_YYYYMMDD_HHMMSS.json
```

Each JSON contains:
- `mode`: "baseline" or "mtd"
- `metrics`: ASR, MRR@10, F1, latency
- `defense_coverage`: per-category ASR breakdown
- `faithfulness`: claim-level evaluation (if enabled)
- `individual_outputs`: per-query answers and top retrieved chunks

---

## Troubleshooting

**GGUF model not found:**
```bash
# Download TinyLlama 1.1B
mkdir -p models
wget -O models/tinyllama-1.1b-q4.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**Empty poisoned KB:**
```bash
# Rebuild — the poisoned KB is generated, not stored in git
python -c "
from src.attacker.poisoned_rag import PoisonedRAGAttacker
import json, pathlib
atk = PoisonedRAGAttacker()
queries = json.loads(pathlib.Path('data/attack_corpus/target_queries.json').read_text())
for q in queries:
    docs = atk.craft_poison_docs(q['query'], q['target_answer'], 5, 'naive')
    atk.inject_into_kb('data/knowledge_bases/kb_clean', docs, 'data/knowledge_bases/kb_poisoned')
"
```

**CUDA warning:**
The `CUDA initialization` warning is harmless — the system runs on CPU. Ignore it.

**Test failures after config change:**
Some tests create temporary configs. If you changed `mtd_config.yaml` to have fewer than 2 KBs in the pool, the `test_pool_too_small_raises` test may behave differently. Ensure the production config always has `kb_pool_size >= 2`.
