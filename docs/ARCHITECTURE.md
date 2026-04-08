# System Architecture

## High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG Pipeline (pipeline.py)                     │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Indexer     │───→│  Retriever   │───→│  Generator   │──→ Answer   │
│  │  (indexer.py) │    │(retriever.py)│    │(generator.py)│              │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘              │
│         │                   │                                           │
│    Load docs &         Dense/Sparse/         Extractive or              │
│    build FAISS+BM25    Hybrid search         GGUF LLM                  │
│         │                   │                                           │
│  ═══════╪═══════════════════╪═══════════════════════════════════════    │
│         │    MTD Controller (mtd_controller.py)      │                  │
│         │    ┌─────────────────────────────────┐     │                  │
│         │    │  State: IDLE→ROTATING→ACTIVE→   │     │                  │
│         │    │         MONITORING               │     │                  │
│         │    └────┬──────────┬──────────┬──────┘     │                  │
│         │         │          │          │             │                  │
│    ┌────▼────┐ ┌──▼────┐ ┌──▼──────┐                                  │
│    │   KB    │ │Retr.  │ │ Embed   │                                   │
│    │Rotator  │ │Switch │ │ Rotator │                                   │
│    └─────────┘ └───────┘ └─────────┘                                   │
│    Shuffling   Diversity  Redundancy    ← SDR Triad                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐    ┌──────────────────────────────────────┐
│     Attacker Modules     │    │         Evaluation System            │
│                          │    │                                      │
│  ├─ poisoned_rag.py      │    │  ├─ metrics.py      (ASR/MRR/F1)    │
│  ├─ corpus_poison.py     │───→│  ├─ evaluator.py    (CLI runner)    │
│  └─ indirect_injection.py│    │  ├─ extended_evaluator.py (benchmarks)│
│                          │    │  ├─ benchmark_adapter.py  (adapters) │
│  Generates poisoned KBs  │    │  └─ comparative_table.py (Table II)  │
│  for attack evaluation   │    │                                      │
└──────────────────────────┘    └──────────────────────────────────────┘
```

---

## Data Flow: Single Query Execution

### Baseline Mode (No Defense)

```
1. User query: "What is a firewall?"
       │
2. ┌───▼──────────────────────────────────────────────────┐
   │ DocumentIndexer.load_documents("kb_poisoned/")        │
   │   → Reads all .txt/.json files                        │
   │   → Splits by double-newline into Documents           │
   │   → Result: 70 documents (20 clean + 50 poisoned)     │
   └───┬──────────────────────────────────────────────────┘
       │
3. ┌───▼──────────────────────────────────────────────────┐
   │ DocumentIndexer.chunk_documents(docs)                  │
   │   → 256-char chunks with 32-char overlap               │
   │   → Result: 105 Chunk objects                          │
   └───┬──────────────────────────────────────────────────┘
       │
4. ┌───▼──────────────────────────────────────────────────┐
   │ DocumentIndexer.build_dense_index(chunks)              │
   │   → SentenceTransformer("all-MiniLM-L6-v2")           │
   │   → Encode 105 chunks → (105, 384) float32 matrix     │
   │   → L2 normalize → FAISS IndexFlatIP                  │
   │                                                        │
   │ DocumentIndexer.build_sparse_index(chunks)             │
   │   → Tokenize by whitespace, lowercase                  │
   │   → BM25Okapi index                                    │
   └───┬──────────────────────────────────────────────────┘
       │
5. ┌───▼──────────────────────────────────────────────────┐
   │ Retriever.retrieve("What is a firewall?", top_k=5)    │
   │                                                        │
   │   Dense mode:                                          │
   │     query_emb = encode("What is a firewall?") → (1,384)│
   │     scores, indices = faiss.search(query_emb, k=5)    │
   │     → Poison doc "Regarding firewall: INJECTED_ANSWER" │
   │       gets score 0.92 (optimized for this query)       │
   │     → Clean doc "A firewall is a device..." gets 0.85  │
   │     → Poison doc ranked #1                             │
   │                                                        │
   │   Result: [RetrievalResult(poison, 0.92, rank=1),     │
   │            RetrievalResult(clean, 0.85, rank=2), ...]  │
   └───┬──────────────────────────────────────────────────┘
       │
6. ┌───▼──────────────────────────────────────────────────┐
   │ Generator.generate(query, retrieved_chunks)            │
   │                                                        │
   │   Extractive mode:                                     │
   │     answer = chunks[0].text  ← poison doc text         │
   │     → "Regarding firewall: INJECTED_ANSWER..."         │
   │                                                        │
   │   GGUF mode:                                           │
   │     prompt = "<|system|>Answer using ONLY context..."   │
   │     answer = TinyLlama(prompt, max_tokens=256)         │
   │     → "A firewall is a device that monitors..."        │
   │        (LLM paraphrases, doesn't echo verbatim)        │
   └───┬──────────────────────────────────────────────────┘
       │
7. ┌───▼──────────────────────────────────────────────────┐
   │ PipelineOutput                                         │
   │   answer: "Regarding firewall: INJECTED_ANSWER..."     │
   │   retrieved_chunks: [poison_doc, clean_doc, ...]       │
   │   retrieval_latency_ms: 10.5                           │
   │   generation_latency_ms: 0.2 (extractive) / 4000 (GGUF)│
   │   active_kb: "kb_poisoned/"                            │
   │   active_retriever: "dense"                            │
   └──────────────────────────────────────────────────────┘
```

### MTD Mode (With Defense)

```
1. User query: "What is a firewall?"
       │
2. ┌───▼──────────────────────────────────────────────────┐
   │ pipeline._rebuild_retriever_if_needed()                │
   │   mtd_cfg = controller.get_active_config()             │
   │   → {"kb": "kb_clean/", "retriever": "sparse",        │
   │      "embedder": "all-MiniLM-L6-v2"}                   │
   │                                                        │
   │   if kb changed from last query:                       │
   │     → Reload documents from new KB                     │
   │     → Rebuild FAISS + BM25 indexes                     │
   │     → Rebuild Retriever with new strategy              │
   └───┬──────────────────────────────────────────────────┘
       │
3. ┌───▼──────────────────────────────────────────────────┐
   │ Retriever.retrieve(query, top_k=5)                     │
   │   Now using SPARSE (BM25) on kb_clean/                 │
   │   → No poison docs in this KB                          │
   │   → Returns clean documents                            │
   │   → ASR = 0% for this query                            │
   └───┬──────────────────────────────────────────────────┘
       │
4. ┌───▼──────────────────────────────────────────────────┐
   │ Generator produces answer from clean context           │
   └───┬──────────────────────────────────────────────────┘
       │
5. ┌───▼──────────────────────────────────────────────────┐
   │ MTDController.step(query, retrieval_scores)            │
   │                                                        │
   │   5a. Update anomaly score                             │
   │       z_score of current scores vs rolling window      │
   │       → anomaly_score = 0.12 (normal)                  │
   │                                                        │
   │   5b. Check KB rotation                                │
   │       query_count=3, interval=10 → not yet             │
   │                                                        │
   │   5c. Step retriever switcher                          │
   │       batch_count=3, batch_size=5 → stay on "sparse"   │
   │                                                        │
   │   5d. No anomaly → no force-switch                     │
   │                                                        │
   │   5e. query_counter=3, embed_interval=10 → no rotation │
   │                                                        │
   │   State: MONITORING                                    │
   └──────────────────────────────────────────────────────┘
```

---

## Class Hierarchy

### RAG Pipeline

```
DocumentIndexer
  ├── load_documents(kb_path) → list[Document]
  ├── chunk_documents(docs) → list[Chunk]
  ├── build_dense_index(chunks) → faiss.IndexFlatIP
  ├── build_sparse_index(chunks) → BM25Okapi
  ├── save_index() / load_index()
  └── .encoder (lazy SentenceTransformer)

Retriever
  ├── retrieve(query, top_k) → list[RetrievalResult]
  ├── _dense_retrieve()   ← FAISS cosine search
  ├── _sparse_retrieve()  ← BM25 keyword scoring
  └── _hybrid_retrieve()  ← alpha-weighted combination

Generator
  ├── generate(query, chunks) → GeneratorOutput
  ├── _generate_extractive()  ← echoes top chunk (worst-case model)
  ├── _generate_llama_cpp()   ← real LLM inference via GGUF
  └── _build_context()        ← joins chunks with --- separator

RAGPipeline
  ├── run(query) → PipelineOutput
  ├── run_batch(queries) → list[PipelineOutput]
  ├── _build_index(kb_path)
  └── _rebuild_retriever_if_needed()  ← MTD integration point
```

### MTD Engine

```
KBRotator
  ├── create_initial_state() → KBPoolState
  ├── should_rotate(state) → bool
  ├── rotate(state) → KBPoolState
  ├── update_anomaly_score(state, scores) → KBPoolState
  └── get_active_kb_path(state) → str

RetrieverSwitcher
  ├── get_current_strategy() → str
  ├── step() → str
  └── force_switch(target) → None

EmbedRotator
  ├── create_initial_state() → EmbedModelState
  ├── get_active_model(state) → SentenceTransformer
  ├── rotate_model(state) → EmbedModelState
  ├── encode_single(texts, state) → ndarray
  └── encode_ensemble(texts, state) → ndarray

MTDController
  ├── step(query, scores) → MTDStatus
  ├── get_status() → MTDStatus
  ├── get_active_config() → dict
  └── _transition(new_state) → None
```

### Attacker Modules

```
PoisonedRAGAttacker           ← AML.T0054
  ├── craft_poison_docs()     naive or optimized strategy
  ├── inject_into_kb()        copy KB + write poison .txt/.json
  └── evaluate_injection()    compute ASR

CorpusPoisonAttacker          ← AML.T0020
  ├── craft_adversarial_passage()  L1 (black-box) or L2 (gray-box)
  ├── inject_at_corpus_level()     write into corpus copy
  └── compute_injection_stats()    report metrics

IndirectInjectionAttacker     ← AML.T0051
  ├── craft_injection_document()   override / role_play / data_exfil
  ├── inject_via_retrieval()       boost similarity for retrievability
  └── evaluate_injection_success() ASR + instruction_followed rate
```

---

## Config-Driven Design

Every runtime parameter comes from YAML config files loaded via `safe_load_config()`:

```
config/rag_config.yaml     ──→  indexer, retriever, generator, pipeline
config/mtd_config.yaml     ──→  kb_rotator, retriever_switcher, embed_rotator, mtd_controller
config/attack_config.yaml  ──→  poisoned_rag, corpus_poison, indirect_injection, evaluator
```

**No magic numbers in source code.** Every threshold, path, model name, interval, and parameter is read from config at `__init__` time and stored as `self.*` attributes.

The `safe_load_config()` function in `src/utils.py` provides:
- File existence check before reading
- Empty file detection
- Single entry point for all YAML loading (replaces 10+ raw `yaml.safe_load()` calls)

---

## Benchmark Adapter Architecture

```
External Benchmark Data (JSON)
        │
        ▼  (if file exists)
┌─────────────────────┐
│  RAGSecBenchAdapter  │──→ 39 BenchmarkQuery objects (13 categories × 3 domains)
│  SafeRAGAdapter      │──→ 20 BenchmarkQuery objects (4 dimensions × 5 queries)
│  RAGCheckerAdapter   │──→ FaithfulnessResult (claim-level evaluation)
└─────────────────────┘
        │
        ▼  (if file not found)
┌─────────────────────┐
│  Synthetic Fallback  │──→ Same format, generated from templates matching
│                      │    the benchmark's methodology and attack categories
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Extended Evaluator   │──→ Runs pipeline on all queries
│                      │──→ Computes ASR, NARR, Defense Coverage, Faithfulness
│                      │──→ Saves JSON results + prints comparison table
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Comparative Table    │──→ Table II: our results vs 6 baseline frameworks
│                      │──→ ASCII (terminal) + LaTeX (paper)
└─────────────────────┘
```

---

## Key Design Decisions

### Why Extractive + GGUF dual mode?

Extractive mode gives **worst-case ASR** (100%) — it models the PoisonedRAG assumption that the LLM faithfully echoes retrieved content. This is the standard for attack evaluation papers. GGUF mode gives **realistic ASR** (0-20%) — showing what happens with a real LLM. Both are needed for the paper: extractive proves the attack works, GGUF proves it matters in practice.

### Why round-robin rotation instead of random?

Round-robin is deterministic and reproducible (`seed=42`). For the paper's experiments, we need exact reproducibility. Random and adaptive policies are implemented for production use but round-robin is the default for experiments.

### Why synthetic benchmark data?

The real RAGSecBench/SafeRAG/RAGChecker datasets require downloading from their repos. The synthetic fallback generates queries following the same methodology (same attack categories, same evaluation dimensions) using our KB domain. When real data is available, the adapters auto-switch.

### Why no certified robustness?

Certified defenses like RobustRAG provide mathematical guarantees but are static. MTD's value is dynamic adaptation — the defense surface changes per query. Adding formal guarantees for a dynamic system is an open problem and acknowledged as future work.
