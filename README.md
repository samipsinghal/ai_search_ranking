# ai_search_ranking
Small inverted-index search engine using Python  
Implements parser → run merger → BM25 query processor.


#  BM25 Search Engine – CS-GY 6913 (Web Search Engines)

This repo contains my implementation of a small-scale search system built for **CS-GY 6913 (Web Search Engines)** at NYU , Fall 2025.  
The goal of the assignment was to design and implement an end-to-end retrieval engine—from raw text to ranked results—using an inverted index and the BM25 ranking function.

---

##  Overview  

This project implements a full end-to-end **retrieval system**:  
from text preprocessing and inverted-index construction to ranking, evaluation, and efficiency reporting.

It follows the classical **Information Retrieval pipeline** but in modern Pythonic form — readable, modular, and measurable.

collection.tsv → index_build → index_merge → query_bm25 → evaluation


---

## Architecture  

**Core pipeline**
| Stage | Script | Purpose |
|-------|---------|----------|
| 1 Index Build | `src/index_build.py` | Parses the corpus, tokenizes text, and writes sorted partial runs. |
| 2 Index Merge | `src/index_merge.py` | Merges all runs into a compressed inverted index (VarByte encoding). |
| 3 Query Engine | `src/query_bm25.py` | Runs interactive BM25 search with snippet and text display. |
| 4 Batch Search | `scripts/search_to_run.py` | Runs all queries in bulk and saves TREC-style run files. |
| 5 Effectiveness | `scripts/report_effectiveness.py` | Evaluates MRR, nDCG, Recall using `pytrec_eval`. |
| 6 Efficiency | `scripts/report_efficiency.py` | Reports latency, index size, and throughput statistics. |
| 7 Query Download | `scripts/download_msmarco_queries.py` | Fetches MS MARCO v1.1 dev/test query sets and qrels. |

**Libraries used:**  
Python 3.12, `pandas`, `pytrec_eval`, `datasets`, and standard library modules only.


## Implementation Optimizations

This section summarizes the design choices and micro-optimizations that improve **latency**, **I/O**, and **space usage** in the system. References point to the relevant source files.

### 1) Tokenization & Parsing
- **Lowercase alphanumeric tokenizer** (`src/common.py`): uses a precompiled regex `[A-Za-z0-9]+` and an HTML stripper that ignores `<script>/<style>`.  
  *Why:* stable, fast, and aligned with typical MS MARCO “bag of words” preprocessing; avoids unicode corner cases without crashing.

### 2) I/O-Efficient Index Construction (External Merge)
- **Run generation** (`src/index_build.py`): processes the corpus in configurable batches (`--batch_docs`), spills sorted triples `(term, docid, tf)` to disk as `run_*.tsv`.  
  *Why:* prevents unbounded memory growth on large corpora and keeps per-spill sort cheap.
- **K-way merge** (`src/index_merge.py`): streams all runs with heap-merged `(term, docid)` order via lightweight `Cursor`s; never materializes full postings in memory.  
  *Why:* minimizes peak RSS while preserving sequential disk access patterns.

### 3) Compression & On-Disk Layout
- **Delta + VarByte** (`src/varbyte.py`, `src/index_merge.py`):  
  - DocIDs are **delta-encoded** and then **variable-byte** compressed.  
  - Term frequencies are **VarByte** compressed.  
  - Per term block layout (little-endian):
    ```
    [df:uint32]
    [docids_len:uint32][docid_deltas_varbyte...]
    [tfs_len:uint32]   [tf_varbyte...]
    ```
  *Why:* delta coding yields 50–80% fewer bytes for monotonically increasing docIDs; VarByte is simple, CPU-light, and fast to decode.

- **Separated streams** (docIDs and TFs stored in separate contiguous slices).  
  *Why:* enables decoding only what is needed and better cache locality during DAAT traversal.

- **Binary metadata file (`lexicon.tsv`)**: stores `term, df, offset, length`.  
  *Why:* allows direct `seek(offset)` + `read(length)` per term with no scanning.

- **Doc length sidecar (`doclen.bin`)**: contiguous `uint32` array, little-endian.  
  *Why:* O(1) random access by docid during BM25 scoring with minimal RAM.

**Result on 100k MS MARCO sample (for reference):**


postings.bin ≈ 9.89 MB
lexicon.tsv ≈ 2.16 MB
doclen.bin ≈ 0.38 MB
TOTAL ≈ 12.43 MB



### 4) Query-Time Latency & I/O
- **On-demand decoding** (`src/query_bm25.py::read_term_postings`):  
  Performs `seek+read(length)` for exactly one term block, decodes minimally, and delta-expands in a tight loop.  
  *Why:* avoids preloading or decompressing entire lists; reduces cache misses and page faults.

- **DAAT scoring** (`score_disjunctive` / `score_conjunctive`):  
  Uses a min-heap over list heads; advances only contributing iterators.  
  *Why:* minimizes passes over large lists; benefits from already-sorted docIDs.

- **Lightweight counters**: per-query **seeks** and **bytes read** are recorded; `scripts/report_efficiency.py` reports **mean / p50 / p90 / p95 / p99** latency and approximate throughput.  
  *Why:* ties latency to I/O cost for principled tuning.

**Observed single-term decode on this corpus:**


mean ~0.00 ms (p95 ~0.01 ms); warm throughput ~360k decodes/sec

(Real multi-term query latency depends on term df and posting lengths.)

### 5) Data Types & Endianness
- **Doc lengths:** `uint32` little-endian in `doclen.bin`.  
  *Why:* compact, fixed-width, direct indexing with `struct.unpack('<I')`.
- **Block headers (`df`, lengths):** `uint32` little-endian.  
  *Why:* predictable structure and platform-agnostic serialization.
- **Offsets/lengths in `lexicon.tsv`:** integers (bytes).  
  *Why:* direct addressing without pointer chasing.

### 6) BM25 Details
- **Parameters:** default `k1=0.9`, `b=0.4` (interactive), tunable via CLI; batch evaluation often uses `k1≈1.2`, `b≈0.75` (MS MARCO baseline friendly).  
  *Why:* exposes the classic efficiency–effectiveness trade-off while keeping code paths identical.

### 7) Practical I/O Tricks
- **Batch size control** during run generation (`--batch_docs`) to balance sort cost vs spill frequency.  
- **Contiguous file layout** for postings to favor sequential reads when multiple term blocks are adjacent.  
- **No gzip/bzip** of entire files (explicitly avoided): compression happens at the postings-block level so seeking stays O(1).

### 8) Evaluation Depth & Output Size
- **`--topk`** in `scripts/search_to_run.py` controls results per query:  
  - `--topk 100` for fast iteration and `MRR@10`/`nDCG@10`.  
  - `--topk 1000` for `Recall@1000` and full baselines.  
  *Why:* shallow runs are faster and smaller; deeper runs enable recall-oriented metrics.

### 9) Typical Baseline (MS MARCO Dev)
With correct tokenization and ID alignment:
- **MRR@10** ≈ 0.19–0.21  
- **nDCG@10** ≈ 0.22–0.24  
- **Recall@1000** ≈ 0.85–0.90

`Use:`

```` ```bash ````

PYTHONPATH=. python -u scripts/search_to_run.py \
  --index_dir index/final \
  --queries data/queries.dev.tsv \
  --run_out runs/dev_run.trec \
  --k1 1.2 --b 0.75 --mode disj --topk 1000

PYTHONPATH=. python scripts/report_effectiveness.py \
  --qrels data/qrels.dev.small.txt \
  --run runs/dev_run.trec

```` ``` ````

10) Future Work (drop-in ideas)
Block-max WAND or MaxScore to skip non-competitive documents.
Static cache for head segments of frequent terms.
PEF/SIMD-BP128 for faster integer decoding.
Memory-mapped postings for OS-level readahead on repeated workloads.


---

If you want, I can tailor that section with your exact measured p50/p95/p99 latencies from `scripts/report_efficiency.py` and add a small before/after table.






---

## Directory Layout

```text

ai_search_ranking/
├── data/
│ ├── collection.sample100000.tsv
│ ├── page_table.tsv
│ ├── queries.dev.tsv
│ └── qrels.dev.small.txt
│
├── index/
│ ├── tmp/ # intermediate run_*.tsv
│ └── final/ # merged + compressed index
│ ├── postings.bin
│ ├── lexicon.tsv
│ └── doclen.bin
│
├── scripts/
│ ├── download_data.py
│ ├── download_msmarco_queries.py
│ ├── report_efficiency.py
│ ├── search_to_run.py
│ └── report_effectiveness.py
│
├── src/
│ ├── common.py
│ ├── index_build.py
│ ├── index_merge.py
│ ├── query_bm25.py
│ ├── llm_snippets.py
│ └── varbyte.py
│
└── tests/
└── test_tokenize.py





---

##  Setup

```bash
git clone https://github.com/<your-username>/ai_search_ranking.git
cd ai_search_ranking

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

If you want to use the optional LLM-snippet feature:

export OPENAI_API_KEY=sk-<your-key>

## Data
You can either download MS MARCO data automatically or provide your own TSVs.

1. Download MS MARCO subset

python -m scripts.download_msmarco_queries


This creates:

data/queries.dev.tsv
data/qrels.dev.small.txt



## Building the Index

# Step 1: Build intermediate runs
python -m src.index_build \
  --input data/collection.sample100000.tsv \
  --outdir index/tmp \
  --batch_docs 50000

# Step 2: Merge runs and compress
python -m src.index_merge \
  --tmpdir index/tmp \
  --outdir index/final

You’ll get:

index/final/
 ├─ postings.bin   # compressed postings
 ├─ lexicon.tsv    # term→offset,length,df
 └─ doclen.bin     # doc length info


## Interactive Search (BM25)

python -m src.query_bm25 \
  --index_dir index/final \
  --collection data/collection.sample100000.tsv \
  --page_table data/page_table.tsv \
  --snippet --show_text


Example:

credit score
 1. doc=36513  score=17.58
    └─ snippet: Both the Equifax **Credit** **Score** and the FICO **Score** are…
    └─ text:    Both the Equifax Credit Score and the FICO Score are general-purpose models…


## Batch Evaluation Workflow

Step 1: Generate a run file (batch search)

PYTHONPATH=. python -u scripts/search_to_run.py \
  --index_dir index/final \
  --queries data/queries.dev.tsv \
  --run_out runs/dev_run.trec \
  --k1 1.5 --b 0.75 --mode disj --topk 1000

(--topk = results per query; use 100 for fast debug, 1000 for full recall)

Step 2: Evaluate effectiveness

PYTHONPATH=. python scripts/report_effectiveness.py \
  --qrels data/qrels.dev.small.txt \
  --run runs/dev_run.trec


Dataset: MS MARCO Passage Ranking.
Code: Python 3.12.
=======
Output example:

MRR@10     : 0.192
nDCG@10    : 0.276
Recall@100 : 0.421
Recall@1000: 0.656


#Efficiency Report

Measure index size and latency:

python -m scripts.report_efficiency

Example output:

Documents indexed : 100,000
Unique terms      : 104,319
Average doc length: 55.0 tokens
Index size (MB)   : 12.4
Single-term decode latency : mean=0.00 ms  p95=0.01 ms
Throughput ≈ 360 k decodes/sec

Baselines

| Model                           |    MRR@10 | Notes                        |
| ------------------------------- | --------: | ---------------------------- |
| BM25 (this repo, k1=0.9 b=0.4)  |    ≈ 0.19 | 100 K MS MARCO subset        |
| MS MARCO Official BM25 Baseline | 0.19–0.21 | Full collection (8.8 M docs) |
| Dense (ANCE / MiniLM)           | 0.33–0.38 | For comparison only          |


#Notes & Tips

topk controls how many results per query are written; use 100 for speed, 1000 for full evaluation.
Evaluation uses pytrec_eval (TREC style metrics).
Efficiency and effectiveness can be run independently.
LLM Snippets (src/llm_snippets.py) can generate contextual summaries for each top result.

#Acknowledgments

Course: CS-GY 6913 – Web Search Engines (Prof. Torsten Suel, NYU , Fall 2025)
Dataset: MS MARCO Passage Ranking v1.1
Implementation: Developed by Samip Singhal as part of the retrieval and ranking module.
