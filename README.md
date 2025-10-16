<<<<<<< HEAD
# ai_search_ranking
Small inverted-index search engine using Python  
Implements parser → run merger → BM25 query processor.


#  BM25 Search Engine – CS-GY 6913 (Web Search Engines)

This repo contains my implementation of a small-scale search system built for **CS-GY 6913 (Web Search Engines)** at NYU , Fall 2025.  
The goal of the assignment was to design and implement an end-to-end retrieval engine—from raw text to ranked results—using an inverted index and the BM25 ranking function.
=======
# AI Search Ranking  
*A compact BM25-based search engine built from scratch for CS-GY 6913 (Web Search Engines, NYU Tandon)*
>>>>>>> f12d062 (update readme and efficiency report with baselines)

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

---

##  Directory Layout

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

Data
You can either download MS MARCO data automatically or provide your own TSVs.

1. Download MS MARCO subset

python -m scripts.download_msmarco_queries


This creates:

data/queries.dev.tsv
data/qrels.dev.small.txt



Building the Index

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


Interactive Search (BM25)

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


Batch Evaluation Workflow

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


<<<<<<< HEAD
Acknowledgements
Built for CS-GY 6913 – Web Search Engines (Prof. Torsten Suel, NYU Tandon, Fall 2025).
Dataset: MS MARCO Passage Ranking.
Code: Python 3.12.
=======
Output example:

MRR@10     : 0.192
nDCG@10    : 0.276
Recall@100 : 0.421
Recall@1000: 0.656


Efficiency Report

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


Notes & Tips

topk controls how many results per query are written; use 100 for speed, 1000 for full evaluation.
Evaluation uses pytrec_eval (TREC style metrics).
Efficiency and effectiveness can be run independently.
LLM Snippets (src/llm_snippets.py) can generate contextual summaries for each top result.

Acknowledgments

Course: CS-GY 6913 – Web Search Engines (Prof. Torsten Suel, NYU , Fall 2025)
Dataset: MS MARCO Passage Ranking v1.1
Implementation: Developed by Samip Singhal as part of the retrieval and ranking module.
>>>>>>> f12d062 (update readme and efficiency report with baselines)
