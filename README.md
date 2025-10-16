# ai_search_ranking
Small inverted-index search engine using Python  
Implements parser → run merger → BM25 query processor.


#  BM25 Search Engine – CS-GY 6913 (Web Search Engines)

This repo contains my implementation of a small-scale search system built for **CS-GY 6913 (Web Search Engines)** at NYU Tandon, Fall 2025.  
The goal of the assignment was to design and implement an end-to-end retrieval engine—from raw text to ranked results—using an inverted index and the BM25 ranking function.

---

##  What it does

The system crawls (or in this case, loads) passages from the **MS MARCO** dataset, builds an **inverted index**, and lets you run BM25-ranked queries interactively.

It’s split into three main executables:

1. **`index_build.py`** – parses and tokenizes the text, writing sorted intermediate runs  
2. **`index_merge.py`** – merges the runs into a compressed binary index (var-byte encoded)  
3. **`query_bm25.py`** – runs ranked search using BM25 and supports both conjunctive and disjunctive queries  

There’s also optional support for **LLM-generated snippets** so the results look more like what you’d see in a modern search engine.

---

##  Directory layout

ai_search_ranking/
├── data/ # corpus & mappings (MS MARCO sample)
├── index/ # intermediate + final indexes
├── scripts/ # helper scripts
├── src/ # main source code
└── tests/ # a few quick sanity tests


---

##  Setup

```bash
git clone https://github.com/<your-username>/ai_search_ranking.git
cd ai_search_ranking

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


If you plan to use the LLM snippet feature with OpenAI models:
export OPENAI_API_KEY=sk-<your-key>


Data : 

I used a 100 K-document subset of MS MARCO for development and debugging.

python -m scripts.download_data --out data/collection.tsv --sample 100000


data/
 ├─ collection.sample100000.tsv   # docid<TAB>text
 └─ page_table.tsv                # docid<TAB>original_passage_id


Building the index : 

# 1. build intermediate runs
python -m src.index_build \
  --input data/collection.sample100000.tsv \
  --outdir index/tmp \
  --batch_docs 50000

# 2. merge & compress them
python -m src.index_merge \
  --tmpdir index/tmp \
  --outdir index/final


index/final/
 ├─ postings.bin   # compressed postings lists
 ├─ lexicon.tsv    # term → offset,length,df
 └─ doclen.bin     # doc length for BM25 normalization


Interactive BM25 search 

python -m src.query_bm25 \
  --index_dir index/final \
  --collection data/collection.sample100000.tsv \
  --page_table data/page_table.tsv \
  --snippet --show_text


Example output 

credit score
 1. doc=36513  score=17.58
    └─ snippet: Both the Equifax **Credit** **Score** and the FICO **Score** are…
    └─ text:    Both the Equifax Credit Score and the FICO Score are general-purpose models…


Notes
Compression: variable-byte encoding for docIDs and term frequencies
Indexing: external merge sort (I/O-efficient)
Ranking: BM25 with configurable k1 and b
Interface: command-line REPL with optional snippets
Extra credit: optional web interface or snippet generation with an LLM


Acknowledgements
Built for CS-GY 6913 – Web Search Engines (Prof. Torsten Suel, NYU Tandon, Fall 2025).
Dataset: MS MARCO Passage Ranking.
Code: Python 3.12.