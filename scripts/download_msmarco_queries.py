#!/usr/bin/env python3
"""
download_msmarco_queries.py
- Queries: Hugging Face (microsoft/ms_marco v1.1, validation split)
- Qrels:   Anserini GitHub raw (dev subset)
Writes:
  data/queries.dev.tsv
  data/qrels.dev.small.txt
"""
import os, urllib.request
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

print("[INFO] Loading MS MARCO v1.1 (validation/dev) from Hugging Face...")
ds = load_dataset("microsoft/ms_marco", "v1.1")
dev = ds["validation"]

with open("data/queries.dev.tsv", "w", encoding="utf-8") as f:
    for row in dev:
        qid = row["query_id"]
        q = row["query"].strip()
        f.write(f"{qid}\t{q}\n")
print("[OK] Wrote data/queries.dev.tsv")

print("[INFO] Downloading qrels (dev subset) from Anserini...")
qrels_url = ("https://raw.githubusercontent.com/castorini/anserini/main/"
             "src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt")
urllib.request.urlretrieve(qrels_url, "data/qrels.dev.small.txt")
print("[OK] Wrote data/qrels.dev.small.txt")
