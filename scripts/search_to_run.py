#!/usr/bin/env python3
"""
search_to_run.py
----------------
Batch-evaluate BM25 over a query set.
Outputs a TREC-style run file for evaluation (qid Q0 docid rank score tag).
"""

import os, sys, heapq, argparse
from src.query_bm25 import load_lexicon, load_doclens, read_term_postings, BM25

def search_all_queries(index_dir, queries_path, run_out, k1, b, mode, topk=1000):
    lexicon = load_lexicon(os.path.join(index_dir, "lexicon.tsv"))
    doclens = load_doclens(os.path.join(index_dir, "doclen.bin"))
    bm25 = BM25(doclens, k1=k1, b=b)
    postings_path = os.path.join(index_dir, "postings.bin")

    with open(postings_path, "rb") as pf, \
         open(queries_path, "r", encoding="utf-8") as qf, \
         open(run_out, "w", encoding="utf-8") as outf:
        for line in qf:
            if not line.strip():
                continue
            qid, qtext = line.strip().split("\t", 1)
            terms = [t for t in qtext.lower().split() if t]
            term_postings = []
            for t in terms:
                meta = lexicon.get(t)
                if not meta: continue
                off, ln, df = meta
                docs, tfs = read_term_postings(pf, off, ln)
                if docs: term_postings.append((df, docs, tfs, bm25.idf(df)))
            if not term_postings:
                continue
            from src.query_bm25 import score_disjunctive, score_conjunctive
            scores = (score_disjunctive if mode=="disj" else score_conjunctive)(bm25, term_postings)
            topK = heapq.nlargest(topk, scores.items(), key=lambda x: x[1])
            for rank, (doc, score) in enumerate(topK, 1):
                outf.write(f"{qid}\tQ0\t{doc}\t{rank}\t{score:.4f}\tBM25\n")
    print(f"[OK] wrote run file -> {run_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--run_out", required=True)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--mode", choices=["disj","conj"], default="disj")
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()

    search_all_queries(args.index_dir, args.queries, args.run_out, args.k1, args.b, args.mode, args.topk)
