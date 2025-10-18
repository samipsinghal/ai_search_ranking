#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_efficiency.py  —  Efficiency & latency report for your BM25 system.

What it reports:
- Corpus stats (N, unique terms, avgdl)
- Index sizes (postings/lexicon/doclen)
- Single-term decode latency (cold-ish vs warm)
- Optional query-set timing (DAAT): mean/min/max + p50/p90/p95/p99
- Optional per-query CSV export

Assumptions:
- The repo layout has:
    index/final/{postings.bin,lexicon.tsv,doclen.bin}
- src.query_bm25 exports:
    load_lexicon(path), load_doclens(path), read_term_postings(fp, off, ln)
    BM25 class, score_disjunctive(bm25, term_postings), score_conjunctive(...)
    IO_BYTES_READ, IO_SEEKS (globals) that the read path updates

Usage examples:
  # basic (prints to terminal, no files)
  python -m scripts.report_efficiency --index_dir index/final --trials 200

  # with a query set (qid<TAB>text), sample 10 queries, show stats
  python -m scripts.report_efficiency \
    --index_dir index/final \
    --queries data/queries.dev.small.tsv \
    --sample_n 10 \
    --mode disj --k1 0.9 --b 0.4
"""
from __future__ import annotations

import os
import csv
import time
import math
import argparse
import random
from typing import List, Tuple, Dict, Iterable
from statistics import mean

# ---- Imports from your search system ----------------------------------------
from src.query_bm25 import (
    load_lexicon, load_doclens, read_term_postings,
    BM25, score_disjunctive, score_conjunctive
)

# Optional globals (if present in your module)
try:
    import src.query_bm25 as qbmod
    HAS_IO_COUNTERS = hasattr(qbmod, "IO_BYTES_READ") and hasattr(qbmod, "IO_SEEKS")
except Exception:
    qbmod = None
    HAS_IO_COUNTERS = False

INDEX_DIR_DEFAULT = "index/final"


# ---- Utilities ---------------------------------------------------------------
def mb(path: str) -> float:
    return os.path.getsize(path) / (1024.0 * 1024.0)


def percentiles(xs: List[float], ps=(50, 90, 95, 99)) -> Dict[int, float]:
    if not xs:
        return {p: 0.0 for p in ps}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    out = {}
    for p in ps:
        k = (p / 100.0) * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        frac = k - f
        out[p] = xs_sorted[f] * (1 - frac) + xs_sorted[c] * frac
    return out


def report_index_stats(index_dir: str):
    lexicon = load_lexicon(os.path.join(index_dir, "lexicon.tsv"))
    doclens = load_doclens(os.path.join(index_dir, "doclen.bin"))
    N = len(doclens)
    avgdl = (sum(doclens) / N) if N else 0.0
    num_terms = len(lexicon)
    return lexicon, doclens, N, avgdl, num_terms


def measure_single_term_latency(index_dir: str,
                                lexicon: Dict[str, Tuple[int, int, int]],
                                trials: int = 100,
                                seed: int = 42):
    """
    'Cold-ish': open/close file each time (new FD & seeks).
    'Warm'    : reuse one file handle for all reads.
    """
    random.seed(seed)
    terms = random.sample(list(lexicon.keys()), min(trials, len(lexicon)))
    cold_ms, warm_ms = [], []

    # Cold-ish
    for t in terms:
        off, ln, _ = lexicon[t]
        with open(os.path.join(index_dir, "postings.bin"), "rb") as fp:
            t0 = time.perf_counter()
            _ = read_term_postings(fp, off, ln)
            cold_ms.append((time.perf_counter() - t0) * 1000.0)

    # Warm
    with open(os.path.join(index_dir, "postings.bin"), "rb") as fp:
        for t in terms:
            off, ln, _ = lexicon[t]
            t0 = time.perf_counter()
            _ = read_term_postings(fp, off, ln)
            warm_ms.append((time.perf_counter() - t0) * 1000.0)

    return terms, cold_ms, warm_ms


def parse_queries_tsv(path: str) -> List[Tuple[str, str]]:
    """Reads qid<TAB>text lines; ignores bad/empty lines."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or "\t" not in line:
                continue
            qid, text = line.strip().split("\t", 1)
            if qid and text:
                out.append((qid, text))
    return out


def measure_query_set(index_dir: str,
                      queries_tsv: str,
                      mode: str,
                      bm25: BM25,
                      lexicon: Dict[str, Tuple[int, int, int]],
                      topk: int = 10,
                      sample_n: int = 0,
                      seed: int = 42):
    """
    Returns per-query rows:
      (qid, tokens, latency_ms, seeks, kb_read, num_terms, num_postings, topk)
    """
    if not os.path.exists(queries_tsv):
        print(f"[WARN] queries file not found: {queries_tsv} (skipping query-set timing)")
        return []

    qrows = parse_queries_tsv(queries_tsv)
    if sample_n and sample_n > 0 and len(qrows) > sample_n:
        random.seed(seed)
        qrows = random.sample(qrows, sample_n)

    results = []
    with open(os.path.join(index_dir, "postings.bin"), "rb") as pf:
        for qid, text in qrows:
            terms = [t for t in text.lower().split() if t]
            term_postings: List[Tuple[int, List[int], List[int], float]] = []

            # reset I/O counters if present
            if HAS_IO_COUNTERS:
                qbmod.IO_BYTES_READ = 0
                qbmod.IO_SEEKS = 0

            # collect postings
            for t in terms:
                meta = lexicon.get(t)
                if not meta:
                    continue
                off, length, df = meta
                docs, tfs = read_term_postings(pf, off, length)
                if docs:
                    term_postings.append((df, docs, tfs, bm25.idf(df)))

            if not term_postings:
                # no terms found in index; still record empty timings for completeness
                results.append((qid, " ".join(terms), 0.0, 0, 0.0, len(terms), 0, topk))
                continue

            t0 = time.perf_counter()
            if mode == "conj":
                _ = score_conjunctive(bm25, term_postings)
            else:
                _ = score_disjunctive(bm25, term_postings)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Cheap size proxies
            num_postings = sum(len(d) for _, d, *_ in term_postings)
            seeks = qbmod.IO_SEEKS if HAS_IO_COUNTERS else 0
            kb_read = (qbmod.IO_BYTES_READ / 1024.0) if HAS_IO_COUNTERS else 0.0

            results.append((qid, " ".join(terms), latency_ms, seeks, kb_read,
                            len(terms), num_postings, topk))
    return results


def summarize_query_rows(rows: List[Tuple]) -> Dict[str, float]:
    """
    rows: (qid, tokens, latency_ms, seeks, kb_read, num_terms, num_postings, topk)
    """
    if not rows:
        return {}
    lats = [r[2] for r in rows]
    seeks = [r[3] for r in rows]
    kbs = [r[4] for r in rows]
    ps = percentiles(lats)
    return {
        "count": len(rows),
        "lat_mean": mean(lats),
        "lat_min": min(lats),
        "lat_max": max(lats),
        "lat_p50": ps[50],
        "lat_p90": ps[90],
        "lat_p95": ps[95],
        "lat_p99": ps[99],
        "seeks_mean": mean(seeks) if seeks else 0.0,
        "kb_mean": mean(kbs) if kbs else 0.0,
    }


# ---- Main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default=INDEX_DIR_DEFAULT)
    ap.add_argument("--trials", type=int, default=100, help="single-term trials")
    ap.add_argument("--queries", default="", help="optional query set TSV (qid<TAB>text)")
    ap.add_argument("--sample_n", type=int, default=0,
                    help="sample N queries from --queries before timing (0 = use all)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    ap.add_argument("--mode", choices=["disj", "conj"], default="disj")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--csv_out", default="",
                    help="optional CSV path for per-query metrics (TSV format)")
    args = ap.parse_args()

    p = os.path.join
    postings_path = p(args.index_dir, "postings.bin")
    lexicon_path = p(args.index_dir, "lexicon.tsv")
    doclen_path = p(args.index_dir, "doclen.bin")

    # Guard for missing index files
    for path in (postings_path, lexicon_path, doclen_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing index artifact: {path}")

    postings_size = mb(postings_path)
    lexicon_size = mb(lexicon_path)
    doclen_size = mb(doclen_path)
    total_size = postings_size + lexicon_size + doclen_size

    lexicon, doclens, N, avgdl, num_terms = report_index_stats(args.index_dir)

    print("=== BM25 System Efficiency Report ===\n")
    print(f"Documents indexed : {N:,}")
    print(f"Unique terms      : {num_terms:,}")
    print(f"Average doc length: {avgdl:.2f} tokens\n")

    print("Index size (MB):")
    print(f"  postings.bin : {postings_size:.2f}")
    print(f"  lexicon.tsv  : {lexicon_size:.2f}")
    print(f"  doclen.bin   : {doclen_size:.2f}")
    print(f"  TOTAL        : {total_size:.2f}\n")

    # Single-term decode timing
    terms, cold_ms, warm_ms = measure_single_term_latency(
        args.index_dir, lexicon, trials=args.trials, seed=args.seed
    )
    cold_p = percentiles(cold_ms)
    warm_p = percentiles(warm_ms)

    print(f"Single-term decode latency over {len(terms)} random terms:")
    if cold_ms:
        print(f"  COLD-ish  mean={mean(cold_ms):.2f} ms  "
              f"p50={cold_p[50]:.2f}  p90={cold_p[90]:.2f}  "
              f"p95={cold_p[95]:.2f}  p99={cold_p[99]:.2f}")
    if warm_ms:
        print(f"  WARM      mean={mean(warm_ms):.2f} ms  "
              f"p50={warm_p[50]:.2f}  p90={warm_p[90]:.2f}  "
              f"p95={warm_p[95]:.2f}  p99={warm_p[99]:.2f}")
        print(f"  ~Throughput (warm, 1t): {1000.0 / max(mean(warm_ms), 1e-9):.1f} decodes/sec\n")
    else:
        print()

    # Optional: real query set timing + (optional) CSV
    if args.queries:
        bm25 = BM25(doclens, k1=args.k1, b=args.b)
        rows = measure_query_set(
            args.index_dir, args.queries, args.mode, bm25, lexicon,
            topk=10, sample_n=args.sample_n, seed=args.seed
        )
        if rows:
            stats = summarize_query_rows(rows)
            print(f"{args.mode.upper()} latency over {stats['count']} queries:")
            print(f"  mean={stats['lat_mean']:.2f} ms  min={stats['lat_min']:.2f}  max={stats['lat_max']:.2f}")
            print(f"  p50={stats['lat_p50']:.2f}  p90={stats['lat_p90']:.2f}  "
                  f"p95={stats['lat_p95']:.2f}  p99={stats['lat_p99']:.2f}")
            print(f"  I/O per query (means): seeks={stats['seeks_mean']:.1f}, bytes={stats['kb_mean']*1024:.0f}\n")

            if args.csv_out:
                os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
                with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(["qid", "tokens", "latency_ms", "seeks", "kb_read",
                                "num_terms", "num_postings", "topk"])
                    w.writerows(rows)
                print(f"[OK] wrote per-query TSV -> {args.csv_out}")

    print("Efficiency report complete.")


if __name__ == "__main__":
    main()
