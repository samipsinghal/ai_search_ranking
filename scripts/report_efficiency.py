#!/usr/bin/env python3
"""
report_efficiency.py
Generates an efficiency summary for the BM25 search system:
- index sizes
- corpus stats
- single-term decode latency (cold-ish vs warm)
- optional multi-term DAAT latency on a query set
- per-query I/O (seeks, bytes read) via query_bm25 counters
"""

import os, time, argparse, random, csv
from statistics import mean
from typing import List, Tuple

from src.query_bm25 import (
    load_lexicon, load_doclens, read_term_postings,
    BM25, score_disjunctive, score_conjunctive
)

INDEX_DIR_DEFAULT = "index/final"

def mb(path): return os.path.getsize(path) / (1024*1024)

def percentiles(xs: List[float], ps=(50,90,95,99)) -> dict:
    if not xs:
        return {p: 0.0 for p in ps}
    xs_sorted = sorted(xs)
    out = {}
    n = len(xs_sorted)
    for p in ps:
        # nearest-rank with interpolation
        k = (p/100)*(n-1)
        f = int(k)
        c = min(f+1, n-1)
        frac = k - f
        out[p] = xs_sorted[f]*(1-frac) + xs_sorted[c]*frac
    return out

def report_index_stats(index_dir: str):
    lexicon = load_lexicon(os.path.join(index_dir, "lexicon.tsv"))
    doclens = load_doclens(os.path.join(index_dir, "doclen.bin"))
    N = len(doclens)
    avgdl = (sum(doclens) / N) if N else 0.0
    return lexicon, doclens, N, avgdl, len(lexicon)

def measure_single_term_latency(index_dir: str, lexicon, trials=100, seed=42):
    """
    'Cold-ish': open/close the file per term (forces new FD & seeks).
    'Warm': reuse one file handle and read same terms again.
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
            cold_ms.append((time.perf_counter() - t0)*1000)

    # Warm
    with open(os.path.join(index_dir, "postings.bin"), "rb") as fp:
        for t in terms:
            off, ln, _ = lexicon[t]
            t0 = time.perf_counter()
            _ = read_term_postings(fp, off, ln)
            warm_ms.append((time.perf_counter() - t0)*1000)

    return terms, cold_ms, warm_ms

def measure_query_set(index_dir: str, queries_tsv: str, mode: str, bm25: BM25, lexicon, topk=10):
    """
    Runs a real query set (qid<TAB>text) through DAAT and returns per-query stats.
    Uses the same read_term_postings path as the REPL, so seeks/bytes are realistic.
    """
    
    if not os.path.exists(queries_tsv):
        print(f"[WARN] queries file not found: {queries_tsv} (skipping query-set timing)")
        return []
    
    from src.query_bm25 import IO_BYTES_READ, IO_SEEKS  # globals
    results = []  # (qid, tokens, latency_ms, seeks, kb, num_terms, num_postings, topk)

    with open(queries_tsv, "r", encoding="utf-8") as qf, \
         open(os.path.join(index_dir, "postings.bin"), "rb") as pf:

        for line in qf:
            if not line.strip() or "\t" not in line:
                continue
            qid, text = line.strip().split("\t", 1)
            terms = [t for t in text.lower().split() if t]
            term_postings: List[Tuple[int, List[int], List[int], float]] = []

            # reset counters
            import src.query_bm25 as qb
            qb.IO_BYTES_READ = 0
            qb.IO_SEEKS = 0

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
                continue

            t0 = time.perf_counter()
            if mode == "disj":
                scores = score_disjunctive(bm25, term_postings)
            else:
                scores = score_conjunctive(bm25, term_postings)
            latency_ms = (time.perf_counter() - t0)*1000

            # cheap size proxies
            num_postings = sum(len(d) for _, d, *_ in term_postings)

            results.append((
                qid, " ".join(terms),
                latency_ms, qb.IO_SEEKS, qb.IO_BYTES_READ/1024.0,
                len(terms), num_postings, topk
            ))
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default=INDEX_DIR_DEFAULT)
    ap.add_argument("--trials", type=int, default=100, help="single-term trials")
    ap.add_argument("--queries", default="", help="optional query set TSV (qid<TAB>text)")
    ap.add_argument("--mode", choices=["disj","conj"], default="disj")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--csv_out", default="", help="optional CSV path for per-query metrics")
    args = ap.parse_args()

    p = os.path.join
    postings_size = mb(p(args.index_dir, "postings.bin"))
    lexicon_size  = mb(p(args.index_dir, "lexicon.tsv"))
    doclen_size   = mb(p(args.index_dir, "doclen.bin"))
    total_size    = postings_size + lexicon_size + doclen_size

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
    terms, cold_ms, warm_ms = measure_single_term_latency(args.index_dir, lexicon, trials=args.trials)
    cold_p = percentiles(cold_ms); warm_p = percentiles(warm_ms)

    print(f"Single-term decode latency over {len(terms)} random terms:")
    print(f"  COLD-ish  mean={mean(cold_ms):.2f} ms  p50={cold_p[50]:.2f}  p90={cold_p[90]:.2f}  p95={cold_p[95]:.2f}  p99={cold_p[99]:.2f}")
    print(f"  WARM      mean={mean(warm_ms):.2f} ms  p50={warm_p[50]:.2f}  p90={warm_p[90]:.2f}  p95={warm_p[95]:.2f}  p99={warm_p[99]:.2f}")
    if warm_ms:
        print(f"  ~Throughput (warm, 1t): {1000.0/mean(warm_ms):.1f} decodes/sec\n")
    else:
        print()

    # Optional: real query set timing + CSV
    if args.queries:
        bm25 = BM25(doclens, k1=args.k1, b=args.b)
        rows = measure_query_set(args.index_dir, args.queries, args.mode, bm25, lexicon, topk=10)
        if rows:
            lats = [r[2] for r in rows]
            seeks = [r[3] for r in rows]
            kb = [r[4] for r in rows]
            lp = percentiles(lats)
            print(f"{args.mode.upper()} query latency on {len(rows)} queries ({args.queries}):")
            print(f"  mean={mean(lats):.2f} ms  p50={lp[50]:.2f}  p90={lp[90]:.2f}  p95={lp[95]:.2f}  p99={lp[99]:.2f}")
            print(f"  I/O per query (mean): seeks={mean(seeks):.1f}, bytes={mean(kb)*1024:.0f}")
            print()

            if args.csv_out:
                os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
                with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["qid","tokens","latency_ms","seeks","kb_read","num_terms","num_postings","topk"])
                    w.writerows(rows)
                print(f"[OK] wrote per-query CSV -> {args.csv_out}")

    print(" Efficiency report complete.")

if __name__ == "__main__":
    main()
