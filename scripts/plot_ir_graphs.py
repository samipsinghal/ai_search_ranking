#!/usr/bin/env python3
"""
plot_ir_graphs.py
Generates explanatory figures for the report:
  1) BM25 tf saturation curve (for a fixed doc length)
  2) BM25 length normalization curve (vary doc length vs avgdl)
  3) IDF vs DF scatter from lexicon.tsv
  4) Zipf-like rank vs DF (log-log) from lexicon.tsv

Usage:
  python scripts/plot_ir_graphs.py --index_dir index/final --k1 0.9 --b 0.4
"""

import os, argparse, math, struct
import matplotlib.pyplot as plt

def load_lexicon(path):
    # lexicon.tsv rows: term \t df \t offset \t length
    dfs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) != 4: continue
            try:
                df = int(parts[1])
            except Exception:
                continue
            dfs.append(df)
    return dfs

def load_doclens(path):
    lens = []
    with open(path, "rb") as f:
        while True:
            b = f.read(4)
            if not b:
                break
            (l,) = struct.unpack("<I", b)
            lens.append(int(l))
    return lens

def idf_bm25(N, df):
    # classic BM25 idf with +0.5 smoothing
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

def tf_weight(tf, dl, avgdl, k1, b):
    denom = tf + k1 * (1 - b + b * (dl / avgdl if avgdl > 0 else 0.0))
    return (tf * (k1 + 1)) / (denom if denom > 0 else 1e-9)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_tf_saturation(avgdl, k1, b, out_dir):
    # Plot score contribution vs tf for several doc length ratios
    tfs = list(range(1, 51))
    ratios = [0.5, 1.0, 2.0]  # |d| / avgdl
    plt.figure()
    for r in ratios:
        dl = r * avgdl
        y = [tf_weight(tf, dl, avgdl, k1, b) for tf in tfs]
        plt.plot(tfs, y, label=f"|d|/avgdl={r:g}")
    plt.xlabel("Term frequency, tf")
    plt.ylabel("BM25 term weight (no IDF)")
    plt.title("BM25 Saturation: tf vs contribution")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    out = os.path.join(out_dir, "bm25_tf_saturation.png")
    plt.savefig(out, bbox_inches="tight", dpi=160)
    plt.close()

def plot_length_normalization(avgdl, k1, b, out_dir):
    # Fix tf and vary |d|/avgdl
    tf = 3
    ratios = [x/10 for x in range(5, 31)]  # 0.5..3.0
    x = ratios
    y = [tf_weight(tf, r*avgdl, avgdl, k1, b) for r in ratios]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("|d| / avgdl")
    plt.ylabel("BM25 term weight (no IDF)")
    plt.title(f"Length Normalization (tf={tf})")
    plt.grid(True, which="both", linestyle=":")
    out = os.path.join(out_dir, "bm25_length_normalization.png")
    plt.savefig(out, bbox_inches="tight", dpi=160)
    plt.close()

def plot_idf_vs_df(dfs, N, out_dir):
    # Scatter plot of IDF vs DF (sample large vocab to keep render reasonable)
    import random
    sample = dfs if len(dfs) <= 200000 else random.sample(dfs, 200000)
    xs = sample
    ys = [idf_bm25(N, df) for df in sample]
    plt.figure()
    plt.scatter(xs, ys, s=1, alpha=0.5)
    plt.xscale("log")
    plt.xlabel("Document frequency (df) [log scale]")
    plt.ylabel("IDF(t)")
    plt.title("IDF vs DF (BM25)")
    plt.grid(True, which="both", linestyle=":")
    out = os.path.join(out_dir, "idf_vs_df.png")
    plt.savefig(out, bbox_inches="tight", dpi=160)
    plt.close()

def plot_zipf_df(dfs, out_dir):
    # Rank-DF plot (Zipf-like) using document frequencies as a proxy
    dfs_sorted = sorted(dfs, reverse=True)
    ranks = range(1, len(dfs_sorted) + 1)
    plt.figure()
    plt.loglog(ranks, dfs_sorted)
    plt.xlabel("Rank (by DF) [log scale]")
    plt.ylabel("Document frequency DF [log scale]")
    plt.title("Zipf-like Rank–Frequency (using DF)")
    plt.grid(True, which="both", linestyle=":")
    out = os.path.join(out_dir, "zipf_rank_df.png")
    plt.savefig(out, bbox_inches="tight", dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, help="index/final directory")
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    args = ap.parse_args()

    out_dir = os.path.join("reports", "figures")
    ensure_dir(out_dir)

    # Load corpus stats
    lex_path = os.path.join(args.index_dir, "lexicon.tsv")
    dl_path  = os.path.join(args.index_dir, "doclen.bin")
    dfs = load_lexicon(lex_path)
    doclens = load_doclens(dl_path)
    N = len(doclens)
    avgdl = (sum(doclens)/N) if N else 0.0

    # 1) BM25 saturation curves (no files needed beyond avgdl)
    plot_tf_saturation(avgdl, args.k1, args.b, out_dir)

    # 2) Length normalization curve
    plot_length_normalization(avgdl, args.k1, args.b, out_dir)

    # 3) IDF vs DF (from lexicon)
    plot_idf_vs_df(dfs, N, out_dir)

    # 4) Zipf-like rank vs DF (from lexicon)
    plot_zipf_df(dfs, out_dir)

    print("[OK] Wrote figures to:", out_dir)
    print("  - bm25_tf_saturation.png")
    print("  - bm25_length_normalization.png")
    print("  - idf_vs_df.png")
    print("  - zipf_rank_df.png")
    print(f"[Stats] N={N:,} avgdl={avgdl:.2f} terms={len(dfs):,}")

if __name__ == "__main__":
    main()
