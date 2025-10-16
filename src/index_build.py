"""
index_build.py
---------------
Builds intermediate posting lists (run files) from the input collection.
Each run file contains sorted (term, docid, tf) lines.

This is the first stage of index construction.

Usage:
    python -m src.index_build --input data/collection.sample100000.tsv --outdir index/tmp --batch_docs 50000
"""

import os, sys, re, argparse
from collections import Counter
from src.common import tokenize  # our tokenizer from common.py

# -----------------------------
# helper: write one run to disk
# -----------------------------
def write_run(run_id, postings, outdir):
    """
    postings: list of (term, docid, tf)
    This function writes them sorted by (term, docid) into a run file.
    """
    os.makedirs(outdir, exist_ok=True)
    run_path = os.path.join(outdir, f"run_{run_id:06d}.tsv")
    postings.sort(key=lambda x: (x[0], x[1]))  # sort by term, then docid

    with open(run_path, "w", encoding="utf-8") as f:
        for term, docid, tf in postings:
            f.write(f"{term}\t{docid}\t{tf}\n")

    print(f"[run {run_id}] wrote {len(postings)} postings -> {run_path}")

# -----------------------------
# main logic: parse and spill
# -----------------------------
def build_index(input_path, outdir, batch_docs=50000):
    """
    Read the TSV file, tokenize passages, and spill sorted runs to disk in batches.
    """
    postings = []  # holds (term, docid, tf) tuples
    doc_lens = []  # document length (for BM25 normalization)
    run_id = 0
    doc_counter = 0

    # open the collection file line-by-line
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            # skip empty or malformed lines
            if not line.strip() or "\t" not in line:
                continue

            docid_str, text = line.split("\t", 1)
            try:
                docid = int(docid_str)
            except ValueError:
                # if docid isn’t integer, assign sequentially
                docid = doc_counter

            # tokenize and count term frequencies
            terms = list(tokenize(text))
            tf_counts = Counter(terms)
            doc_lens.append(len(terms))
            doc_counter += 1

            # collect postings
            for term, tf in tf_counts.items():
                postings.append((term, docid, tf))

            # every batch_docs, sort and write a run file to disk
            if doc_counter % batch_docs == 0:
                write_run(run_id, postings, outdir)
                postings.clear()
                run_id += 1

    # flush any remaining postings
    if postings:
        write_run(run_id, postings, outdir)

    # write doc lengths sidecar file (doclen.bin)
    doclen_path = os.path.join(os.path.dirname(outdir), "final", "doclen.bin")
    os.makedirs(os.path.dirname(doclen_path), exist_ok=True)

    with open(doclen_path, "wb") as f:
        for l in doc_lens:
            # store each doc length as a 32-bit unsigned int (little endian)
            f.write(l.to_bytes(4, "little", signed=False))

    print(f"[OK] wrote {len(doc_lens)} doc lengths -> {doclen_path}")
    print(f"[OK] total documents processed: {doc_counter}")

# -----------------------------
# entrypoint
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to collection.tsv")
    parser.add_argument("--outdir", required=True, help="Where to write run files")
    parser.add_argument("--batch_docs", type=int, default=50000,
                        help="Number of docs per run before spilling to disk")
    args = parser.parse_args()

    build_index(args.input, args.outdir, batch_docs=args.batch_docs)