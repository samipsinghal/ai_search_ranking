"""
index_merge.py
---------------
Stage 2 of the inverted-index construction pipeline.

This script merges all sorted partial posting runs (produced by index_build.py)
into a single compressed inverted index consisting of two output files:

    postings.bin  — the binary file containing all compressed postings lists
    lexicon.tsv   — a human-readable lexicon mapping each term to its byte offset in postings.bin

Each run_*.tsv is sorted by (term, docid). This script performs an external
multi-way merge using a heap to maintain the smallest (term, docid) across all runs,
similar to a k-way merge sort, but streaming from disk.

Resulting postings for each term are:
    [df:uint32]
    [docids_len:uint32][varbyte-encoded docid_deltas]
    [tfs_len:uint32][varbyte-encoded term frequencies]

This layout is intentionally simple so the query processor can
seek directly into postings.bin and decode each list independently.

Usage:
    python -m src.index_merge --tmpdir index/tmp --outdir index/final
"""

import os
import sys
import glob
import heapq
import struct
import argparse
from typing import Iterator, Tuple, Optional, List

# Varbyte encoding utilities — simple, fast integer compression
from src.varbyte import vb_encode_list


# ============================================================================
# 1. Input stream utilities
# ============================================================================

def iter_run(path: str) -> Iterator[Tuple[str, int, int]]:
    """
    Stream (term, docid, tf) triples from one sorted run file on disk.

    Each line looks like:
        term<TAB>docid<TAB>tf

    The function is a generator: it yields tuples lazily,
    so we never load the entire file into memory.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue  # skip malformed lines
            term, docid_s, tf_s = parts
            try:
                yield term, int(docid_s), int(tf_s)
            except ValueError:
                # rare: corrupted docid or tf field — skip quietly
                continue


class Cursor:
    """
    Cursor wraps a run iterator and exposes its current 'head' record.

    This is useful for k-way merge: we keep one Cursor per run file.
    The heap always contains the smallest head (term, docid) among all cursors.
    """

    __slots__ = ("run_id", "it", "head")

    def __init__(self, run_id: int, it: Iterator[Tuple[str, int, int]]):
        self.run_id = run_id
        self.it = it
        self.head: Optional[Tuple[str, int, int]] = None
        self._advance()

    def _advance(self):
        """Move the cursor forward by one record. If exhausted, head=None."""
        try:
            self.head = next(self.it)
        except StopIteration:
            self.head = None


# ============================================================================
# 2. Core merge + write routines
# ============================================================================

def _flush_term(term: str,
                postings: List[Tuple[int, int]],
                postings_f,
                lexicon_f,
                file_offset: int) -> int:
    """
    Write all postings for a single term to postings.bin and
    append a metadata line to lexicon.tsv.

    Each posting is a (docid, tf) pair. If multiple runs contained
    the same (term, docid), we sum their term frequencies.

    Returns the new file offset after writing.
    """
    if not postings:
        return file_offset

    # ----------------------------------------------------
    # 1. Coalesce duplicates (same docid across runs)
    # ----------------------------------------------------
    merged_docids: List[int] = []
    merged_tfs: List[int] = []
    prev_doc = None
    acc_tf = 0

    for d, tf in postings:
        if prev_doc is None:
            prev_doc = d
            acc_tf = tf
        elif d == prev_doc:
            # same docid appeared in multiple runs
            acc_tf += tf
        else:
            merged_docids.append(prev_doc)
            merged_tfs.append(acc_tf)
            prev_doc = d
            acc_tf = tf

    # flush last accumulated pair
    if prev_doc is not None:
        merged_docids.append(prev_doc)
        merged_tfs.append(acc_tf)

    df = len(merged_docids)
    if df == 0:
        return file_offset

    # ----------------------------------------------------
    # 2. Delta-encode docids (saves 50–80% space)
    # ----------------------------------------------------
    deltas = [merged_docids[0]]
    deltas.extend(merged_docids[i] - merged_docids[i - 1] for i in range(1, df))

    # ----------------------------------------------------
    # 3. Varbyte-encode docids + term frequencies
    # ----------------------------------------------------
    enc_docids = vb_encode_list(deltas)
    enc_tfs = vb_encode_list(merged_tfs)

    # ----------------------------------------------------
    # 4. Write binary block layout:
    #    [df][len(docids)][docids...][len(tfs)][tfs...]
    # ----------------------------------------------------
    block = (
        struct.pack("<I", df) +
        struct.pack("<I", len(enc_docids)) + enc_docids +
        struct.pack("<I", len(enc_tfs)) + enc_tfs
    )

    postings_f.write(block)
    length = len(block)

    # ----------------------------------------------------
    # 5. Record metadata for this term in lexicon.tsv
    # ----------------------------------------------------
    lexicon_f.write(f"{term}\t{df}\t{file_offset}\t{length}\n")

    return file_offset + length


def merge_runs(tmpdir: str, outdir: str) -> None:
    """
    Perform a multi-way merge across all run_*.tsv files.

    This is an external-memory algorithm (never loads everything at once).
    It maintains a min-heap keyed by (term, docid), so we always know
    which posting to process next globally.
    """
    os.makedirs(outdir, exist_ok=True)

    # Locate all intermediate runs
    run_paths = sorted(glob.glob(os.path.join(tmpdir, "run_*.tsv")))
    if not run_paths:
        print(f"[ERROR] No run files found in {tmpdir}", file=sys.stderr)
        sys.exit(1)

    postings_path = os.path.join(outdir, "postings.bin")
    lexicon_path = os.path.join(outdir, "lexicon.tsv")

    print(f"[INFO] Merging {len(run_paths)} runs from {tmpdir} ...")

    with open(postings_path, "wb") as postings_f, open(lexicon_path, "w", encoding="utf-8") as lexicon_f:

        # ------------------------------------------------
        # 1. Initialize cursors and heap
        # ------------------------------------------------
        cursors: List[Cursor] = []
        heap: List[Tuple[str, int, int]] = []  # (term, docid, run_id)
        for rid, path in enumerate(run_paths):
            cur = Cursor(rid, iter_run(path))
            cursors.append(cur)
            if cur.head is not None:
                t, d, _ = cur.head
                heapq.heappush(heap, (t, d, rid))

        # ------------------------------------------------
        # 2. Merge phase: process smallest (term, docid)
        # ------------------------------------------------
        current_term: Optional[str] = None
        postings: List[Tuple[int, int]] = []  # holds postings for one term
        file_offset = 0  # running byte offset in postings.bin

        while heap:
            term, docid, rid = heapq.heappop(heap)
            cur = cursors[rid]
            _, _, tf = cur.head  # safe: this head matches (term, docid)

            # term boundary check
            if current_term is None:
                current_term = term
            elif term != current_term:
                # we've finished collecting all postings for previous term
                file_offset = _flush_term(current_term, postings, postings_f, lexicon_f, file_offset)
                postings.clear()
                current_term = term

            postings.append((docid, tf))

            # advance the cursor and reinsert its new head into heap
            cur._advance()
            if cur.head is not None:
                t2, d2, _ = cur.head
                heapq.heappush(heap, (t2, d2, rid))

        # ------------------------------------------------
        # 3. Flush the final term
        # ------------------------------------------------
        if current_term is not None:
            file_offset = _flush_term(current_term, postings, postings_f, lexicon_f, file_offset)

    print(f"[OK] Wrote postings -> {postings_path}")
    print(f"[OK] Wrote lexicon  -> {lexicon_path}")


# ============================================================================
# 3. CLI entrypoint
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Merge sorted run files into a compressed inverted index.")
    parser.add_argument("--tmpdir", required=True, help="Directory containing run_*.tsv files from index_build.py")
    parser.add_argument("--outdir", required=True, help="Output directory for final index files")
    args = parser.parse_args()

    merge_runs(args.tmpdir, args.outdir)


if __name__ == "__main__":
    main()
