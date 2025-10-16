"""
query_bm25.py
--------------
BM25 query processor for the on-disk index produced by:
  - index_build.py  (writes doclen.bin)
  - index_merge.py  (writes postings.bin + lexicon.tsv)

Per-term layout in postings.bin (matches index_merge.py):
    [df:uint32]
    [docids_len:uint32][docid_deltas_varbyte...]
    [tfs_len:uint32]   [tf_varbyte...]

Features:
  • load lexicon.tsv into memory: term -> (offset, length, df)
  • decode postings per term on demand
  • BM25 scoring with DAAT (document-at-a-time), disjunctive/conjunctive
  • optional: print passage text and/or a lightweight query-biased snippet

Usage:
    python -m src.query_bm25 \
        --index_dir index/final \
        --mode disj \
        --k 10 \
        --k1 0.9 --b 0.4 \
        --page_table data/page_table.tsv \
        --collection data/collection.sample100000.tsv \
        --snippet --show_text
    # type queries on stdin; Ctrl-D to exit
"""

import os
import sys
import io
import math
import heapq
import struct
import argparse
from typing import Dict, Tuple, List
from collections import defaultdict

from src.varbyte import vb_decode_list

MAX_TEXT = 220      # chars when printing full text
SNIPPET_WINDOW = 35 # chars around first match


# -----------------------------
# Index I/O helpers
# -----------------------------
def load_lexicon(path: str) -> Dict[str, Tuple[int, int, int]]:
    """
    Read lexicon.tsv rows: term \t df \t offset \t length
    Returns dict: term -> (offset, length, df)
    """
    lex: Dict[str, Tuple[int, int, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                term, df_s, off_s, len_s = line.split("\t")
                off, ln, df = int(off_s), int(len_s), int(df_s)
            except ValueError:
                continue
            lex[term] = (off, ln, df)
    return lex


def load_doclens(path: str) -> List[int]:
    """
    doclen.bin stores 32-bit unsigned ints (little-endian), one per doc.
    """
    out: List[int] = []
    with open(path, "rb") as f:
        while True:
            b = f.read(4)
            if not b:
                break
            (l,) = struct.unpack("<I", b)
            out.append(int(l))
    return out


def read_term_postings(fp, offset: int, length: int) -> Tuple[List[int], List[int]]:
    """
    Seek to [offset], read [length] bytes, parse the term block as:
        df:uint32
        docids_len:uint32,  docids_varbyte[...]
        tfs_len:uint32,     tfs_varbyte[...]
    Return (docids, tfs) with absolute docIDs (delta-decoded).
    """
    fp.seek(offset)
    block = fp.read(length)
    pos = 0

    (df,) = struct.unpack_from("<I", block, pos); pos += 4
    (docids_len,) = struct.unpack_from("<I", block, pos); pos += 4
    enc_docids = block[pos:pos+docids_len]; pos += docids_len
    (tfs_len,) = struct.unpack_from("<I", block, pos); pos += 4
    enc_tfs = block[pos:pos+tfs_len]; pos += tfs_len

    deltas = vb_decode_list(enc_docids)
    tfs = vb_decode_list(enc_tfs)

    # delta -> absolute docids
    docids: List[int] = []
    running = 0
    for i, d in enumerate(deltas):
        if i == 0:
            running = d
        else:
            running += d
        docids.append(running)

    # sanity guard
    if df != len(docids) or df != len(tfs):
        m = min(len(docids), len(tfs), df)
        docids, tfs = docids[:m], tfs[:m]
    return docids, tfs


# -----------------------------
# BM25
# -----------------------------
class BM25:
    def __init__(self, doclens: List[int], k1: float = 0.9, b: float = 0.4):
        self.doclens = doclens
        self.N = len(doclens)
        self.avgdl = (sum(doclens) / self.N) if self.N else 0.0
        self.k1 = k1
        self.b = b

    def idf(self, df: int) -> float:
        # BM25 idf with +0.5 smoothing
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def tf_weight(self, tf: int, dl: int) -> float:
        denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl > 0 else 0))
        return (tf * (self.k1 + 1)) / (denom if denom > 0 else 1e-9)


# -----------------------------
# DAAT scoring
# -----------------------------
def score_disjunctive(
    bm25: BM25,
    term_postings: List[Tuple[int, List[int], List[int], float]]
) -> Dict[int, float]:
    """
    Disjunctive DAAT (OR): union of all postings.
    term_postings = list of (df, docids[], tfs[], idf)
    Returns doc -> score
    """
    scores = defaultdict(float)

    heap: List[Tuple[int, int, int]] = []  # (docid, term_index, pos_in_list)
    for ti, (_df, docs, _tfs, _idf) in enumerate(term_postings):
        if docs:
            heap.append((docs[0], ti, 0))
    heapq.heapify(heap)

    while heap:
        doc, ti, pi = heapq.heappop(heap)
        curr = doc
        contributors: List[Tuple[int, int]] = [(ti, pi)]

        # absorb any other heads pointing to same doc
        while heap and heap[0][0] == curr:
            _, ti2, pi2 = heapq.heappop(heap)
            contributors.append((ti2, pi2))

        dl = bm25.doclens[curr] if curr < bm25.N else bm25.avgdl
        s = 0.0
        for ti2, pi2 in contributors:
            df, docs2, tfs2, idf = term_postings[ti2]
            tf = tfs2[pi2]
            s += idf * bm25.tf_weight(tf, dl)
            nxt = pi2 + 1
            if nxt < len(docs2):
                heapq.heappush(heap, (docs2[nxt], ti2, nxt))
        scores[curr] += s

    return scores


def score_conjunctive(
    bm25: BM25,
    term_postings: List[Tuple[int, List[int], List[int], float]]
) -> Dict[int, float]:
    """
    Conjunctive DAAT (AND): intersection of all term postings.
    """
    if not term_postings or any(len(docs) == 0 for _, docs, *_ in term_postings):
        return {}

    positions = [0 for _ in term_postings]
    scores: Dict[int, float] = {}

    while True:
        try:
            curr = [term_postings[i][1][positions[i]] for i in range(len(term_postings))]
        except IndexError:
            break  # some list exhausted

        dmin, dmax = min(curr), max(curr)
        if dmin == dmax:
            doc = dmin
            dl = bm25.doclens[doc] if doc < bm25.N else bm25.avgdl
            s = 0.0
            for i, (df, docs, tfs, idf) in enumerate(term_postings):
                tf = tfs[positions[i]]
                s += idf * bm25.tf_weight(tf, dl)
                positions[i] += 1
            scores[doc] = s
        else:
            # advance pointers at the minimum docid
            for i in range(len(term_postings)):
                if term_postings[i][1][positions[i]] == dmin:
                    positions[i] += 1
                    if positions[i] >= len(term_postings[i][1]):
                        return scores
    return scores


# -----------------------------
# Optional text + snippets
# -----------------------------
def build_line_offsets(tsv_path: str) -> List[int]:
    """
    Record byte offset of each line so we can random-access doc text by docid.
    Assumes docid == 0-based line index in collection.tsv.
    """
    offsets: List[int] = []
    off = 0
    with open(tsv_path, "rb") as f:
        for line in f:
            offsets.append(off)
            off += len(line)
    return offsets


def read_doc_text(tsv_path: str, offsets: List[int], docid: int) -> str:
    """
    Return the text field at line 'docid' from collection TSV (after first TAB).
    """
    if docid < 0 or docid >= len(offsets):
        return ""
    with open(tsv_path, "rb") as f:
        f.seek(offsets[docid])
        raw = f.readline()
    try:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
    except Exception:
        line = raw.decode("latin1", errors="replace").rstrip("\n")
    if "\t" in line:
        _, text = line.split("\t", 1)
    else:
        text = line
    return text


def make_snippet(text: str, terms: List[str]) -> str:
    """
    Simple query-biased snippet: find first occurrence of any term,
    return +/- SNIPPET_WINDOW chars, and bold the terms (case-insensitive).
    """
    low = text.lower()
    first_pos = -1
    for t in terms:
        p = low.find(t)
        if p != -1 and (first_pos == -1 or p < first_pos):
            first_pos = p

    if first_pos == -1:
        base = text[:MAX_TEXT]
        if len(text) > MAX_TEXT:
            base += "…"
    else:
        start = max(0, first_pos - SNIPPET_WINDOW)
        end = min(len(text), first_pos + SNIPPET_WINDOW)
        base = text[start:end]
        if start > 0:
            base = "…" + base
        if end < len(text):
            base = base + "…"

    out = base
    for t in sorted(set(terms), key=len, reverse=True):
        # crude highlighting: exact + capitalized
        out = out.replace(t, f"**{t}**")
        out = out.replace(t.capitalize(), f"**{t.capitalize()}**")
    return out


def format_results(
    topK: List[Tuple[int, float]],
    page_table: Dict[int, str],
    terms: List[str],
    collection_path: str = "",
    offsets: List[int] = None,
    show_text: bool = False,
    show_snippet: bool = False
) -> str:
    lines: List[str] = []
    for rank, (doc, score) in enumerate(topK, 1):
        ext = page_table.get(doc, "")
        head = f"{rank:2d}. doc={doc:<8d} score={score:.4f}" + (f"  id={ext}" if ext else "")
        lines.append(head)
        if (show_text or show_snippet) and collection_path and offsets is not None:
            text = read_doc_text(collection_path, offsets, doc)
            if show_snippet:
                snip = make_snippet(text, terms)
                lines.append(f"    └─ snippet: {snip}")
            if show_text:
                clip = text[:MAX_TEXT] + ("…" if len(text) > MAX_TEXT else "")
                lines.append(f"    └─ text:    {clip}")
    return "\n".join(lines)


def load_page_table(path: str) -> Dict[int, str]:
    """
    Optional: docid \t original_passage_id
    """
    if not path or not os.path.exists(path):
        return {}
    pt: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            i, ext = line.rstrip("\n").split("\t", 1)
            try:
                pt[int(i)] = ext
            except ValueError:
                pass
    return pt


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, help="Directory with postings.bin, lexicon.tsv, doclen.bin")
    ap.add_argument("--mode", choices=["disj", "conj"], default="disj", help="disjunctive (OR) vs conjunctive (AND)")
    ap.add_argument("--k", type=int, default=10, help="top-K results")
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--page_table", default="", help="Optional page_table.tsv for external IDs")
    ap.add_argument("--collection", default="", help="Path to collection.tsv to enable text/snippets")
    ap.add_argument("--show_text", action="store_true", help="Print passage text (truncated)")
    ap.add_argument("--snippet", action="store_true", help="Print simple query-biased snippet")
    args = ap.parse_args()

    postings_path = os.path.join(args.index_dir, "postings.bin")
    lexicon_path  = os.path.join(args.index_dir, "lexicon.tsv")
    doclen_path   = os.path.join(args.index_dir, "doclen.bin")
    if not (os.path.exists(postings_path) and os.path.exists(lexicon_path) and os.path.exists(doclen_path)):
        print("[ERR] Missing index files in", args.index_dir, file=sys.stderr)
        sys.exit(2)

    lexicon = load_lexicon(lexicon_path)
    doclens = load_doclens(doclen_path)
    page_table = load_page_table(args.page_table)
    bm25 = BM25(doclens, k1=args.k1, b=args.b)

    offsets = None
    if (args.show_text or args.snippet) and args.collection:
        if not os.path.exists(args.collection):
            print(f"[WARN] --collection not found: {args.collection} (skipping text/snippet)")
        else:
            print(f"[OK] Pre-indexing collection line offsets: {args.collection}")
            offsets = build_line_offsets(args.collection)

    print(f"[OK] Loaded index: N={bm25.N} avgdl={bm25.avgdl:.2f} terms={len(lexicon)}")
    print(f"[OK] Mode: {args.mode}  k1={args.k1}  b={args.b}")
    print("Enter queries (Ctrl-D to exit):")

    # Main REPL
    with open(postings_path, "rb") as pf:
        for raw in sys.stdin:
            q = raw.strip()
            if not q:
                continue
            # simple whitespace tokenization; your common.tokenize could be used too
            terms = [t for t in q.lower().split() if t]
            # collect postings for in-vocab terms
            term_postings: List[Tuple[int, List[int], List[int], float]] = []
            for t in terms:
                meta = lexicon.get(t)
                if not meta:
                    continue
                off, length, df = meta
                docs, tfs = read_term_postings(pf, off, length)
                if docs:
                    term_postings.append((df, docs, tfs, bm25.idf(df)))

            if not term_postings:
                print("(no results)\n")
                continue

            if args.mode == "disj":
                scores = score_disjunctive(bm25, term_postings)
            else:
                scores = score_conjunctive(bm25, term_postings)

            topK = heapq.nlargest(args.k, scores.items(), key=lambda x: x[1])
            print(format_results(
                topK, page_table, terms,
                collection_path=args.collection,
                offsets=offsets,
                show_text=args.show_text,
                show_snippet=args.snippet
            ))
            print()

if __name__ == "__main__":
    main()
