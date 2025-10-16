#!/usr/bin/env python3
"""
Exports MS MARCO passages to collection.tsv (docid<TAB>text)

- Uses Hugging Face dataset: sentence-transformers/msmarco-corpus (config="passage")
- Writes dense integer docIDs (0..N-1) for simplicity + speed downstream
- Optional: also write a page_table.tsv mapping docid -> original passage_id
"""

import argparse, os
from datasets import load_dataset

# Nice progress bars if tqdm is installed; otherwise just iterate
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

def sanitize(text: str) -> str:
    """Make sure the text is single-line and tab-free, since we produce TSV."""
    return (text or "").replace("\t", " ").replace("\n", " ").strip()

def write_tsv(ds, out_path, limit=None, page_table_out=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pt_fp = open(page_table_out, "w", encoding="utf-8") if page_table_out else None

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(ds, desc=f"Writing {os.path.basename(out_path)}")):
            # HF schema here: 'passage_id' and 'passage' for the "passage" config
            text = sanitize(row.get("passage") or row.get("text") or "")
            f.write(f"{i}\t{text}\n")
            if pt_fp:
                # Keep a mapping so your query UI can show original IDs later
                pt_fp.write(f"{i}\t{row.get('passage_id','')}\n")
            n += 1
            if limit and n >= limit:
                break

    if pt_fp:
        pt_fp.close()
    print(f"[OK] Wrote {n} passages -> {out_path}")
    if page_table_out:
        print(f"[OK] Wrote page table -> {page_table_out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/collection.tsv", help="Output TSV: docid<TAB>text")
    ap.add_argument("--sample", type=int, default=0, help="If >0, also write a sample TSV with first N docs")
    ap.add_argument("--limit", type=int, default=0, help="Cap export at N rows (dev only)")
    ap.add_argument("--page_table_out", default="data/page_table.tsv",
                    help="Optional mapping docid<TAB>original_passage_id (set to '' to skip)")
    args = ap.parse_args()

    # ✅ The correct config is "passage" (available configs are ['passage','query'])
    ds = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train")

    page_table_path = args.page_table_out or None
    if page_table_path == "":
        page_table_path = None

    write_tsv(
        ds,
        args.out,
        limit=(args.limit or None),
        page_table_out=page_table_path
    )

    if args.sample:
        base, ext = os.path.splitext(args.out)
        sample_path = f"{base}.sample{args.sample}{ext or '.tsv'}"
        write_tsv(
            ds,
            sample_path,
            limit=args.sample,
            page_table_out=None  # samples don’t need a page table
        )

if __name__ == "__main__":
    main()
