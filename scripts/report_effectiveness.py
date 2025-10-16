#!/usr/bin/env python3
"""
report_effectiveness.py
------------------------
Evaluates a TREC run file against qrels using pytrec_eval.
"""

import argparse, pytrec_eval

def load_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels

def load_run(path):
    run = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            run.setdefault(qid, {})[docid] = float(score)
    return run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    run = load_run(args.run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'recall_1000', 'recip_rank', 'ndcg_cut.10'})
    results = evaluator.evaluate(run)

    mrr = sum(v['recip_rank'] for v in results.values()) / len(results)
    map_ = sum(v['map'] for v in results.values()) / len(results)
    ndcg10 = sum(v['ndcg_cut_10'] for v in results.values()) / len(results)
    recall1000 = sum(v['recall_1000'] for v in results.values()) / len(results)

    print(f"Documents evaluated: {len(run)} queries = {len(results)}")
    print(f"MRR@10     : {mrr:.4f}")
    print(f"MAP        : {map_:.4f}")
    print(f"nDCG@10    : {ndcg10:.4f}")
    print(f"Recall@1000: {recall1000:.4f}")

if __name__ == "__main__":
    main()
