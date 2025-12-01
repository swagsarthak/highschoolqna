import math

def compute_map(results):
    APs = []
    for r in results:
        gold = set(r["gold"])
        preds = r["preds"]
        hits = 0
        total = 0
        for i, p in enumerate(preds):
            if p in gold:
                hits += 1
                total += hits / (i+1)
        if hits > 0:
            APs.append(total / hits)
    return sum(APs) / len(APs) if APs else 0

def compute_mrr(results):
    rr = []
    for r in results:
        gold = set(r["gold"])
        preds = r["preds"]
        for i, p in enumerate(preds):
            if p in gold:
                rr.append(1/(i+1))
                break
    return sum(rr)/len(rr) if rr else 0

def compute_recall(results, k=5):
    rs = []
    for r in results:
        gold = set(r["gold"])
        preds = set(r["preds"][:k])
        rs.append(len(gold & preds) / len(gold) if gold else 0)
    return sum(rs)/len(rs)

def compute_ndcg(results, k=10):
    def dcg(scores):
        return sum(s / math.log2(i+2) for i,s in enumerate(scores))
    ndcgs = []
    for r in results:
        gold = set(r["gold"])
        preds = r["preds"][:k]
        scores = [1 if p in gold else 0 for p in preds]
        ideal = sorted(scores, reverse=True)
        if sum(ideal)==0:
            continue
        ndcgs.append(dcg(scores)/dcg(ideal))
    return sum(ndcgs)/len(ndcgs) if ndcgs else 0
