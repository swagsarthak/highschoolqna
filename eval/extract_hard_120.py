import json
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# =============== 路径配置 ===============

THIS_DIR = Path(__file__).resolve().parent          # eval/
PROJECT_ROOT = THIS_DIR.parent                      # highschoolqna/

# 1) 题库（大 pool），你可以按实际文件名改：
#   每行一个 JSON，字段可能是：
#   - id / source / question / gold
#   - 或 id / query / gold_answer
QA_POOL = PROJECT_ROOT / "eval" / "math_gold_pool.jsonl"

# 2) 固定长度 chunk 文件（你刚给我的这种）
CHUNKS_FILE = PROJECT_ROOT / "cleaning" / "chunks" / "math" / "math_unified_chunks.jsonl"

# 3) 输出：最终 120 题
OUTPUT = PROJECT_ROOT / "eval" / "math_gold_hard_120_with_chunks.jsonl"

# 想要的各数据集题目数（总和 120）
DATASET_TARGETS = {
    "GSM8K": 35,
    "GRADE-MATH": 35,
    "ASDIV": 25,
    "SVAMP": 25,
}

# gold_chunks 最多保留多少个 chunk id
TOPK_CHUNKS = 5

# =============== 一些小工具函数 ===============

def normalize_question(rec: Dict[str, Any]) -> str:
    """兼容不同字段名：question / query"""
    if "question" in rec:
        return rec["question"]
    if "query" in rec:
        return rec["query"]
    if "prompt" in rec:
        return rec["prompt"]
    return ""

def normalize_answer(rec: Dict[str, Any]) -> str:
    """兼容 gold / gold_answer 等"""
    if "gold_answer" in rec:
        return rec["gold_answer"]
    if "gold" in rec:
        return rec["gold"]
    if "answer" in rec:
        return rec["answer"]
    return ""

def detect_dataset(rec: Dict[str, Any]) -> str:
    """
    尝试从 source / dataset / id 里猜，这是 GSM8K / Grade-Math / SVAMP / ASDIV 里的谁
    """
    raw = (str(rec.get("source", "")) + " " +
           str(rec.get("dataset", "")) + " " +
           str(rec.get("id", ""))).upper()

    if "GSM8K" in raw:
        return "GSM8K"
    if "GRADE-MATH" in raw or "GRADE-MATH-18K" in raw or "GRADE-MATH-18K" in raw or "GRADE-MATH-18K" in raw:
        return "GRADE-MATH"
    if "SVAMP" in raw:
        return "SVAMP"
    if "ASDIV" in raw:
        return "ASDIV"
    return "OTHER"

def is_bad_mc_question(q: str, a: str) -> bool:
    """
    粗糙过滤那种「有 MC 结构但题干没选项」或者明显不完整的：
    - 题目里出现选项提示，但没有 A/B/C/D 等
    - 答案里既没有 '####' 最终数值，也没有像样的解题步骤
    你要更 aggressive 可以自己再加规则。
    """
    q_lower = q.lower()

    # 明确带 "options", "choose", 但正文里没有 A/B/C/D 明显编号
    if ("option" in q_lower or "choose" in q_lower) and not re.search(r"\b(A|B|C|D)[\).\]]", q):
        return True

    # 很多 AQUA 那种，gold 只写了方程，没有最后答案
    # 这里要求答案里包含 '####' 或者至少有一个明显的数字
    a_clean = a.strip()
    if "####" not in a_clean:
        # 没 '####' 时，再看有没有明显结果数字（带等号 / is / are）
        if not re.search(r"= *-?\d+(\.\d+)?", a_clean) and not re.search(r"[-+]?\d+(\.\d+)?", a_clean):
            return True

    return False

def tokenize(text: str) -> List[str]:
    """非常简单的分词：小写 + 保留长度>=3 的英文/数字串"""
    text = text.lower()
    # 把非字母数字统统变成空格
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if len(t) >= 3]
    return tokens

# =============== 载入 chunks ===============

def load_chunks():
    """
    读 math_unified_chunks.jsonl
    每行形如：
    {
      "id": "math-text-0000480",
      "word_count": 120,
      "text": "....",
      "source_pdf": "Prealgebra2e-WEB_0qbw93r.pdf"
    }
    """
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("id") or obj.get("chunk_id")
            text = obj.get("text", "")
            if not cid or not text:
                continue
            chunks.append((cid, text))

    print(f"[INFO] Loaded chunks: {len(chunks)}")
    return chunks

def build_inverted_index(chunks):
    """
    为了粗糙匹配问题→chunk，用倒排索引加点简单打分：
    score(chunk) = 共有 token 数
    """
    inverted = defaultdict(set)  # token -> set(chunk_idx)
    chunk_tokens = []           # idx -> set(tokens)

    for idx, (_, text) in enumerate(chunks):
        toks = set(tokenize(text))
        chunk_tokens.append(toks)
        for tok in toks:
            inverted[tok].add(idx)

    return inverted, chunk_tokens

def find_gold_chunks(question: str,
                     chunks,
                     inverted,
                     chunk_tokens,
                     topk: int = TOPK_CHUNKS) -> List[str]:
    """
    给一个 question，按 token 重合度找到若干最相关 chunk 的 id 列表
    """
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return []

    candidate_idxs = set()
    for tok in q_tokens:
        candidate_idxs |= inverted.get(tok, set())

    if not candidate_idxs:
        return []

    scores = []
    for idx in candidate_idxs:
        overlap = len(q_tokens & chunk_tokens[idx])
        if overlap > 0:
            scores.append((overlap, idx))

    if not scores:
        return []

    scores.sort(reverse=True)   # 按 overlap 从大到小排
    top = scores[:topk]

    # 把对应的 chunk id 拿出来
    gold_ids = [chunks[idx][0] for _, idx in top]
    return gold_ids

# =============== 主逻辑 ===============

def main():
    # 1. 读大题库
    all_recs = []
    with open(QA_POOL, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = normalize_question(obj)
            a = normalize_answer(obj)
            if not q or not a:
                continue

            ds = detect_dataset(obj)
            obj["_question"] = q
            obj["_answer"] = a
            obj["_dataset"] = ds
            all_recs.append(obj)

    print(f"[INFO] Loaded QA pool: {len(all_recs)} examples")

    # 2. 按数据集分桶 + 过滤怪 MC 题
    buckets = defaultdict(list)
    for rec in all_recs:
        ds = rec["_dataset"]
        if ds not in DATASET_TARGETS:
            continue

        if is_bad_mc_question(rec["_question"], rec["_answer"]):
            continue

        buckets[ds].append(rec)

    for ds, items in buckets.items():
        print(f"[INFO] {ds}: {len(items)} candidates after filtering")

    # 3. 每个数据集随机抽需要的数量
    selected = []
    for ds, target in DATASET_TARGETS.items():
        cand = buckets.get(ds, [])
        if not cand:
            print(f"[WARN] No candidates for dataset {ds}, skip.")
            continue

        if len(cand) <= target:
            print(f"[WARN] {ds}: only {len(cand)} available, take all (target {target})")
            picked = cand
        else:
            picked = random.sample(cand, target)

        print(f"[OK] {ds}: {len(picked)} picked")
        selected.extend(picked)

    random.shuffle(selected)
    print(f"[INFO] Total selected: {len(selected)}")

    # 4. 载入 chunks + 建倒排索引
    chunks = load_chunks()
    inverted, chunk_tokens = build_inverted_index(chunks)

    # 5. 对每道题找 gold_chunks，并写输出
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for i, rec in enumerate(selected):
            q = rec["_question"]
            a = rec["_answer"]
            gold_chunks = rec.get("gold_chunks")
            # 如果原来就有 gold_chunks，就保留；否则我们用文本匹配算一个
            if not gold_chunks:
                gold_chunks = find_gold_chunks(q, chunks, inverted, chunk_tokens)

            out_rec = {
                "id": rec.get("id", f"math-hard-{i:04d}"),
                "query": q,
                "gold_answer": a,
                "gold_chunks": gold_chunks,
                "_dataset": rec["_dataset"],
            }
            out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved → {OUTPUT}")

if __name__ == "__main__":
    main()
