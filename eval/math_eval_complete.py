"""
Complete Math Evaluation Pipeline for STEM Tutor Paper

Generates 5 figures for Results section:
  ① Math Retrieval Performance (Bar Chart)
     - Recall@1, Recall@3, Recall@5, nDCG@3, nDCG@5

  ② Symbolic Correctness by Dataset (Bar Chart)
     - Per-dataset symbolic accuracy

  ③ Mathematical Hallucination Distribution (Pie Chart)
     - arithmetic_error, algebra_mismatch, parse_error,
       unsupported_symbol, incorrect_formula_instantiation, random_nonsense

  ④ Accuracy by Difficulty Level (Bar Chart)
     - Easy / Medium / Hard symbolic accuracy

  ⑤ Error Breakdown (Bar Chart)
     - parsing_error, arithmetic_slip, algebraic_mismatch,
       sympy_mismatch, formula_misuse

Usage:
  python math_eval_complete.py --gold math_gold_hard_120_with_chunks.jsonl

If Ollama is not available, use --simulate to generate demo results.
"""

from __future__ import annotations
import argparse
import json
import re
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches

# ============ Configuration ============

class Config:
    # Output directory
    FIG_DIR = Path("figs")
    RESULTS_DIR = Path("results")

    # Style settings - clean academic style
    STYLE = {
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'axes.titleweight': 'bold',
    }

    # Color palette
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#28965A',
        'warning': '#F18F01',
        'danger': '#C73E1D',
        'neutral': '#6B7280',
        'light': '#E5E7EB',
    }

    # Dataset colors
    DATASET_COLORS = {
        'GSM8K': '#2E86AB',
        'GRADE-MATH': '#A23B72',
        'ASDIV': '#28965A',
        'SVAMP': '#F18F01',
    }

    # Difficulty colors
    DIFFICULTY_COLORS = {
        'Easy': '#28965A',
        'Medium': '#F18F01',
        'Hard': '#C73E1D',
    }

    # Hallucination type colors (6 types)
    HALLU_COLORS = [
        '#2E86AB',  # arithmetic_error
        '#A23B72',  # algebra_mismatch
        '#28965A',  # parse_error
        '#F18F01',  # unsupported_symbol
        '#C73E1D',  # incorrect_formula_instantiation
        '#6B7280',  # random_nonsense
    ]

    # Error type colors (5 types)
    ERROR_COLORS = [
        '#2E86AB',  # parsing_error
        '#A23B72',  # arithmetic_slip
        '#28965A',  # algebraic_mismatch
        '#F18F01',  # sympy_mismatch
        '#C73E1D',  # formula_misuse
    ]

Config.FIG_DIR.mkdir(exist_ok=True)
Config.RESULTS_DIR.mkdir(exist_ok=True)
plt.rcParams.update(Config.STYLE)


# ============ Data Classes ============

@dataclass
class GoldItem:
    """A single evaluation item from gold file"""
    id: str
    query: str
    gold_answer: str
    gold_chunks: List[str]
    dataset: str

    # Derived
    gold_numeric: Optional[float] = None
    difficulty: str = "Medium"
    num_steps: int = 1


@dataclass
class EvalResult:
    """Result of evaluating a single item"""
    id: str
    query: str
    gold_answer: str
    gold_numeric: Optional[float]
    pred_answer: str
    pred_numeric: Optional[float]

    # Correctness
    is_correct: bool = False

    # Retrieval info
    retrieved_chunks: List[str] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    gold_chunk_ranks: List[int] = field(default_factory=list)  # rank of each gold chunk

    # Classification
    dataset: str = ""
    difficulty: str = "Medium"

    # Error analysis (only if incorrect)
    error_type: str = ""  # for Error Breakdown chart
    hallucination_type: str = ""  # for Hallucination Distribution chart


@dataclass
class EvalSummary:
    """Aggregated evaluation results"""
    total: int = 0
    correct: int = 0

    # Retrieval metrics
    mrr: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0

    # Per-dataset accuracy
    dataset_accuracy: Dict[str, float] = field(default_factory=dict)
    dataset_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # (correct, total)

    # Per-difficulty accuracy
    difficulty_accuracy: Dict[str, float] = field(default_factory=dict)
    difficulty_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # Hallucination distribution (6 types)
    hallucination_counts: Dict[str, int] = field(default_factory=dict)

    # Error breakdown (5 types)
    error_counts: Dict[str, int] = field(default_factory=dict)

    # All results
    results: List[EvalResult] = field(default_factory=list)


# ============ Answer Extraction ============

# Regex patterns
HASH_ANSWER_RE = re.compile(r"####\s*(.+?)(?:\n|$)")
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
FRACTION_RE = re.compile(r"(-?\d+)\s*/\s*(\d+)")


def extract_numeric_answer(text: str) -> Tuple[str, Optional[float]]:
    """
    Extract the final numeric answer from text.
    Returns (raw_answer_string, numeric_value)
    """
    if not text:
        return "", None

    text = text.strip()

    # 1. Check for #### format (GSM8K style)
    m = HASH_ANSWER_RE.search(text)
    if m:
        ans = m.group(1).strip()
        # Clean commas and extract number
        ans_clean = ans.replace(",", "").replace("$", "")
        nums = NUMBER_RE.findall(ans_clean)
        if nums:
            try:
                return ans, float(nums[-1])
            except:
                pass
        return ans, None

    # 2. Check for \boxed{}
    m = BOXED_RE.search(text)
    if m:
        ans = m.group(1).strip()
        nums = NUMBER_RE.findall(ans.replace(",", ""))
        if nums:
            try:
                return ans, float(nums[-1])
            except:
                pass
        return ans, None

    # 3. Check for fraction
    m = FRACTION_RE.search(text)
    if m:
        try:
            num = float(m.group(1))
            denom = float(m.group(2))
            if denom != 0:
                return f"{m.group(1)}/{m.group(2)}", num / denom
        except:
            pass

    # 4. Extract last number (ignoring units)
    # Remove common units first
    text_clean = re.sub(r"\s*\([^)]*\)\s*$", "", text)  # Remove "(units)" at end
    text_clean = text_clean.replace(",", "").replace("$", "")

    nums = NUMBER_RE.findall(text_clean)
    if nums:
        try:
            return nums[-1], float(nums[-1])
        except:
            return nums[-1], None

    return text, None


def check_numeric_equivalence(gold: Optional[float], pred: Optional[float], tol: float = 1e-6) -> bool:
    """Check if two numbers are equivalent within tolerance"""
    if gold is None or pred is None:
        return False

    # Exact match
    if gold == pred:
        return True

    # Absolute tolerance
    if abs(gold - pred) < tol:
        return True

    # Relative tolerance (for larger numbers)
    if gold != 0 and abs((gold - pred) / gold) < 1e-4:
        return True

    return False


# ============ Difficulty Classification ============

def classify_difficulty(item: GoldItem) -> str:
    """
    Classify question difficulty based on dataset source.

    - ASDIV: Elementary school level → Easy
    - SVAMP: Simple word problems → Easy
    - GRADE-MATH: Grade school math → Medium
    - GSM8K: Multi-step reasoning → Hard
    """
    dataset = item.dataset.upper()

    if dataset in ["ASDIV", "SVAMP"]:
        return "Easy"
    elif dataset in ["GRADE-MATH"]:
        return "Medium"
    elif dataset in ["GSM8K"]:
        return "Hard"
    else:
        # Fallback: use query length as proxy
        query_words = len(item.query.split())
        if query_words < 30:
            return "Easy"
        elif query_words < 50:
            return "Medium"
        else:
            return "Hard"


# ============ Error Classification ============

def classify_error(gold_item: GoldItem, result: EvalResult) -> Tuple[str, str]:
    """
    Classify the error type and hallucination type.

    Error types (for Error Breakdown):
    - parsing_error: Failed to extract valid number
    - arithmetic_slip: Close but wrong (within 20% or off by small amount)
    - algebraic_mismatch: Wrong algebraic setup
    - sympy_mismatch: Symbolic form mismatch
    - formula_misuse: Used wrong formula/approach

    Hallucination types (for Hallucination Distribution):
    - arithmetic_error: Calculation mistake
    - algebra_mismatch: Wrong equation setup
    - parse_error: Output format issue
    - unsupported_symbol: Introduced undefined symbols
    - incorrect_formula_instantiation: Misused a formula
    - random_nonsense: Completely off-topic
    """
    pred = result.pred_answer.lower() if result.pred_answer else ""
    gold_num = result.gold_numeric
    pred_num = result.pred_numeric
    query = result.query.lower()

    # Default
    error_type = "algebraic_mismatch"
    hallu_type = "algebra_mismatch"

    # 1. Parse error - no valid number extracted
    if pred_num is None:
        # Check if it's truly unparseable or random nonsense
        if not pred or len(pred) < 3:
            return "parsing_error", "random_nonsense"
        if "sorry" in pred or "cannot" in pred or "don't know" in pred:
            return "parsing_error", "random_nonsense"
        if not any(c.isdigit() for c in pred):
            return "parsing_error", "parse_error"
        return "parsing_error", "parse_error"

    # 2. Check for arithmetic slip (close but wrong)
    if gold_num is not None and pred_num is not None:
        diff = abs(gold_num - pred_num)

        # Very close - arithmetic slip
        if gold_num != 0:
            rel_diff = diff / abs(gold_num)
            if rel_diff < 0.05:  # Within 5%
                return "arithmetic_slip", "arithmetic_error"
            if rel_diff < 0.2:  # Within 20%
                return "arithmetic_slip", "arithmetic_error"
        elif diff < 5:  # Small absolute difference
            return "arithmetic_slip", "arithmetic_error"

        # Check for common arithmetic mistakes
        # Off by factor of 10
        if gold_num != 0 and (abs(pred_num / gold_num - 10) < 0.01 or abs(pred_num / gold_num - 0.1) < 0.01):
            return "arithmetic_slip", "arithmetic_error"

        # Off by factor of 2 (common doubling/halving error)
        if gold_num != 0 and (abs(pred_num / gold_num - 2) < 0.01 or abs(pred_num / gold_num - 0.5) < 0.01):
            return "algebraic_mismatch", "incorrect_formula_instantiation"

    # 3. Check for unsupported symbols/variables in response
    pred_full = result.pred_answer
    if pred_full:
        # Check for undefined variables
        undefined_vars = re.findall(r"\b[a-zA-Z]\s*=", pred_full)
        if len(undefined_vars) > 2:
            return "formula_misuse", "unsupported_symbol"

    # 4. Check for formula misuse
    # If answer is way off (more than 5x or less than 0.2x)
    if gold_num is not None and pred_num is not None and gold_num != 0:
        ratio = pred_num / gold_num
        if ratio > 5 or ratio < 0.2:
            return "formula_misuse", "incorrect_formula_instantiation"

    # 5. SymPy mismatch - when numbers don't match but format is okay
    if gold_num is not None and pred_num is not None:
        return "sympy_mismatch", "algebra_mismatch"

    return error_type, hallu_type


# ============ Retrieval Metrics ============

def compute_mrr(gold_chunks: List[str], retrieved: List[str]) -> float:
    """Compute Mean Reciprocal Rank"""
    if not gold_chunks:
        return 0.0
    gold_set = set(gold_chunks)
    for i, chunk_id in enumerate(retrieved):
        if chunk_id in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(gold_chunks: List[str], retrieved: List[str], k: int) -> float:
    """Compute Recall@k"""
    if not gold_chunks:
        return 0.0
    retrieved_k = set(retrieved[:k])
    gold_set = set(gold_chunks)
    hits = len(retrieved_k & gold_set)
    return hits / len(gold_set)


def compute_ndcg_at_k(gold_chunks: List[str], retrieved: List[str], k: int) -> float:
    """Compute nDCG@k"""
    if not gold_chunks:
        return 0.0

    gold_set = set(gold_chunks)

    # DCG
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved[:k]):
        if chunk_id in gold_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_chunks), k)))

    return dcg / idcg if idcg > 0 else 0.0


# ============ Real System Calls ============

import subprocess
import sys

def call_retrieve(query: str, subject: str = "math", top_k: int = 50) -> Tuple[List[str], List[float]]:
    """
    Call retrieve.py to get ranked chunks.
    Returns (chunk_ids, scores)
    """
    # Find retrieve.py - adjust path as needed
    possible_paths = [
        Path("../retrieval/retrieve.py"),
        Path("retrieval/retrieve.py"),
        Path("./retrieve.py"),
    ]

    retrieve_script = None
    for p in possible_paths:
        if p.exists():
            retrieve_script = p
            break

    if retrieve_script is None:
        print("[WARN] retrieve.py not found, using simulation")
        return simulate_retrieval([], 1000)

    try:
        # We need to import and call directly for full ranking
        # Add parent to path
        sys.path.insert(0, str(retrieve_script.parent.parent))
        sys.path.insert(0, str(retrieve_script.parent))

        from retrieve import retrieve_full_ranked

        results = retrieve_full_ranked(query, subject=subject)
        chunk_ids = [r["id"] for r in results[:top_k]]
        scores = [r["score"] for r in results[:top_k]]

        return chunk_ids, scores
    except Exception as e:
        print(f"[WARN] retrieve.py call failed: {e}, using simulation")
        return simulate_retrieval([], 1000)


def call_qa(query: str, subject: str = "math", model: str = "qwen2.5:1.5b") -> str:
    """
    Call qa.py to get model answer.
    Returns the raw answer string.
    """
    # Find qa.py
    possible_paths = [
        Path("../retrieval/qa.py"),
        Path("retrieval/qa.py"),
        Path("./qa.py"),
    ]

    qa_script = None
    for p in possible_paths:
        if p.exists():
            qa_script = p
            break

    if qa_script is None:
        print("[WARN] qa.py not found")
        return ""

    try:
        cmd = [
            sys.executable,
            str(qa_script),
            query,
            "--subject", subject,
            "--llm-model", model,
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            cwd=str(qa_script.parent)
        )

        return proc.stdout
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        print(f"[WARN] qa.py call failed: {e}")
        return ""


def extract_answer_from_qa_output(raw_output: str) -> str:
    """Extract the answer portion from qa.py output"""
    # qa.py outputs: "Answer:\n <answer>\n\nContexts used:\n..."
    match = re.search(r"Answer:\s*\n?\s*(.+?)(?:\n\nContexts used:|\Z)", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_output.strip()


# ============ Simulation (for testing without Ollama) ============

def simulate_retrieval(gold_chunks: List[str], total_chunks: int = 1000) -> Tuple[List[str], List[float]]:
    """Simulate retrieval results for testing"""
    retrieved = []
    scores = []

    # Add some gold chunks with high probability
    for gc in gold_chunks:
        if random.random() < 0.6:
            retrieved.append(gc)
            scores.append(random.uniform(0.7, 0.95))

    # Fill with random chunks
    while len(retrieved) < 50:
        fake_id = f"fake-chunk-{random.randint(0, total_chunks)}"
        if fake_id not in retrieved:
            retrieved.append(fake_id)
            scores.append(random.uniform(0.3, 0.7))

    # Sort by score
    pairs = sorted(zip(retrieved, scores), key=lambda x: -x[1])
    retrieved, scores = zip(*pairs)

    return list(retrieved), list(scores)


def simulate_generation(gold_item: GoldItem) -> Tuple[str, Optional[float], bool]:
    """Simulate model generation for testing"""
    _, gold_num = extract_numeric_answer(gold_item.gold_answer)

    base_acc = {
        "Easy": 0.85,
        "Medium": 0.70,
        "Hard": 0.55,
    }.get(gold_item.difficulty, 0.70)

    dataset_adj = {
        "GSM8K": -0.05,
        "GRADE-MATH": 0.05,
        "ASDIV": 0.0,
        "SVAMP": 0.0,
    }.get(gold_item.dataset, 0.0)

    is_correct = random.random() < (base_acc + dataset_adj)

    if is_correct:
        pred_answer = gold_item.gold_answer
        pred_num = gold_num
    else:
        if gold_num is not None:
            error_type = random.choice(["arithmetic", "formula", "parse"])

            if error_type == "arithmetic":
                pred_num = gold_num + random.choice([-1, 1, -2, 2, -5, 5])
                pred_answer = str(int(pred_num) if pred_num == int(pred_num) else pred_num)
            elif error_type == "formula":
                pred_num = gold_num * random.choice([2, 0.5, 10, 0.1])
                pred_answer = str(int(pred_num) if pred_num == int(pred_num) else pred_num)
            else:
                pred_num = None
                pred_answer = "I cannot determine the answer from the given information."
        else:
            pred_num = random.randint(1, 100)
            pred_answer = str(pred_num)

    return pred_answer, pred_num, is_correct


# ============ Main Evaluation Logic ============

def load_gold_file(path: Path) -> List[GoldItem]:
    """Load gold evaluation file"""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            item = GoldItem(
                id=data.get("id", ""),
                query=data.get("query", ""),
                gold_answer=data.get("gold_answer", ""),
                gold_chunks=data.get("gold_chunks", []),
                dataset=data.get("_dataset", "UNKNOWN"),
            )

            # Extract numeric answer
            _, item.gold_numeric = extract_numeric_answer(item.gold_answer)

            # Classify difficulty
            item.difficulty = classify_difficulty(item)

            items.append(item)

    return items


def run_evaluation(gold_items: List[GoldItem], simulate: bool = True, model: str = "qwen2.5:1.5b") -> EvalSummary:
    """
    Run evaluation on all gold items.
    If simulate=True, generates synthetic results for testing.
    If simulate=False, calls actual retrieve.py and qa.py.
    """
    summary = EvalSummary()
    summary.total = len(gold_items)

    # Initialize counters
    dataset_correct = defaultdict(int)
    dataset_total = defaultdict(int)
    difficulty_correct = defaultdict(int)
    difficulty_total = defaultdict(int)
    hallu_counts = defaultdict(int)
    error_counts = defaultdict(int)

    # Retrieval metrics accumulators
    all_mrr = []
    all_recall_3 = []
    all_recall_5 = []
    all_ndcg_3 = []
    all_ndcg_5 = []

    from tqdm import tqdm

    for i, item in enumerate(tqdm(gold_items, desc="Evaluating")):
        # Get retrieval results
        if simulate:
            retrieved, scores = simulate_retrieval(item.gold_chunks)
        else:
            retrieved, scores = call_retrieve(item.query, subject="math", top_k=50)

        # Get generation results
        if simulate:
            pred_answer, pred_num, _ = simulate_generation(item)
        else:
            raw_output = call_qa(item.query, subject="math", model=model)
            pred_answer = extract_answer_from_qa_output(raw_output)
            _, pred_num = extract_numeric_answer(pred_answer)

        # Check correctness
        is_correct = check_numeric_equivalence(item.gold_numeric, pred_num)

        # Create result object
        result = EvalResult(
            id=item.id,
            query=item.query,
            gold_answer=item.gold_answer,
            gold_numeric=item.gold_numeric,
            pred_answer=pred_answer,
            pred_numeric=pred_num,
            is_correct=is_correct,
            retrieved_chunks=retrieved,
            retrieval_scores=scores,
            dataset=item.dataset,
            difficulty=item.difficulty,
        )

        # Compute retrieval metrics
        mrr = compute_mrr(item.gold_chunks, retrieved)
        r3 = compute_recall_at_k(item.gold_chunks, retrieved, 3)
        r5 = compute_recall_at_k(item.gold_chunks, retrieved, 5)
        n3 = compute_ndcg_at_k(item.gold_chunks, retrieved, 3)
        n5 = compute_ndcg_at_k(item.gold_chunks, retrieved, 5)

        all_mrr.append(mrr)
        all_recall_3.append(r3)
        all_recall_5.append(r5)
        all_ndcg_3.append(n3)
        all_ndcg_5.append(n5)

        # Update counters
        dataset_total[item.dataset] += 1
        difficulty_total[item.difficulty] += 1

        if is_correct:
            summary.correct += 1
            dataset_correct[item.dataset] += 1
            difficulty_correct[item.difficulty] += 1
        else:
            # Classify error
            error_type, hallu_type = classify_error(item, result)
            result.error_type = error_type
            result.hallucination_type = hallu_type
            error_counts[error_type] += 1
            hallu_counts[hallu_type] += 1

        summary.results.append(result)

    # Compute final metrics
    summary.mrr = np.mean(all_mrr)
    summary.recall_at_3 = np.mean(all_recall_3)
    summary.recall_at_5 = np.mean(all_recall_5)
    summary.ndcg_at_3 = np.mean(all_ndcg_3)
    summary.ndcg_at_5 = np.mean(all_ndcg_5)

    # Per-dataset accuracy
    for ds in dataset_total:
        summary.dataset_counts[ds] = (dataset_correct[ds], dataset_total[ds])
        summary.dataset_accuracy[ds] = dataset_correct[ds] / dataset_total[ds] if dataset_total[ds] > 0 else 0

    # Per-difficulty accuracy
    for diff in difficulty_total:
        summary.difficulty_counts[diff] = (difficulty_correct[diff], difficulty_total[diff])
        summary.difficulty_accuracy[diff] = difficulty_correct[diff] / difficulty_total[diff] if difficulty_total[diff] > 0 else 0

    # Error and hallucination counts
    summary.hallucination_counts = dict(hallu_counts)
    summary.error_counts = dict(error_counts)

    return summary


# ============ Plotting Functions ============

def plot_fig1_retrieval_performance(summary: EvalSummary, save_path: Path):
    """
    Figure 1: Math Retrieval Performance
    Bar chart with MRR, Recall@3, Recall@5, nDCG@3, nDCG@5
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['MRR', 'Recall@3', 'Recall@5', 'nDCG@3', 'nDCG@5']
    values = [
        summary.mrr,
        summary.recall_at_3,
        summary.recall_at_5,
        summary.ndcg_at_3,
        summary.ndcg_at_5,
    ]

    # Color gradient
    colors = ['#1a5276', '#2874a6', '#3498db', '#5dade2', '#85c1e9']

    x = np.arange(len(metrics))
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=2, width=0.6)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Math Retrieval Performance', fontsize=14, fontweight='bold', pad=15)

    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_fig2_symbolic_correctness_by_dataset(summary: EvalSummary, save_path: Path):
    """
    Figure 2: Symbolic Correctness by Dataset
    Bar chart showing accuracy per dataset
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort datasets by accuracy
    datasets = list(summary.dataset_accuracy.keys())
    accuracies = [summary.dataset_accuracy[ds] for ds in datasets]
    counts = [summary.dataset_counts[ds] for ds in datasets]

    # Sort descending
    sorted_idx = np.argsort(accuracies)[::-1]
    datasets = [datasets[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]

    x = np.arange(len(datasets))
    colors = [Config.DATASET_COLORS.get(ds, Config.COLORS['neutral']) for ds in datasets]

    bars = ax.bar(x, accuracies, color=colors, edgecolor='white', linewidth=2, width=0.6)

    # Add value labels with counts
    for bar, acc, (correct, total) in zip(bars, accuracies, counts):
        height = bar.get_height()
        ax.annotate(f'{acc:.1%}\n({correct}/{total})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Symbolic Accuracy', fontsize=12)
    ax.set_title('Symbolic Correctness by Dataset', fontsize=14, fontweight='bold', pad=15)

    # Overall accuracy line
    overall_acc = summary.correct / summary.total if summary.total > 0 else 0
    ax.axhline(y=overall_acc, color=Config.COLORS['danger'], linestyle='--',
               linewidth=2, label=f'Overall: {overall_acc:.1%}')
    ax.legend(loc='upper right', frameon=True, fancybox=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_fig3_hallucination_distribution(summary: EvalSummary, save_path: Path):
    """
    Figure 3: Mathematical Hallucination Distribution
    Pie chart showing types of hallucinations
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define all hallucination types
    hallu_types = [
        ('arithmetic_error', 'Arithmetic Error'),
        ('algebra_mismatch', 'Algebra Mismatch'),
        ('parse_error', 'Parse Error'),
        ('unsupported_symbol', 'Unsupported Symbol'),
        ('incorrect_formula_instantiation', 'Formula Misinstantiation'),
        ('random_nonsense', 'Random Nonsense'),
    ]

    labels = []
    sizes = []
    colors = []

    for i, (key, label) in enumerate(hallu_types):
        count = summary.hallucination_counts.get(key, 0)
        if count > 0:
            labels.append(label)
            sizes.append(count)
            colors.append(Config.HALLU_COLORS[i])

    if not sizes:
        # No errors - show placeholder
        labels = ['No Hallucinations']
        sizes = [1]
        colors = [Config.COLORS['success']]

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes))})',
        colors=colors,
        startangle=90,
        pctdistance=0.75,
        explode=[0.02] * len(sizes),
        wedgeprops=dict(edgecolor='white', linewidth=2),
    )

    # Style
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(11)

    ax.set_title('Mathematical Hallucination Distribution', fontsize=14, fontweight='bold', pad=20)

    # Add total count annotation
    total_errors = sum(sizes) if sizes[0] != 1 or labels[0] != 'No Hallucinations' else 0
    ax.annotate(f'Total Errors: {total_errors}',
                xy=(0, -1.2), fontsize=12, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=Config.COLORS['light']))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_fig4_accuracy_by_difficulty(summary: EvalSummary, save_path: Path):
    """
    Figure 4: Accuracy by Difficulty Level
    Bar chart showing Easy/Medium/Hard accuracy
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    difficulties = ['Easy', 'Medium', 'Hard']
    accuracies = []
    counts = []
    colors = []

    for diff in difficulties:
        acc = summary.difficulty_accuracy.get(diff, 0)
        cnt = summary.difficulty_counts.get(diff, (0, 0))
        accuracies.append(acc)
        counts.append(cnt)
        colors.append(Config.DIFFICULTY_COLORS[diff])

    x = np.arange(len(difficulties))
    bars = ax.bar(x, accuracies, color=colors, edgecolor='white', linewidth=2, width=0.5)

    # Add value labels
    for bar, acc, (correct, total) in zip(bars, accuracies, counts):
        height = bar.get_height()
        ax.annotate(f'{acc:.1%}\n({correct}/{total})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Symbolic Accuracy', fontsize=12)
    ax.set_title('Accuracy by Difficulty Level', fontsize=14, fontweight='bold', pad=15)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_fig5_error_breakdown(summary: EvalSummary, save_path: Path):
    """
    Figure 5: Error Breakdown
    Bar chart showing distribution of error types
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define all error types
    error_types = [
        ('parsing_error', 'Parsing Error'),
        ('arithmetic_slip', 'Arithmetic Slip'),
        ('algebraic_mismatch', 'Algebraic Mismatch'),
        ('sympy_mismatch', 'SymPy Mismatch'),
        ('formula_misuse', 'Formula Misuse'),
    ]

    labels = []
    counts = []
    colors = []

    for i, (key, label) in enumerate(error_types):
        count = summary.error_counts.get(key, 0)
        labels.append(label)
        counts.append(count)
        colors.append(Config.ERROR_COLORS[i])

    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, edgecolor='white', linewidth=2, width=0.6)

    # Add value labels
    total_errors = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = count / total_errors * 100 if total_errors > 0 else 0
        if count > 0:
            ax.annotate(f'{count}\n({pct:.1f}%)',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=15, ha='right')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Error Breakdown', fontsize=14, fontweight='bold', pad=15)

    # Add total annotation
    ax.annotate(f'Total Errors: {total_errors}',
                xy=(0.98, 0.95), xycoords='axes fraction',
                fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=Config.COLORS['light']))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def save_summary_json(summary: EvalSummary, save_path: Path):
    """Save evaluation summary as JSON"""
    data = {
        "total_questions": summary.total,
        "correct": summary.correct,
        "accuracy": summary.correct / summary.total if summary.total > 0 else 0,
        "retrieval": {
            "mrr": summary.mrr,
            "recall_at_3": summary.recall_at_3,
            "recall_at_5": summary.recall_at_5,
            "ndcg_at_3": summary.ndcg_at_3,
            "ndcg_at_5": summary.ndcg_at_5,
        },
        "dataset_accuracy": summary.dataset_accuracy,
        "dataset_counts": {k: {"correct": v[0], "total": v[1]} for k, v in summary.dataset_counts.items()},
        "difficulty_accuracy": summary.difficulty_accuracy,
        "difficulty_counts": {k: {"correct": v[0], "total": v[1]} for k, v in summary.difficulty_counts.items()},
        "hallucination_counts": summary.hallucination_counts,
        "error_counts": summary.error_counts,
    }

    save_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[SAVED] {save_path}")


def save_detailed_results(summary: EvalSummary, save_path: Path):
    """Save detailed per-question results as JSONL"""
    with save_path.open("w", encoding="utf-8") as f:
        for r in summary.results:
            data = {
                "id": r.id,
                "query": r.query,
                "gold_answer": r.gold_answer,
                "gold_numeric": r.gold_numeric,
                "pred_answer": r.pred_answer,
                "pred_numeric": r.pred_numeric,
                "is_correct": r.is_correct,
                "dataset": r.dataset,
                "difficulty": r.difficulty,
                "error_type": r.error_type,
                "hallucination_type": r.hallucination_type,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"[SAVED] {save_path}")


# ============ Main Entry Point ============

def main():
    parser = argparse.ArgumentParser(description="Math Evaluation Pipeline")
    parser.add_argument("--gold", type=Path, default=Path("math_gold_hard_120_with_chunks.jsonl"),
                        help="Path to gold evaluation file")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate results (for testing without Ollama)")
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b",
                        help="LLM model for generation (default: qwen2.5:1.5b)")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Output directory for figures and results")
    args = parser.parse_args()

    # Update paths
    Config.FIG_DIR = args.output_dir / "figs"
    Config.RESULTS_DIR = args.output_dir / "results"
    Config.FIG_DIR.mkdir(exist_ok=True, parents=True)
    Config.RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("MATH EVALUATION PIPELINE")
    print("=" * 60)

    # Load gold file
    print(f"\n[1/7] Loading gold file: {args.gold}")
    if not args.gold.exists():
        print(f"[ERROR] Gold file not found: {args.gold}")
        return

    gold_items = load_gold_file(args.gold)
    print(f"       Loaded {len(gold_items)} items")

    # Print dataset distribution
    dataset_dist = defaultdict(int)
    difficulty_dist = defaultdict(int)
    for item in gold_items:
        dataset_dist[item.dataset] += 1
        difficulty_dist[item.difficulty] += 1

    print(f"       Datasets: {dict(dataset_dist)}")
    print(f"       Difficulty: {dict(difficulty_dist)}")

    # Run evaluation
    mode_str = "(SIMULATED)" if args.simulate else f"(Model: {args.model})"
    print(f"\n[2/7] Running evaluation {mode_str}...")
    summary = run_evaluation(gold_items, simulate=args.simulate, model=args.model)

    print(f"       Accuracy: {summary.correct}/{summary.total} ({summary.correct/summary.total:.1%})")

    # Generate figures
    print(f"\n[3/7] Generating Figure 1: Retrieval Performance...")
    plot_fig1_retrieval_performance(summary, Config.FIG_DIR / "fig1_retrieval_performance.png")

    print(f"[4/7] Generating Figure 2: Symbolic Correctness by Dataset...")
    plot_fig2_symbolic_correctness_by_dataset(summary, Config.FIG_DIR / "fig2_symbolic_correctness_by_dataset.png")

    print(f"[5/7] Generating Figure 3: Hallucination Distribution...")
    plot_fig3_hallucination_distribution(summary, Config.FIG_DIR / "fig3_hallucination_distribution.png")

    print(f"[6/7] Generating Figure 4: Accuracy by Difficulty...")
    plot_fig4_accuracy_by_difficulty(summary, Config.FIG_DIR / "fig4_accuracy_by_difficulty.png")

    print(f"[7/7] Generating Figure 5: Error Breakdown...")
    plot_fig5_error_breakdown(summary, Config.FIG_DIR / "fig5_error_breakdown.png")

    # Save results
    print(f"\n[SAVE] Saving summary and detailed results...")
    save_summary_json(summary, Config.RESULTS_DIR / "eval_summary.json")
    save_detailed_results(summary, Config.RESULTS_DIR / "eval_detailed.jsonl")

    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  Figures: {Config.FIG_DIR}/")
    for f in sorted(Config.FIG_DIR.glob("*.png")):
        print(f"    - {f.name}")
    print(f"  Data: {Config.RESULTS_DIR}/")
    for f in sorted(Config.RESULTS_DIR.glob("*")):
        print(f"    - {f.name}")

    print(f"\n--- Summary ---")
    print(f"Total: {summary.total} | Correct: {summary.correct} | Accuracy: {summary.correct/summary.total:.1%}")
    print(f"Retrieval: MRR={summary.mrr:.3f} R@3={summary.recall_at_3:.3f} R@5={summary.recall_at_5:.3f}")
    print(f"           nDCG@3={summary.ndcg_at_3:.3f} nDCG@5={summary.ndcg_at_5:.3f}")


if __name__ == "__main__":
    main()
