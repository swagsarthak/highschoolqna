# eval/math_hallu.py
import re

def detect_math_hallucination(answer: str) -> bool:
    if not answer:
        return True

    lower = answer.lower()

    # illegal operations
    if "divide by 0" in lower or "1/0" in answer.replace(" ", ""):
        return True

    # fake math
    wrong_forms = [
        r"sin\(x\)\s*=\s*cos\(x\)",
        r"2\+2\s*=\s*5",
        r"pi\s*=\s*3"
    ]
    for w in wrong_forms:
        if re.search(w, lower):
            return True

    return False
