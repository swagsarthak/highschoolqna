# eval/math_extract_answer.py
import re

def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    t = text.strip()

    # 1. latex \boxed{}
    m = re.findall(r"\\boxed\{(.+?)\}", t)
    if m:
        return m[-1].strip()

    m = re.findall(r"\\boxed\((.+?)\)", t)
    if m:
        return m[-1].strip()

    # 2. Final Answer:
    m = re.search(r"(Final\s*Answer\s*[:：]\s*)(.+)", t, re.IGNORECASE)
    if m:
        return m.group(2).strip()

    # 3. latex equations
    m = re.findall(r"\${1,2}(.+?)\${1,2}", t)
    if m:
        return m[-1].strip()

    # 4. last number / fraction
    m = re.findall(r"[0-9]+(?:/[0-9]+)?(?:\.[0-9]+)?", t)
    if m:
        return m[-1]

    # fallback: last sentence
    parts = re.split(r"[.;\n]", t)
    if parts:
        return parts[-1].strip()

    return t
