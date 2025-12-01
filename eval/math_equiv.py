# eval/math_equiv.py
from sympy import sympify, Eq
from sympy.core.sympify import SympifyError

def math_equivalent(a: str, b: str) -> bool:
    try:
        A = sympify(a)
        B = sympify(b)
        return Eq(A, B)
    except SympifyError:
        return a.strip() == b.strip()
