"""
Symbolic Mathematics & Algebraic Verifier for Mathematical Reasoning RL.
Uses SymPy to verify mathematical equivalence, formula derivations, and step consistency
without reliance on brittle exact-string matching.
"""

import re
from typing import Dict, Any, Tuple, Optional, List
import sympy
from sympy import simplify, N, sympify, Symbol, Eq


class SymbolicMathVerifier:
    """
    Evaluates mathematical expressions, solutions, and proof steps symbolically.
    Determines equivalence between model predictions and gold ground truth:
      simplify(E_pred - E_gold) == 0
    """

    def __init__(self, numeric_tolerance: float = 1e-6):
        self.numeric_tolerance = numeric_tolerance

    def _clean_latex_math(self, expr_str: str) -> str:
        """Strips LaTeX formatting into sympy-parseable syntax."""
        expr_str = expr_str.strip()
        # Remove delimiters $ or $$ or \( or \)
        expr_str = re.sub(r"^\$+|\$+$", "", expr_str)
        expr_str = re.sub(r"^\\\[|\\\]$", "", expr_str)
        expr_str = re.sub(r"^\\\(|\\\)$", "", expr_str)

        # Convert standard LaTeX commands
        expr_str = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expr_str)
        expr_str = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", expr_str)
        expr_str = re.sub(r"\\cdot|\\times", "*", expr_str)
        expr_str = re.sub(r"\\left|\\right", "", expr_str)
        expr_str = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", expr_str)
        expr_str = re.sub(r"\^([0-9a-zA-Z])", r"**\1", expr_str)
        expr_str = re.sub(r"\\pi", "pi", expr_str)

        return expr_str.strip()

    def parse_expression(self, expr_str: str) -> Optional[sympy.Expr]:
        """Parses a mathematical string into a SymPy expression."""
        cleaned = self._clean_latex_math(expr_str)
        try:
            return sympify(cleaned, evaluate=False)
        except Exception:
            # Fallback simple string cleanup
            try:
                cleaned_loose = re.sub(r"([0-9])([a-zA-Z])", r"\1*\2", cleaned)
                return sympify(cleaned_loose, evaluate=False)
            except Exception:
                return None

    def verify_equivalence(
        self,
        predicted_str: str,
        gold_str: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determines if predicted mathematical expression is symbolically equivalent to gold.
        Returns:
            (is_equivalent: bool, details_dict)
        """
        pred_expr = self.parse_expression(predicted_str)
        gold_expr = self.parse_expression(gold_str)

        if pred_expr is None or gold_expr is None:
            # Fallback to normalized string comparison if parsing fails
            clean_p = re.sub(r"\s+", "", self._clean_latex_math(predicted_str).lower())
            clean_g = re.sub(r"\s+", "", self._clean_latex_math(gold_str).lower())
            string_match = (clean_p == clean_g)
            return string_match, {
                "method": "string_fallback",
                "predicted_parsed": str(pred_expr),
                "gold_parsed": str(gold_expr),
                "reward": 1.0 if string_match else 0.0
            }

        try:
            # 1. Symbolic simplification of difference: simplify(pred - gold) == 0
            diff = simplify(pred_expr - gold_expr)
            if diff == 0:
                return True, {"method": "symbolic_zero_diff", "reward": 1.0, "diff": str(diff)}

            # 2. Numerical evaluation for floating/constant expressions
            try:
                num_pred = complex(N(pred_expr))
                num_gold = complex(N(gold_expr))
                if abs(num_pred - num_gold) < self.numeric_tolerance:
                    return True, {
                        "method": "numerical_tolerance",
                        "reward": 1.0,
                        "delta": abs(num_pred - num_gold)
                    }
            except Exception:
                pass

            return False, {"method": "symbolic_divergence", "reward": 0.0, "diff": str(diff)}

        except Exception as e:
            return False, {"method": "error", "error": str(e), "reward": 0.0}

    def verify_step_sequence(self, steps: List[str]) -> Tuple[bool, int, List[Dict[str, Any]]]:
        """
        Verifies step-by-step equality chain: Step 0 == Step 1 == Step 2 ...
        Identifies the exact step where deductive reasoning breaks down.
        Returns:
            (all_valid: bool, first_error_step_index: int, step_evaluations: list)
        """
        evaluations = []
        for i in range(len(steps) - 1):
            s_current = steps[i]
            s_next = steps[i + 1]

            is_eq, details = self.verify_equivalence(s_current, s_next)
            evaluations.append({
                "step_from": i,
                "step_to": i + 1,
                "valid": is_eq,
                "details": details
            })
            if not is_eq:
                return False, i + 1, evaluations

        return True, -1, evaluations
