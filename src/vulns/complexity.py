"""
Vulnerability scoring and complexity tracking.

Scoring uses severity weights: CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1
Complexity tracking uses radon's cc_visit for cyclomatic complexity.

Verified:
- radon.complexity.cc_visit(code_string) returns list of Function/Class objects
- Each has .name, .complexity, .lineno attributes
- radon is a pure Python lib, no native dependencies
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ComplexityMetrics:
    """Code complexity metrics for a single code sample."""
    cyclomatic_complexity: float  # Average CC across functions
    max_complexity: int  # Highest single-function CC
    num_functions: int
    num_classes: int
    lines_of_code: int
    num_imports: int

    def to_dict(self) -> dict:
        return {
            "cyclomatic_complexity": round(self.cyclomatic_complexity, 2),
            "max_complexity": self.max_complexity,
            "num_functions": self.num_functions,
            "num_classes": self.num_classes,
            "lines_of_code": self.lines_of_code,
            "num_imports": self.num_imports,
        }


def compute_complexity(code: str) -> ComplexityMetrics:
    """
    Compute code complexity metrics.

    Uses radon for cyclomatic complexity and ast for structural counts.
    Falls back to ast-only analysis if radon is unavailable.
    """
    # Lines of code (non-empty, non-comment)
    lines = code.strip().split("\n")
    loc = sum(
        1 for line in lines
        if line.strip() and not line.strip().startswith("#")
    )

    # Try radon for cyclomatic complexity
    avg_cc = 0.0
    max_cc = 0

    try:
        from radon.complexity import cc_visit

        results = cc_visit(code)
        if results:
            complexities = [r.complexity for r in results]
            avg_cc = sum(complexities) / len(complexities)
            max_cc = max(complexities)
    except ImportError:
        logger.debug("radon not available, using fallback CC estimation")
        avg_cc, max_cc = _estimate_complexity_from_ast(code)
    except Exception as e:
        logger.warning("radon CC computation failed: %s", e)
        avg_cc, max_cc = _estimate_complexity_from_ast(code)

    # AST-based counts
    num_functions = 0
    num_classes = 0
    num_imports = 0

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                num_functions += 1
            elif isinstance(node, ast.ClassDef):
                num_classes += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                num_imports += 1
    except SyntaxError:
        pass

    return ComplexityMetrics(
        cyclomatic_complexity=avg_cc,
        max_complexity=max_cc,
        num_functions=num_functions,
        num_classes=num_classes,
        lines_of_code=loc,
        num_imports=num_imports,
    )


def _estimate_complexity_from_ast(code: str) -> tuple[float, int]:
    """
    Fallback CC estimation using ast only.

    Counts decision points: if, elif, for, while, and, or,
    except, with, assert, ternary.
    CC = 1 + number_of_decision_points (per function).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0, 0

    decision_nodes = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert,
    )

    func_complexities = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = 1  # Base complexity
            for child in ast.walk(node):
                if isinstance(child, decision_nodes):
                    cc += 1
                elif isinstance(child, ast.BoolOp):
                    # Each 'and'/'or' adds a decision point
                    cc += len(child.values) - 1
            func_complexities.append(cc)

    if not func_complexities:
        return 0.0, 0

    return (
        sum(func_complexities) / len(func_complexities),
        max(func_complexities),
    )


def compute_complexity_delta(
    code_before: str, code_after: str
) -> dict:
    """Compute the change in complexity between two versions."""
    before = compute_complexity(code_before)
    after = compute_complexity(code_after)

    return {
        "cc_delta": round(
            after.cyclomatic_complexity - before.cyclomatic_complexity, 2
        ),
        "loc_delta": after.lines_of_code - before.lines_of_code,
        "functions_delta": after.num_functions - before.num_functions,
        "imports_delta": after.num_imports - before.num_imports,
        "before": before.to_dict(),
        "after": after.to_dict(),
    }
