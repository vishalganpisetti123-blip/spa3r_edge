"""
Pure numerical comparison engine.

Responsibility: given two numpy arrays, compute error metrics and return
a structured result. No model loading, no file I/O, no side effects.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class CompareResult:
    max_error: float
    mean_error: float
    p95_error: float
    p99_error: float
    cosine_similarity: float
    shape_a: Tuple
    shape_b: Tuple
    passed: bool
    threshold: float
    label: str = ""


def compare(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = 1e-2,
    label: str = "",
) -> CompareResult:
    """Compare two arrays and return structured error metrics.

    Args:
        a: Reference array (e.g. PyTorch output).
        b: Candidate array (e.g. ONNX output).
        threshold: Max-error threshold for pass/fail.
        label: Human-readable label for this comparison.

    Returns:
        CompareResult with all metrics populated.

    Raises:
        ValueError: If shapes do not match.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    abs_err = np.abs(a.astype(np.float64) - b.astype(np.float64))

    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    cosine_sim = float(
        np.dot(a_flat, b_flat) / (norm_a * norm_b + 1e-12)
    )

    max_err = float(np.max(abs_err))

    return CompareResult(
        max_error=max_err,
        mean_error=float(np.mean(abs_err)),
        p95_error=float(np.percentile(abs_err, 95)),
        p99_error=float(np.percentile(abs_err, 99)),
        cosine_similarity=cosine_sim,
        shape_a=tuple(a.shape),
        shape_b=tuple(b.shape),
        passed=max_err < threshold,
        threshold=threshold,
        label=label,
    )


def format_result(result: CompareResult) -> str:
    """Format a CompareResult as a human-readable string."""
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    lines = [
        f"  {result.label or 'Comparison'}",
        f"    Shape:          {result.shape_a}",
        f"    Max error:      {result.max_error:.3e}  (threshold: {result.threshold:.0e})",
        f"    Mean error:     {result.mean_error:.3e}",
        f"    P95 error:      {result.p95_error:.3e}",
        f"    P99 error:      {result.p99_error:.3e}",
        f"    Cosine sim:     {result.cosine_similarity:.8f}",
        f"    Status:         {status}",
    ]
    return "\n".join(lines)
