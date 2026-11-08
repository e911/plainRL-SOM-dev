"""Numeric check of the EMA bias correction in WeightedSOMBuffer (constant input x = 1, alpha = 0.9)."""

from __future__ import annotations


def main() -> None:
    """Print the code's corrected mean, the correct-factor mean and the corrected variance after t hits.

    Returns:
        None.
    """
    a = 0.9
    for t_hits in (1, 2, 3, 5, 20):
        m = v = 0.0
        for _ in range(t_hits):
            d = m - 1.0
            m = (1 - a) * m + a * 1.0
            v = (1 - a) * v + a * d * d
        print(f"t={t_hits:2d} raw m={m:.3f}, code's corrected m={m / (1 - a ** t_hits):.3f}, "
              f"correct-factor m={m / (1 - (1 - a) ** t_hits):.3f}; var(code)={v / (1 - a ** t_hits):.3f} (true 0)")


if __name__ == "__main__":
    main()
