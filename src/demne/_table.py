"""Minimal pretty-print table with cell wrapping for terminal output."""
import textwrap


def print_table(headers: list[str], rows: list[list], col_widths: list[int]) -> None:
    """Print a box-drawing table. Cells exceeding col_widths wrap to next line."""
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def _fmt(cells):
        wrapped = [textwrap.wrap(str(c), col_widths[i]) or [""] for i, c in enumerate(cells)]
        height = max(len(w) for w in wrapped)
        for line_i in range(height):
            parts = []
            for i, w in enumerate(wrapped):
                cell = w[line_i] if line_i < len(w) else ""
                parts.append(f" {cell:<{col_widths[i]}} ")
            yield "|" + "|".join(parts) + "|"

    print(sep)
    for line in _fmt(headers):
        print(line)
    print(sep)
    for row in rows:
        for line in _fmt(row):
            print(line)
        print(sep)
