# src/convert_brat_to_conll.py
# Usage:
#   python NER/src/1convert_brat_to_conll.py --brat_dir NER/data/raw/brat --out_dir NER/data/conll
#       --labels BIOMARKER STATUS VALUE METHOD --dev_ratio 0.1 --test_ratio 0.1 --seed 42
import argparse
import pathlib
import random
import re
import sys
from collections import Counter

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def read_brat_entities(ann_path: pathlib.Path, keep_labels: set):
    spans = []  # list[(start,end,label,orig_text)]
    if not ann_path.exists():
        return spans
    for line in read_text(ann_path).splitlines():
        if not line or not line.startswith("T"):  # ignore relations / notes
            continue
        try:
            # T1\tLABEL start end(;start end ...)\ttext
            _tid, head, orig = line.split("\t", 2)
            parts = head.split()
            label, offs = parts[0], " ".join(parts[1:])
            if keep_labels and label not in keep_labels:
                continue
            # discontinuous spans separated by ';'
            frags = [frag.strip() for frag in offs.split(";")]
            for frag in frags:
                a = frag.split()
                if len(a) >= 2:
                    s, e = int(a[0]), int(a[1])
                    spans.append((s, e, label, orig))
        except Exception:
            print(
                f"[WARN] Could not parse line in {ann_path.name}: {line}",
                file=sys.stderr,
            )
    # sort by start, then end
    spans.sort(key=lambda x: (x[0], x[1]))
    return spans


def assign_bio(tokens, spans, text):
    tags = ["O"] * len(tokens)
    # simple conflict resolver: prefer longest span first
    spans = sorted(spans, key=lambda x: -(x[1] - x[0]))
    [False] * len(tokens)
    for s, e, lab, orig_text in spans:
        # sanity check
        slice_txt = text[s:e]
        if slice_txt and orig_text and slice_txt not in orig_text:
            print(
                f"[WARN] Offset text mismatch: '{slice_txt[:30]}' vs ann '{orig_text[:30]}'",
                file=sys.stderr,
            )
        idxs = [i for i, (_, ts, te) in enumerate(tokens) if not (te <= s or ts >= e)]
        if not idxs:
            continue
        # if any token already tagged with another entity, skip (overlap)
        if any(tags[i] != "O" for i in idxs):
            continue
        tags[idxs[0]] = f"B-{lab}"
        for i in idxs[1:]:
            tags[i] = f"I-{lab}"
    return tags


def sentence_break(prev_end: int, next_start: int, text: str) -> bool:
    gap = text[prev_end:next_start]
    if "\n" in gap or "\r" in gap:
        return True
    # strong punctuation followed by space/newline
    return bool(re.search(r"[\.!?]\s*$", gap))


def doc_to_conll(tokens, tags, text):
    lines = []
    for i, ((tok, _s, e), tag) in enumerate(zip(tokens, tags, strict=False)):
        lines.append(f"{tok} {tag}")
        # sentence boundary between token i and i+1
        if i < len(tokens) - 1:
            if sentence_break(e, tokens[i + 1][1], text):
                lines.append("")
    # ensure doc boundary
    if not lines or lines[-1] != "":
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brat_dir", required=True)
    ap.add_argument("--out_dir", default="data/conll")
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    brat_dir = pathlib.Path(args.brat_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(args.labels) if args.labels else set()  # empty = keep all labels

    docs_conll = []
    label_counts = Counter()
    file_count = 0

    for txt_path in sorted(brat_dir.rglob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        text = read_text(txt_path)
        tokens = tokenize(text)
        spans = read_brat_entities(ann_path, keep)
        tags = assign_bio(tokens, spans, text)
        conll = doc_to_conll(tokens, tags, text)
        docs_conll.append(conll)
        # stats
        for t in tags:
            if t != "O":
                label_counts[t.split("-", 1)[1]] += 1
        file_count += 1

    if not docs_conll:
        print("[ERROR] Found no .txt files.", file=sys.stderr)
        sys.exit(1)

    random.Random(args.seed).shuffle(docs_conll)
    n = len(docs_conll)
    n_dev = max(1, int(args.dev_ratio * n))
    n_train = max(1, n - n_dev)
    train = docs_conll[:n_train]
    dev = docs_conll[n_train : n_train + n_dev]

    (out_dir / "train.conll").write_text("\n".join(train), encoding="utf-8")
    (out_dir / "dev.conll").write_text("\n".join(dev), encoding="utf-8")

    print(f"[OK] Processed {file_count} documents")
    print(f"[OK] Splits: train={len(train)} dev={len(dev)} -> {out_dir}")
    print("[STATS] Entity token counts:")
    for lab, cnt in sorted(label_counts.items()):
        print(f"  {lab}: {cnt}")


if __name__ == "__main__":
    main()

try:
    t = globals().get("tracker")
    if t:
        t.stop()
except Exception as e:
    print(
        f"\nWarning: Generalized error in Eco2AI tracking (likely 'N/A' vs float dtype issue): {e}"
    )
    print("Carbon emission tracking data could not be saved, but analysis results are preserved.")
