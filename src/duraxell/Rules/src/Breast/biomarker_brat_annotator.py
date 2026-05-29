import argparse
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

from eco2ai import Tracker, set_params

# -------------------- Regex patterns (robust, French-friendly) --------------------
ESTROGEN = (
    r"""(?iux)
(?:(?<=^)|(?<![\w%]))
(?:r[Ã©eÃ¨]cepteurs?\s*(?:(aux?)|(des?))?\s*(?:(((oe?)|Å“)estrog[eÃ©Ã¨]nes?)|((hormonaux)))|r\.?\s*o\.?|r\.?\s*e\.?|\bRE\b|\bRO\b|lum(inal)?b|\bR0\b|\bRH\b|\bER\b|((oe)|Å“)strog[eÃ©Ã¨]nes?)
((?:\s*\.\s*)?
((?:\s*(?:\(|\[)[^)\]]{0,40}(?:\)|\]))?
(?:\s*clones?\s*[:=]?\s*\S{1,20})?
\s*(?::|=|,|;|\(|\best\b|\bsont\b|\ba\b|\bÃ \b)?\s*
((?P<value>(\>|\<)?\d{1,3}(?:[.,]\d+)?\s*%?\+{0,4}|pos(?:iti[fv]e?s?)?|n[Ã©eÃ¨]g(?:ati[fv]e?s?)?|\+{1,4}|\-{1,4}){1,})\s?\)?){1,2})|(triple\s?n[Ã©|e|Ã¨]g(ati(f|(ve))s?)?)
    """,
    "Estrogen_receptor",
)

PROGESTERONE = (
    r"""(?iux)
(?:(?<=^)|(?<![\w%]))
(?:r[Ã©eÃ¨]cepteurs?\s*(?:aux?|des?|[Ã a]\sla\s|de\sla)?\s*((prog[eÃ©Ã¨]st[eÃ©Ã¨]rones?)|(hormonaux))|r\.?\s*p\.?|\bRP\b|\bRH\b|\bPR\b|\bPGR\b|\bPgR\b|prog[eÃ©Ã¨]st[eÃ©Ã¨]rones?)
((?:\s*\.\s*)?
((?:\s*(?:\(|\[)[^)\]]{0,40}(?:\)|\]))?
(?:\s*clones?\s*[:=]?\s*\S{1,20})?
\s*(?::|=|,|;|\(|\best\b|\bsont\b|\ba\b|\bÃ \b)?\s*
((?P<value>(\>|\<)?\d{1,3}(?:[.,]\d+)?\s*%?\+{0,4}(\s*(?:ans?|an|mois|semaines?|jours)\b)?|pos(?:iti[fv]e?s?)?|n[Ã©eÃ¨]g(?:ati[fv]e?s?)?|\+{1,4}|\-{1,4}){1,})\s?\)?){1,2})|(triple\s?n[Ã©|e|Ã¨]g(ati(f|(ve))s?)?) """,
    "Progesterone_receptor",
)

HER2 = (
    r"""(
        (?:(?:\s?|fish|/|anti\-?|\.|si|g[eÃ©Ã¨]ne|^)\s*)
        (?:
            her[^a-zA-Z\d:]?2? |
            c[^a-zA-Z\d:]?erbb?2 | c[^a-zA-Z\d:]?erb[^a-zA-Z\d:]?b[^a-zA-Z\d:]?2
        )
        (?:
            \s*[:=,;]?\s*
            (?:(?:score)?\s*)
            (?:\+|\-| ecd | [^a-zA-Z\d]?en\s*cours?[^a-zA-Z\d]? | en\s*attente |
                (?:non)?\s*amplifi\w* | pos(?:iti)?f?v?e? |((faible|forte)\s*intensit[eÃ©Ã¨])| n[Ã©eÃ¨]g(?:ati)?f?v?e? |
                zero | un | (?:sur)?expression | (?:non)?\s*(?:sur)?\s*[eÃ©Ã¨]xprim[Ã©eÃ¨]s? |
                (une?|deux|trois?)\s*(?:croix?)? | \d+\s*%?\+?
            )
            [^a-zA-Z\d]?\(?\s*
        ){1,}
     )
     |
     (?:triple\s?n[Ã©eÃ¨]g(?:ati(?:f|ve)s?)?)
    """,
    "HER2_status",
)

Ki67 = (
    r"""(?iux)\bki\s*[- ]?\s*(67)?(?:\s*[:=,;<>]\s*|\s*(?:est\s+)?(?:estim[Ã©e]e?|Ã©valu[Ã©e]e?)\s+(?:Ã |a|de)\s+|\s+[Ã a]\s+|\s*)?(\d+(?:[.,]\d+)?\s*%?\+{0,4}|en\s*cours?|non\s*[Ã©Ã¨]?valuable|index\s*de\s*prolif[eÃ©Ã¨]ration\s*(?:faible|fort)(?:\s*\d+\s*%)?)(?:\s*[?.!])?""",
    "Ki67",
)

FISH = (
    r"""(?:
        (?:si)?\s*(?:fish(?:\s*her2)?)\s*
        (?:
            \+{1,4} | \-+ | en\s*cours | en\s*attente |
            pos(?:iti)?f?v?e? | n[Ã©eÃ¨]g(?:ati)?f?v?e? | zero |
            (?:non)?\s*concluant | un | deux |
            (?:non)?\s*sur\s*[eÃ©Ã¨]xprime | (?:non)?\s*amplifi\w* |
            amplification\s+du\s+g[eÃ©Ã¨]ne\s+her2? | trois |
            \d+\s*%?\+?
        )+
    )""",
    "FISH",
)


# -------------------- Core extraction (original API) --------------------
def clean_receptor_match(value):

    txt = value.lower()

    # reject if immediately followed by 'ans', 'an', 'mois', 'semaines', 'jours'
    if re.search(r"\b\d+\s*(ans?|mois|semaines?|jours)\b", txt):
        return "nomatch"
    if "anti" in txt:
        return "nomatch"

    return "match"


def extractBiomarkers(marker, text: str):
    biomarker = re.compile(marker[0], re.IGNORECASE | re.UNICODE | re.VERBOSE)
    matches = []
    for m in biomarker.finditer(text):
        matchedtext = m.group().lower()
        # Skip naked "her2" tokens or leading "si " noise
        if re.sub(
            r"[^a-z0-9]", "", matchedtext
        ) == "her2" or matchedtext.strip().startswith("si "):
            continue
        elif clean_receptor_match(matchedtext) == "nomatch":
            continue
        # When pulling IHC HER2, drop FISH/gene/ecd mentions
        testFish = any(tok in matchedtext for tok in ("fish", "gene", "gÃ¨ne", "gÃ©ne"))
        if marker[1] == "HER2_status" and (testFish and "ecd" in matchedtext):
            continue
        matches.append((m.group(), m.start(), m.end(), marker[1]))
    return matches


# -------------------- Utilities --------------------
def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def _norm(s: str) -> str:
    s = _strip_accents(s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------- HER2 / FISH post-processing --------------------
_IHC_TOKEN_RE = re.compile(
    r"""
    (?:\bscore\s*(?P<score>[0-3])\b) |
    (?P<three>(?:\b3\+|\+{3}|\btrois\b)) |
    (?P<two>(?:\b2\+|\+{2}|\bdeux\b)) |
    (?P<one>(?:\b1\+|\bun\b)) |
    (?P<zero>\b0\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_her2_ihc_value(span_text: str):
    t = _norm(span_text)
    matches = [m.group(0).lower() for m in _IHC_TOKEN_RE.finditer(t)]
    if not matches:
        return None
    tok = _norm(matches[-1])
    if "3+" in tok or "+++" in tok or "trois" in tok or tok == "3" or "score 3" in t:
        return "3+"
    if "2+" in tok or "++" in tok or "deux" in tok or tok == "2" or "score 2" in t:
        return "2+"
    if "1+" in tok or tok == "1" or "score 1" in t or tok == "+":
        return "1+"
    if tok == "0" or "score 0" in t:
        return "0"
    return None


def _parse_her2_text_status(span_text: str):
    t = _norm(span_text)

    # Subtype shorthand implies positive
    if re.search(r"\bher\s*[-\s]?2\s*/\s*lumb\b|\bher\s*[-\s]?2\s*enrichi", t):
        return "positif"

    # Strong negation cues first
    neg = (
        r"\bnon\s*(?:amplifi\w*|sur[-\s]?exprim\w*)\b"
        r"|\bpas\s*(?:d[eu]\s*)?sur[-\s]?exprim\w*\b"
        r"|\babsence\s+d['â€™]?(?:une?\s*)?sur[-\s]?expression(?:\s+significative)?\s+de\s+her\s*[-\s]?2\b"
        r"|\-|\bneg\w*\b"
        r"|\bfaible\sintensit[Ã©eÃ¨]\b"
    )
    if re.search(neg, t):
        return "negatif"

    pos = r"\bamplifi\w*\b|(?!\d)\+(?!\d)|\bforte\sintensit[Ã©eÃ¨]\b|\bsur[-\s]?exprim\w*\b|\bpositi\w*\b"
    if re.search(pos, t):
        return "positif"

    return None


def _parse_fish_value(span_text: str):
    t = _norm(span_text)
    if re.search(r"\ben\s*(cours|attente)\b", t):
        return "en_cours"
    if re.search(r"\bnon\s*concluant\b", t):
        return "non_disponible"
    if re.search(r"\bnon\s+(?:amplifi|sur\s*exprime)\b", t):
        return "negatif"
    if re.search(r"\bneg\w*\b", t) or re.search(r"(^|\W)\-+(\W|$)", t):
        return "negatif"
    if (
        re.search(r"\bamplifi\w*\b", t)
        or re.search(r"\bpositi\w*\b", t)
        or re.search(r"(^|\W)\+{1,4}(\W|$)", t)
    ):
        return "positif"
    return "non_disponible"


def postprocess_her2_and_fish_annotations(text: str):
    """
    NEW: HER2 spans can yield BOTH HER2_IHC and HER2_status when both are present
    in the same span (e.g., 'HER2 nÃ©gatif score 1+').
    """
    enriched = []
    her2_spans = extractBiomarkers(HER2, text)
    for span_text, start, end, _ in her2_spans:
        # Try to emit BOTH, not either/or
        ihc_val = _parse_her2_ihc_value(span_text)
        status_val = _parse_her2_text_status(span_text)

        if ihc_val is not None:
            enriched.append((span_text, start, end, "HER2_IHC", ihc_val))
        if status_val is not None:
            enriched.append((span_text, start, end, "HER2_status", status_val))

        # If neither found, skip

    fish_spans = extractBiomarkers(FISH, text)
    for span_text, start, end, _ in fish_spans:
        fish_val = _parse_fish_value(span_text)
        enriched.append((span_text, start, end, "HER2_FISH", fish_val))
    return enriched


# -------------------- ER/PR parsing (>10% is positive) --------------------
def _classify_receptor_value(span_text: str) -> str:
    """
    Return 'positif' | 'negatif' | 'non_disponible' for ER/PR spans.
    Rule: if a % number exists, >10 => positif, else <=10 => negatif.
          Else fall back to +/-, or 'positif/negatif' keywords.
    NOTE: preserves '+' and '-' (does not hyphen-normalize away).
    """
    import re

    # Keep a version that preserves +/- for symbol checks
    t_signs = _strip_accents(span_text.lower())
    t_signs = re.sub(r"\s+", " ", t_signs).strip()

    # Word-normalized version for lexical matches
    t_words = _norm(span_text)  # hyphens -> spaces is fine here

    # Numeric % rule
    nums = [
        int(n) for n in re.findall(r"\b(\d{1,3})\s*%?", t_signs) if 0 <= int(n) <= 100
    ]
    if nums:
        return "positif" if nums[-1] > 10 else "negatif"

    # Lexical / symbol cues
    if re.search(r"\b(pos|lum(inal)?b)\w*\b", t_words) or "+" in t_signs:
        return "positif"

    if (
        re.search(r"\bneg\w*\b", t_words)
        or "-" in t_signs
        or re.search(r"\btriple\s*n[e|Ã©|Ã¨]g", t_signs)
    ):
        return "negatif"

    return "non_disponible"


def postprocess_er_pr_annotations(text: str):
    enriched = []
    for marker in (ESTROGEN, PROGESTERONE):
        spans = extractBiomarkers(marker, text)
        label = "Estrogen_receptor" if marker is ESTROGEN else "Progesterone_receptor"
        for span_text, start, end, _ in spans:
            val = _classify_receptor_value(span_text)
            enriched.append((span_text, start, end, label, val))
    return enriched


# -------------------- Ki-67 (optional biomarker) --------------------
def _parse_ki67_value(span_text: str):
    vals = re.findall(r"(\d{1,3})\s*%?", span_text)
    if vals:
        v = vals[-1]
        return f"{v}%"
    return None


def postprocess_ki67_annotations(text: str):
    enriched = []
    spans = extractBiomarkers(Ki67, text)
    for span_text, start, end, _ in spans:
        v = _parse_ki67_value(span_text)
        if v:
            enriched.append((span_text, start, end, "Ki67", v))
    return enriched


# -------------------- BRAT writer --------------------
def _sanitize_attr_value(v: str) -> str:
    return re.sub(r"\s+", "_", v)


def brat_annotate_biomarkers(text: str):
    ann = []
    ann.extend(postprocess_er_pr_annotations(text))
    ann.extend(postprocess_her2_and_fish_annotations(text))
    ann.extend(postprocess_ki67_annotations(text))
    return ann


def write_brat_ann(text: str, ann_path: Path):
    annotations = brat_annotate_biomarkers(text)
    annotations.sort(key=lambda x: (x[1], x[2]))

    lines = []
    T = 1
    A = 1

    for span_text, start, end, label, value in annotations:
        tid = f"T{T}"
        safe_span = span_text.replace("\n", " ")
        tline = f"{tid}\t{label} {start} {end}\t{safe_span}"
        lines.append(tline)

        if value:
            aval = _sanitize_attr_value(value)
            aid = f"A{A}"
            aline = f"{aid}\tValue {tid} {aval}"
            lines.append(aline)
            A += 1

        T += 1

    ann_path.write_text("\n".join(lines), encoding="utf-8")
    return lines


def brat_annotate_biomarkers_regex_only(text: str):
    """
    OLD SCHEME: use only the regex extractBiomarkers() matches.
    Returns a list of tuples (span_text, start, end, label) with labels:
      Estrogen_receptor, Progesterone_receptor, HER2_status, Ki67, FISH
    """
    ann = []
    for marker in (ESTROGEN, PROGESTERONE, HER2, Ki67, FISH):
        ann.extend(extractBiomarkers(marker, text))
    return ann


def write_brat_ann_regex_only(text: str, ann_path: Path):
    """
    OLD SCHEME writer: writes ONLY T-lines (no attributes).
    Format:
      Tn\t<label> start end\t<span_text>
    """
    spans = brat_annotate_biomarkers_regex_only(text)
    spans.sort(key=lambda x: (x[1], x[2]))  # stable numbering

    lines = []
    for i, (span_text, start, end, label) in enumerate(spans, 1):
        safe_span = span_text.replace("\n", " ")
        lines.append(f"T{i}\t{label} {start} {end}\t{safe_span}")

    ann_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return lines


def annotate_txt_folder_regex_only(
    in_dir: str,
    out_dir: str | None = None,
    *,
    recursive: bool = False,
    overwrite: bool = True,
):
    """
    Create OLD-SCHEME .ann files for every .txt under `in_dir`.
    - Uses regex-only spans (no values).
    - Writes <stem>.ann next to the .txt (or in `out_dir` if provided).
    - Set overwrite=False to skip if .ann already exists.
    """
    import os

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    processed = []
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if not f.lower().endswith(".txt"):
                continue
            txt_path = Path(root) / f
            if out_dir:
                rel = Path(root).relative_to(in_dir)
                out_sub = Path(out_dir) / rel
                out_sub.mkdir(parents=True, exist_ok=True)
                ann_path = out_sub / (txt_path.stem + ".ann")
            else:
                ann_path = txt_path.with_suffix(".ann")

            if ann_path.exists() and not overwrite:
                processed.append(str(ann_path))
                continue

            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            write_brat_ann_regex_only(text, ann_path)
            processed.append(str(ann_path))

        if not recursive:
            break

    return processed


def annotate_txt_folder_new_scheme(
    in_dir: str,
    out_dir: str | None = None,
    *,
    recursive: bool = False,
    overwrite: bool = True,
    exts: tuple[str, ...] = (".txt",),
):
    """
    Create NEW-SCHEME .ann files for every text under `in_dir`.
      - Uses your post-processed pipeline (brat_annotate_biomarkers + write_brat_ann).
      - If `out_dir` is given, mirror the subfolder structure there; else write next to each .txt.
      - `overwrite=False` skips files when an .ann already exists.
      - `exts` controls which file extensions are treated as text.

    Returns: list of paths to the .ann files written or skipped.
    """
    import os
    from pathlib import Path

    in_dir = str(in_dir)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    processed: list[str] = []
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if not f.lower().endswith(exts):
                continue

            txt_path = Path(root) / f

            if out_dir:
                rel = Path(root).relative_to(in_dir)
                target_dir = Path(out_dir) / rel
                target_dir.mkdir(parents=True, exist_ok=True)
                ann_path = target_dir / (txt_path.stem + ".ann")
            else:
                ann_path = txt_path.with_suffix(".ann")

            if ann_path.exists() and not overwrite:
                processed.append(str(ann_path))
                continue

            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            write_brat_ann(text, ann_path)
            processed.append(str(ann_path))

        if not recursive:
            break

    return processed


# -------------------- CLI --------------------
def _cli():
    parser = argparse.ArgumentParser(
        description="Create BRAT .ann biomarker annotations from a text report."
    )
    parser.add_argument("input", help="Input text file (.txt) to annotate")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .ann path (defaults to same stem as input with .ann)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    out_path = Path(args.output) if args.output else in_path.with_suffix(".ann")

    text = in_path.read_text(encoding="utf-8", errors="ignore")
    write_brat_ann(text, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    _cli()

    

