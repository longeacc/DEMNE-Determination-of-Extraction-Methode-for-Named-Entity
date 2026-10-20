"""Optuna recalibration of the DEMNE weights & thresholds.

    python main.py optimize_params --in_domain
    python main.py optimize_params --all_corpora
    python -m demne.optimize_params --in_domain        # same, without main.py

Corpus layout (resolved against the CURRENT WORKING DIRECTORY, not the package)
--------------------------------------------------------------------------------
    data/OPT_CORPORA/<corpus>/          target domain — optimisation
    data/TEST_CORPORA/<corpus>/         target domain — held-out report
    data/OFF_DOMAIN_CORPORA/<corpus>/   off-domain — final fit with --all_corpora only
    data/ALL_CORPORA/<corpus>/          optional — final-fit corpora; if absent or
                                        empty, OPT_CORPORA ∪ TEST_CORPORA is used

Every <corpus>/ folder (any name) holds BRAT pairs (*.ann + *.txt, sub-folders
allowed) and ONE gold file: *_routing.csv (preferred) or *.csv, with an
``entity``/``Entity`` column and a ``Label``/``label`` column in
{RULES, RÈGLES, REGLES, TBM, LLM}. Gold entities absent from the .ann files are
ignored; entity names are matched case-insensitively.

Pipeline
--------
1. Optuna (NSGA-II, objectives = concordance and asymmetric cost-score) on
   OPT_CORPORA, then a held-out before/after report on TEST_CORPORA.
2. Leave-one-corpus-out CV on the final-fit corpora (expected score on an unseen
   corpus).
3. Final fit on ALL_CORPORA (+ OFF_DOMAIN_CORPORA with --all_corpora). The result
   is ALWAYS written to demne_params.json — the file ``demne.params.load_params()``
   reads, so the CLI scorers and the Streamlit dashboard pick it up directly.
   Only the tuned keys are updated; every other key of the JSON is preserved.
   The previous file is backed up in Results/ first.

Speed: raw per-entity features (structure, redundancy, neg/unc counts, freq) are
extracted ONCE per corpus; each trial only re-applies the weight/threshold formulas.
In a source checkout, a drift-guard asserts this fast path equals the dashboard's
MetricsCalculator (parser included) at the current params.

Requires the ``optimize`` extra: ``pip install demne[optimize]``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import sys
import types
from collections import Counter, defaultdict
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

from demne import params as _params_mod
from demne._brat import BratAnnotation, parse_ann_content, read_text
from demne.params import (
    NEGATION_PATTERNS,
    UNCERTAINTY_PATTERNS,
    has_contradiction,
    load_params,
)
from demne.routing import route
from demne.tfidf_coverage import coverage_f1_by_x

if TYPE_CHECKING:
    import optuna

# --------------------------------------------------------------------------- #
# Fixed run settings (deliberately not exposed on the CLI)
# --------------------------------------------------------------------------- #
TRIALS = 1000  # final fit
HOLDOUT_TRIALS = 300  # OPT_CORPORA study evaluated on TEST_CORPORA
CV_TRIALS = 300  # per leave-one-corpus-out fold (0 = skip CV)
SEED = 42
CHUNK_SIZE = 40  # docs per sub-sample for large corpora (None = whole corpus)
CONTEXT_WINDOW = 50  # chars around a mention — same as dashboard BratCorpusParser

_MAX_X = 15  # largest top-X precomputed for the coverage metric
_MAX_NEG_SCOPE = 15  # largest negation scope (in tokens) precomputed
_DEFAULT_NEG_SCOPE = 6  # = MetricsCalculator.NEG_SCOPE_TOKENS

# Folder names under <cwd>/data/
OPT_DIR = "OPT_CORPORA"
TEST_DIR = "TEST_CORPORA"
OFF_DOMAIN_DIR = "OFF_DOMAIN_CORPORA"
ALL_DIR = "ALL_CORPORA"

# Ordered escalation scale: Rules (0) < TBM (1) < LLM (2).
CLASSES = ["RÈGLES", "TBM", "LLM"]
RANK = {c: i for i, c in enumerate(CLASSES)}
_LABEL_MAP = {"RULES": "RÈGLES", "RÈGLES": "RÈGLES", "REGLES": "RÈGLES", "TBM": "TBM", "LLM": "LLM"}

# Source checkout only: the dashboard holds the reference MetricsCalculator.
_REPO_DASHBOARD = Path(__file__).resolve().parents[2] / "dashboard"


class OptimizeParamsError(Exception):
    """User-facing error (missing optuna, bad data/ layout…) — printed, no traceback."""


# --------------------------------------------------------------------------- #
# Corpus discovery
# --------------------------------------------------------------------------- #
def _data_root() -> Path:
    return Path.cwd() / "data"


def _results_dir() -> Path:
    d = Path.cwd() / "Results"
    d.mkdir(exist_ok=True)
    return d


def _gold_csv(corpus_dir: Path) -> Path | None:
    csvs = sorted(corpus_dir.glob("*_routing.csv")) or sorted(corpus_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def discover_corpora(category_dir: Path) -> dict[str, Path]:
    """{corpus_name: path} for every sub-folder of `category_dir` (sorted by name).

    Raises OptimizeParamsError if a sub-folder has no .ann file or no gold CSV.
    """
    if not category_dir.is_dir():
        return {}
    corpora: dict[str, Path] = {}
    problems: list[str] = []
    for d in sorted(p for p in category_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if next(d.rglob("*.ann"), None) is None:
            problems.append(f"{d} : no .ann file (BRAT *.ann + *.txt pairs expected)")
        if _gold_csv(d) is None:
            problems.append(f"{d} : no gold file *_routing.csv / *.csv")
        corpora[d.name] = d
    if problems:
        raise OptimizeParamsError("Invalid corpus folder(s):\n  - " + "\n  - ".join(problems))
    return corpora


def _merge_corpora(*groups: dict[str, Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for g in groups:
        for name, path in g.items():
            if name in out and out[name].resolve() != path.resolve():
                raise OptimizeParamsError(
                    f"Two corpora share the name « {name} » ({out[name]} and {path}): "
                    "rename one of the folders."
                )
            out[name] = path
    return out


def resolve_layout(all_corpora: bool) -> dict[str, Any]:
    """Scan <cwd>/data/ → {"opt", "test", "off", "all", "final"} corpus dicts."""
    root = _data_root()
    opt = discover_corpora(root / OPT_DIR)
    test = discover_corpora(root / TEST_DIR)
    off = discover_corpora(root / OFF_DOMAIN_DIR)
    explicit_all = discover_corpora(root / ALL_DIR)

    required = [
        (OPT_DIR, opt, "target-domain corpora, used for the optimisation"),
        (TEST_DIR, test, "target-domain corpora, held out (final report)"),
    ]
    if all_corpora:
        required.append((OFF_DOMAIN_DIR, off, "off-domain corpora (required by --all_corpora)"))
    missing = [(n, why) for n, found, why in required if not found]
    if missing:
        lines = [f"  {root / n}/<corpus_name>/   ← {why}" for n, why in missing]
        raise OptimizeParamsError(
            "Missing or empty corpus folder(s):\n" + "\n".join(lines) + "\n\n"
            "Each <corpus_name>/ must hold BRAT pairs (*.ann + *.txt) and a "
            "gold file *_routing.csv (columns entity, Label ∈ RULES/TBM/LLM).\n"
            "Voir docs/OPTIMIZE_PARAMS.md."
        )

    all_ = explicit_all or _merge_corpora(opt, test)
    final = _merge_corpora(all_, off) if all_corpora else all_
    return {
        "opt": opt,
        "test": test,
        "off": off,
        "all": all_,
        "final": final,
        "all_explicit": bool(explicit_all),
    }


# BRAT parsing — demne._brat, the same parser the scorers and the dashboard use
# (so it also works after a pip install, where dashboard/ is absent)
# --------------------------------------------------------------------------- #
@dataclass
class _Document:
    text: str
    annotations: list[BratAnnotation]


def parse_brat_dir(path: Path) -> list[_Document]:
    """Read every .ann/.txt pair under `path` (sub-folders included) as documents."""
    files: dict[str, dict[str, str]] = {}
    for root, _, names in os.walk(path):
        for name in names:
            base, ext = os.path.splitext(name)
            if ext.lower() in (".txt", ".ann"):
                files.setdefault(base, {})[ext.lower()] = read_text(Path(root) / name)
    return [
        _Document(
            parts.get(".txt", ""),
            parse_ann_content(parts[".ann"], parts.get(".txt", ""), CONTEXT_WINDOW),
        )
        for parts in files.values()
        if ".ann" in parts
    ]


# --------------------------------------------------------------------------- #
# Gold standard
# --------------------------------------------------------------------------- #
def load_gold(corpus_dir: Path) -> dict[str, str]:
    """Read the *_routing.csv → {entity_lower: normalised_label}."""
    gold_file = _gold_csv(corpus_dir)
    if gold_file is None:
        raise OptimizeParamsError(f"No routing CSV in {corpus_dir}")
    gold: dict[str, str] = {}
    with open(gold_file, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ent = (row.get("entity") or row.get("Entity") or "").strip()
            raw = (row.get("Label") or row.get("label") or "").strip().upper()
            if not ent or raw not in _LABEL_MAP:
                continue
            gold[ent.lower()] = _LABEL_MAP[raw]
    return gold


# --------------------------------------------------------------------------- #
# Raw feature extraction (once per corpus) — mirrors MetricsCalculator formulas
# --------------------------------------------------------------------------- #
def _normalize_pattern(v: str) -> str:
    # Constant form: case-insensitive (→L) + collapse of repeats (LLD==LLLD).
    # Must stay identical to MetricsCalculator (drift-guard). Symbols are kept.
    v = re.sub(r"[A-Za-zÀ-ÖØ-Þß-ÿ]", "L", v)  # letters BEFORE digits (see metrics.py)
    v = re.sub(r"[0-9]", "D", v)
    v = re.sub(r"(.)\1+", r"\1", v)
    return v


def _is_negation_risky(value: str, context: str, scope: int) -> bool:
    """Scope-aware negation — same rule as MetricsCalculator /
    RiskContextScorer.is_negation_risky, the scope being passed explicitly
    (it is calibrated). See their docstring for the rationale."""
    val = (value or "").lower().strip()
    ctx = (context or val).lower()
    if not val:
        return False
    if any(re.search(pat, val) for pat in NEGATION_PATTERNS):
        return False  # negation inside the mention → captured by the rule
    idx = ctx.find(val)
    before = ctx[:idx] if idx >= 0 else ctx
    window = " ".join(before.split()[-scope:])
    return any(re.search(pat, window) for pat in NEGATION_PATTERNS)


def _has_uncertainty(text: str) -> bool:
    return any(re.search(pat, text) for pat in UNCERTAINTY_PATTERNS)


def extract_features(documents: list[_Document], entity_type: str) -> dict | None:
    """Raw per-entity features, computed ONCE per corpus chunk (trials only re-weight them)."""
    annotations = [a for doc in documents for a in doc.annotations if a.entity_type == entity_type]
    values = [a.text.strip() for a in annotations if a.text]
    if not values:
        return None
    contexts = [
        (a.context.lower().strip() if a.context else a.text.lower().strip())
        for a in annotations
        if a.text
    ]

    # Te raw
    patterns = [_normalize_pattern(v) for v in values]
    uniq = set(patterns)
    if len(uniq) <= 1:
        h_norm = 0.0
    else:
        c = Counter(patterns)
        ent = -sum((n / len(patterns)) * math.log(n / len(patterns)) for n in c.values())
        h_norm = ent / math.log(len(uniq))
    structure = 1.0 - h_norm
    has_symbol = any(ch in p for p in uniq for ch in ["%", "+", "-", ">", "<"])
    has_digit = any("D" in p for p in uniq)

    # He raw
    tokens: list[str] = []
    for val in values:
        tokens.extend(w.lower() for w in re.split(r"[^a-zA-Z0-9%]+", val) if w.strip())
    redundancy = (len(tokens) - len(set(tokens))) / len(tokens) if tokens else 0.0

    # R raw — scope-aware negation (mirror of MetricsCalculator.is_negation_risky).
    # A keyword anywhere in the window gives up to 100 % false positives.
    # The scope K is itself calibrated, so the count is precomputed for every K.
    total_texts = len(contexts) if contexts else len(values)
    neg_by_k: dict[int, int] = {
        k: sum(1 for a in annotations if _is_negation_risky(a.text, a.context or "", k))
        for k in range(1, _MAX_NEG_SCOPE + 1)
    }
    unc = sum(1 for t in contexts if _has_uncertainty(t))
    # Contradiction: share of documents holding both a positive and a negative value.
    n_docs = contradicted_docs = 0
    for doc in documents:
        doc_texts = [a.text.lower() for a in doc.annotations if a.entity_type == entity_type]
        if doc_texts:
            n_docs += 1
            contradicted_docs += has_contradiction(doc_texts)

    # Freq raw
    total_tokens = sum(len(doc.text.split()) for doc in documents) or 1
    freq = len(annotations) / max(total_tokens, 1)

    return {
        "structure": structure,
        "has_symbol": has_symbol,
        "has_digit": has_digit,
        "redundancy": redundancy,
        "neg_by_k": neg_by_k,  # {scope K: number of mentions with a risky negation}
        "unc": unc,
        "cont_rate": contradicted_docs / n_docs if n_docs else 0.0,
        "total_texts": total_texts,
        "count": len(annotations),
        "freq": freq,
    }


def metrics_from_features(feat: dict, params: dict) -> dict[str, float]:
    """Te/He/R/Freq/Feas from raw features under `params` (same formulas as MetricsCalculator)."""
    tb = params["templatability_bonus"]
    bonus = tb["symbol_bonus"] if feat["has_symbol"] else 0.0
    if feat["has_digit"] and feat["structure"] > tb["digit_gate"]:
        bonus += tb["digit_bonus"]
    te = min(1.0, max(0.0, feat["structure"] + bonus))

    sig = params["homogeneity_sigmoid"]
    z = -sig["k"] * (feat["redundancy"] - sig["x0"])
    try:
        he = 1 / (1 + math.exp(z))
    except OverflowError:
        he = 0.0 if z > 0 else 1.0
    he = min(1.0, max(0.0, he))

    rw = params["risk_weights"]
    tt = feat["total_texts"]
    k = int(rw.get("scope_tokens", _DEFAULT_NEG_SCOPE))
    neg = feat["neg_by_k"][k]
    r = (
        (neg / tt) * rw["negation"]
        + (feat["unc"] / tt) * rw["uncertainty"]
        + feat["cont_rate"] * rw["contradiction"]
        if tt
        else 0.0
    )
    r = min(1.0, max(0.0, r))

    fw = params["feasibility_weights"]
    sat = fw.get("sat_count", 300)
    feas_freq = min(1.0, feat["count"] / sat) if sat > 0 else 0.0
    feas = min(1.0, max(0.0, fw["alpha_freq"] * feas_freq + fw["beta_he"] * he))

    return {
        "Te": round(te, 4),
        "He": round(he, 4),
        "R": round(r, 4),
        "Freq": round(feat["freq"], 4),
        "Feas": round(feas, 4),
    }


def thresholds_dict(params: dict) -> dict[str, float]:
    """Routing thresholds of `params` under the keys `route()` expects."""
    dt = params["decision_thresholds"]
    return {
        "Te": dt["TE_HIGH"],
        "He": dt["HE_HIGH"],
        "R": dt["R_HIGH"],
        "Feas": dt["FEAS_NER"],
        "Y": dt.get("TFIDF_Y", 0.70),
    }


# --------------------------------------------------------------------------- #
# Dataset build
# --------------------------------------------------------------------------- #
def _chunk_docs(docs: list, chunk_size: int | None, seed: int) -> list[list]:
    """Split docs into shuffled sub-samples of ~chunk_size. Whole corpus if too small."""
    if not chunk_size or len(docs) <= chunk_size:
        return [docs]
    d = list(docs)
    random.Random(seed).shuffle(d)
    chunks = [d[i : i + chunk_size] for i in range(0, len(d), chunk_size)]
    # Fold a tiny trailing remainder into the previous chunk.
    if len(chunks) >= 2 and len(chunks[-1]) < chunk_size / 2:
        chunks[-2].extend(chunks.pop())
    return chunks


@cache
def _corpus_samples(
    name: str, cdir: Path, chunk_size: int | None, seed: int
) -> tuple[tuple[dict, ...], str]:
    """Samples of ONE corpus (cached: OPT/TEST corpora reappear in the final fit)."""
    gold = load_gold(cdir)
    docs = parse_brat_dir(cdir)
    # Coverage-F1: a corpus-level property, computed ONCE and shared by all the
    # chunks of the entity. f1_by_x[ent_lower] = {x: f1} for x=1..MAX_X.
    ment: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        for a in doc.annotations:
            if a.text:
                ment[a.entity_type.lower()].append(a.text.strip())
    cov = coverage_f1_by_x(dict(ment), max_x=_MAX_X)  # {etype_lower: {x: f1}}
    chunks = _chunk_docs(docs, chunk_size, seed)
    samples = []
    for ci, chunk in enumerate(chunks):
        by_lower: dict[str, str] = {}
        for t in {a.entity_type for doc in chunk for a in doc.annotations}:
            by_lower.setdefault(t.lower(), t)  # first casing wins
        for ent_lower, label in gold.items():
            etype = by_lower.get(ent_lower)
            if etype is None:
                continue
            feat = extract_features(chunk, etype)
            if feat is None:
                continue
            samples.append(
                {
                    "corpus": name,
                    "chunk": ci,
                    "entity": etype,
                    "label": label,
                    "feat": feat,
                    "f1_by_x": cov.get(ent_lower, {}),
                }
            )
    stats = f"docs={len(docs):>4}  chunks={len(chunks):>3}  GS={len(gold):>3}  samples={len(samples):>4}"
    return tuple(samples), stats


def build_dataset(
    corpora: dict[str, Path], chunk_size: int | None = None, seed: int = SEED
) -> list[dict]:
    """Samples [{corpus, chunk, entity, label, feat}] for entities present in GS + .ann.

    Large corpora are split into sub-samples (chunk_size docs) so each entity yields
    several data points instead of one aggregate → stronger, more robust signal.
    chunk_size=None keeps each corpus whole (used for the held-out TEST split).
    """
    samples: list[dict] = []
    for name, cdir in corpora.items():
        corpus_samples, stats = _corpus_samples(name, cdir, chunk_size, seed)
        print(f"  {name:<20} {stats}")
        samples.extend(corpus_samples)
    return samples


# --------------------------------------------------------------------------- #
# Scoring — per (dataset, entity), balanced across datasets
# --------------------------------------------------------------------------- #
def _pair_cost_score(true: str, pred: str) -> float:
    """Asymmetric ordinal cost-score for ONE routing (1 = perfect, lower = worse).

    d = rank(pred) − rank(ref) on the Rules<TBM<LLM scale:
      d < 0  under-escalation (method too weak)    → ℓ = 2·|d|
      d > 0  over-escalation  (method too strong)  → ℓ = 1·|d|
      d = 0  correct                               → ℓ = 0
    Score = 1 − ℓ/2.
    """
    d = RANK[pred] - RANK[true]
    loss = 2 * (-d) if d < 0 else (d if d > 0 else 0)
    return 1.0 - loss / 2.0


def routed_predictions(samples: list[dict], params: dict) -> Iterator[tuple[str, str, str, str]]:
    """Yield (corpus, entity, gold label, predicted label) for every sample."""
    th = thresholds_dict(params)
    x = int(params["decision_thresholds"].get("TFIDF_X", 5))
    for s in samples:
        m = metrics_from_features(s["feat"], params)
        # coverage-F1 at the current top-X — without it the TF-IDF branch would be
        # disconnected in the optimizer and X/Y would have no effect.
        res = route(
            te=m["Te"],
            he=m["He"],
            r=m["R"],
            feas=m["Feas"],
            tfidf_score=s["f1_by_x"].get(x, 0.0),
            te_high=th["Te"],
            he_high=th["He"],
            r_high=th["R"],
            feas_ner=th["Feas"],
            tfidf_y=th["Y"],
        )
        pred = _LABEL_MAP.get(res["method"], res["method"])  # "RULES" → "RÈGLES"
        yield s["corpus"], s["entity"], s["label"], pred


def macro_recall(samples: list[dict], params: dict) -> float:
    """BALANCED mean recall per class (RULES/TBM/LLM count equally).

    Counters the concordance bias when RULES is the majority class: "route
    everything to RULES" can no longer inflate the score.
    """
    stat: dict = defaultdict(lambda: [0, 0])  # classe -> [corrects, total]
    for _, _, true, pred in routed_predictions(samples, params):
        stat[true][1] += 1
        if true == pred:
            stat[true][0] += 1
    recs = [cor / tot for cor, tot in stat.values() if tot]
    return mean(recs) if recs else 0.0


def objective_scores(samples: list[dict], params: dict) -> tuple[float, float]:
    """Optimisation objective: (concordance, cost-score), both to be maximised.

    Every entity weighs the same (mean balanced per corpus). `macro_recall` stays
    available for diagnosis but is not optimised: routing aims at the rate of good
    decisions and at the ordered quality of the errors, not at balance between
    methods.
    """
    conc, cost, _, _ = hierarchical_scores(samples, params)
    return conc, cost


def hierarchical_scores(
    samples: list[dict], params: dict
) -> tuple[float, float, dict[str, dict], dict[tuple[str, str], dict]]:
    """Score per (dataset, entity) then mean entities→dataset→global (balanced).

    Every entity weighs the same in its dataset; every dataset weighs the same overall,
    so a large corpus does not dominate the score.

    Returns (global_conc, global_cost, per_dataset, per_entity):
      per_entity[(corpus, entity)] = {label, conc, cost, n, pred}
      per_dataset[corpus]          = {conc, cost, n_entities}
    """
    by_ent: dict = defaultdict(list)  # (corpus, entity) -> [(true, pred), ...]
    for corpus, entity, true, pred in routed_predictions(samples, params):
        by_ent[(corpus, entity)].append((true, pred))

    per_entity = {}
    for key, prs in by_ent.items():
        conc = mean(1.0 if t == p else 0.0 for t, p in prs)  # fraction de chunks corrects
        cost = mean(_pair_cost_score(t, p) for t, p in prs)
        preds = Counter(p for _, p in prs)
        per_entity[key] = {
            "label": prs[0][0],
            "conc": conc,
            "cost": cost,
            "n": len(prs),
            "pred": preds.most_common(1)[0][0],
        }

    by_ds: dict = defaultdict(list)
    for (corpus, _), v in per_entity.items():
        by_ds[corpus].append(v)
    per_dataset = {
        c: {
            "conc": mean(x["conc"] for x in v),
            "cost": mean(x["cost"] for x in v),
            "n_entities": len(v),
        }
        for c, v in by_ds.items()
    }
    global_conc = mean(d["conc"] for d in per_dataset.values()) if per_dataset else 0.0
    global_cost = mean(d["cost"] for d in per_dataset.values()) if per_dataset else 0.0
    return global_conc, global_cost, per_dataset, per_entity


# --------------------------------------------------------------------------- #
# Optuna search space
# --------------------------------------------------------------------------- #
PARAM_SPACE = {
    # Decision thresholds — wide (these drive every routing decision).
    # Lower bound 0.10: Te does not separate RULES/TBM, so the optimizer would push it
    # to 0 (inactive gate, illogical). 0.10 remains a real gate for ~0 cost (see sweep).
    "TE_HIGH": (0.05, 0.90),
    # Lower bound kept high: He only discriminates RÈGLES (≈0.99) from TBM (≈0.90)
    # near the top of its range — a low HE_HIGH lets every class into the RÈGLES branch.
    "HE_HIGH": (0.3, 0.99),
    # Recalibrated for scope-aware negation: R no longer exceeds ~0.10-0.25
    # (against 0.5-0.8 with the former noisy count). A range starting at 0.20
    # left the threshold above the observed maximum → inert R node.
    "R_HIGH": (0.5, 0.9),
    "FEAS_NER": (0, 0.80),
    # Extractability branch (top-keyword coverage): X = number of top keywords (step 1);
    # Y = F1 threshold → RULES path.
    # Lower bound Y=0.5: guard against RULES over-routing (otherwise Y→low exploits the
    # class imbalance and makes the LLM recall collapse).
    # Lower bound 3: the metric measures coverage by a SET of keywords. With X=1 it
    # degenerates into "is the single most discriminant word enough?", which is no
    # longer what we want to measure (a rule uses several terms), even if the optimizer
    # converges there.
    #
    # The other TF-IDF keys of demne_params.json stay FIXED, because none of them acts
    # on the routing (checked by searching src/, dashboard/ and main.py):
    #   TFIDF_SIM           clustering threshold of the former E_tfidf metric; the
    #                       coverage-F1 has no clustering (see tfidf_coverage.py).
    #                       Only stored by DecisionTreeBuilder, never read.
    #   TFIDF_WINDOW_TOKENS context window of that same former metric; read nowhere.
    #   TFIDF_USE_F1        read nowhere: the coverage is always an F1.
    # Varying them would not change the objective; Optuna would write random values
    # into the JSON.
    "TFIDF_X": (3, 15),
    "TFIDF_Y": (0.5, 1.0),
    # Feasibility weights — wide.
    "alpha_freq": (0.0, 1.0),
    "beta_he": (0.0, 1.0),
    # Saturating count for Feas: a "sufficient" number of occurrences for a template.
    # step=10 (entier lisible), pas 0.01.
    "sat_count": (50, 1000),
    # Homogeneity sigmoid — narrow: shape is well understood (polarisation around
    # redundancy≈0.5), only fine-tuned, not searched from scratch.
    "k": (6.0, 14.0),
    "x0": (0.35, 0.65),
    # Templatability bonuses — wide.
    "symbol_bonus": (0.0, 0.50),
    "digit_bonus": (0.0, 0.50),
    "digit_gate": (0.0, 1.0),
    # Risk weights — wide.
    "neg_weight": (0.0, 1.0),
    "unc_weight": (0.0, 1.0),
    # Negation scope (tokens before the mention): an arbitrary value, hence calibrated
    # like the rest instead of hard-coded. Integer.
    "neg_scope": (1, 15),
}

# demne_params.json keys written back after a run — everything else is preserved.
TUNED_KEYS: dict[str, tuple[str, ...]] = {
    "decision_thresholds": ("TE_HIGH", "HE_HIGH", "R_HIGH", "FEAS_NER", "TFIDF_X", "TFIDF_Y"),
    "risk_weights": ("negation", "uncertainty", "scope_tokens"),
    "feasibility_weights": ("alpha_freq", "beta_he", "sat_count"),
    "homogeneity_sigmoid": ("k", "x0"),
    "templatability_bonus": ("symbol_bonus", "digit_bonus", "digit_gate"),
}


def suggest_params(trial: optuna.Trial) -> dict:
    """Sample one value per PARAM_SPACE entry for this Optuna trial."""
    # step=0.01 → thresholds/weights stay readable (0.32, not 0.3188127789…).
    # sat_count is a count → integer step of 10.
    out = {}
    _int_step = {"sat_count": 10, "TFIDF_X": 1, "neg_scope": 1}
    for n, (lo, hi) in PARAM_SPACE.items():
        out[n] = trial.suggest_float(n, lo, hi, step=_int_step.get(n, 0.01))
    return out


def apply_params(base: dict, vals: dict) -> dict:
    """Copy of `base` with the sampled tunables written in (other keys untouched)."""
    p = deepcopy(base)
    p["decision_thresholds"].update(
        TE_HIGH=vals["TE_HIGH"],
        HE_HIGH=vals["HE_HIGH"],
        R_HIGH=vals["R_HIGH"],
        FEAS_NER=vals["FEAS_NER"],
        TFIDF_X=int(vals["TFIDF_X"]),
        TFIDF_Y=vals["TFIDF_Y"],
    )
    p["feasibility_weights"].update(
        alpha_freq=vals["alpha_freq"],
        beta_he=vals["beta_he"],
        sat_count=vals["sat_count"],
    )
    p["homogeneity_sigmoid"].update(k=vals["k"], x0=vals["x0"])
    p["templatability_bonus"].update(
        symbol_bonus=vals["symbol_bonus"],
        digit_bonus=vals["digit_bonus"],
        digit_gate=vals["digit_gate"],
    )
    p["risk_weights"].update(
        negation=vals["neg_weight"],
        uncertainty=vals["unc_weight"],
        scope_tokens=int(vals["neg_scope"]),
    )
    return p


# --------------------------------------------------------------------------- #
# Studies & reports
# --------------------------------------------------------------------------- #
def _require_optuna() -> types.ModuleType:
    try:
        import optuna
    except ImportError as exc:
        raise OptimizeParamsError(
            'optuna is not installed. Install the extra:  pip install "demne[optimize]"\n'
            '(from the repository: pip install -e ".[optimize]")'
        ) from exc
    return optuna


def report(
    name: str, samples: list[dict], params: dict, detail: bool = True
) -> tuple[float, float]:
    """Print per-dataset (and optionally per-entity) scores; return (concordance, cost-score)."""
    gconc, gcost, per_ds, per_ent = hierarchical_scores(samples, params)
    print(
        f"\n=== {name} : concordance={gconc:.3f}  cost-score={gcost:.3f} "
        f"(balanced mean per dataset) ==="
    )
    print(f"  {'dataset':<20} {'#ent':>4} {'concordance':>12} {'cost-score':>11}")
    for c, d in sorted(per_ds.items()):
        print(f"  {c:<20} {d['n_entities']:>4} {d['conc']:>12.3f} {d['cost']:>11.3f}")
    if detail:
        print(
            f"\n  {'dataset':<20} {'entity':<34} {'gold':<6} {'pred':<6} {'ok':>5} {'cost':>5}  n"
        )
        for (corpus, ent), v in sorted(per_ent.items(), key=lambda kv: (kv[0][0], -kv[1]["conc"])):
            ok = (
                "✓"
                if v["conc"] >= 0.999
                else ("✗" if v["conc"] <= 1e-9 else f"{v['conc']*100:.0f}%")
            )
            mark = "" if v["label"] == v["pred"] else "  ⟵"
            print(
                f"  {corpus[:20]:<20} {ent[:34]:<34} {v['label']:<6} {v['pred']:<6} {ok:>5} "
                f"{v['cost']:>5.2f} {v['n']:>2}{mark}"
            )
    return gconc, gcost


def run_study(
    samples: list[dict],
    base: dict,
    *,
    seed: int,
    n_trials: int,
    study_name: str,
    storage: str | None = None,
    show_pareto: bool = False,
) -> dict:
    """Optuna multi-objective study on `samples` → best params (dict).

    Final pick = highest concordance, cost-score as tiebreak.
    """
    optuna = _require_optuna()

    def objective(trial: Any) -> tuple[float, float]:
        return objective_scores(samples, apply_params(base, suggest_params(trial)))

    study = optuna.create_study(
        directions=["maximize", "maximize"],  # (concordance, cost-score)
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    pareto = study.best_trials
    if show_pareto:
        print(f"\nPareto front ({len(pareto)} solutions)  [concordance, cost-score]:")
        for t in sorted(pareto, key=lambda t: (-t.values[0], -t.values[1])):
            print(f"  concordance={t.values[0]:.3f}  cost-score={t.values[1]:.3f}")
    chosen = max(pareto, key=lambda t: (t.values[0], t.values[1]))
    return apply_params(base, chosen.params)


def loco_cv(all_samples: list[dict], base: dict, *, seed: int, n_trials: int) -> dict | None:
    """Leave-one-corpus-out CV : optimise on N−1 corpora, score the held-out one.

    Honest out-of-domain generalisation : each corpus takes a turn as unseen
    validation. Fresh in-memory study per fold (no shared storage).
    """
    corpora = sorted({s["corpus"] for s in all_samples})
    if len(corpora) < 2:
        print("\nLeave-one-corpus-out CV skipped (a single corpus in the final fit).")
        return None
    print(f"\n################  LEAVE-ONE-CORPUS-OUT CV ({n_trials} trials/fold)  ################")
    print(f"  {'held-out':<20} {'#ent':>4} {'concordance':>12} {'cost-score':>11}")
    folds = {}
    for c in corpora:
        train = [s for s in all_samples if s["corpus"] != c]
        held = [s for s in all_samples if s["corpus"] == c]
        best = run_study(train, base, seed=seed, n_trials=n_trials, study_name=f"loco_{c}")
        hc, hco, _, per_ent = hierarchical_scores(held, best)
        folds[c] = {"concordance": hc, "cost_score": hco, "n_entities": len(per_ent)}
        print(f"  {c[:20]:<20} {len(per_ent):>4} {hc:>12.3f} {hco:>11.3f}")
    m_conc = mean(f["concordance"] for f in folds.values())
    m_cost = mean(f["cost_score"] for f in folds.values())
    print(f"  {'MEAN':<20} {'':>4} {m_conc:>12.3f} {m_cost:>11.3f}")
    return {"folds": folds, "mean_concordance": m_conc, "mean_cost_score": m_cost}


# --------------------------------------------------------------------------- #
# demne_params.json write-back
# --------------------------------------------------------------------------- #
def params_file() -> Path:
    """The demne_params.json that load_params() reads (CLI, scorers, dashboard)."""
    found = _params_mod._find_params_file()
    return (
        found or Path(_params_mod.__file__).resolve().parent / "data" / _params_mod.PARAMS_FILENAME
    )


def _tunable_view(p: dict) -> dict:
    return {sect: {k: p[sect][k] for k in keys} for sect, keys in TUNED_KEYS.items()}


def _clean(v: Any) -> Any:
    """Round floats (recursively) so float artefacts like 0.5700000000000001 become 0.57."""
    if isinstance(v, float):
        return round(v, 10)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def write_params(path: Path, best: dict) -> None:
    """Update ONLY the tuned keys in `path`; every other key/section is preserved."""
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    out = deepcopy(raw)
    for sect, keys in TUNED_KEYS.items():
        target = out.setdefault(sect, {})
        for k in keys:
            target[k] = _clean(best[sect][k])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: write a sibling temp file, then replace — never a half-written JSON.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(out), indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Drift guards (source checkout only)
# --------------------------------------------------------------------------- #
def _load_dashboard_core() -> tuple[Any, Any] | None:
    """(BratCorpusParser, MetricsCalculator) from dashboard/core, or None after a pip install.

    Loaded by file location under a private name, so neither sys.path nor Streamlit is touched.
    """
    core_dir = _REPO_DASHBOARD / "core"
    if not (core_dir / "metrics.py").exists():
        return None
    name = "_demne_dashboard_core"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, core_dir / "__init__.py", submodule_search_locations=[str(core_dir)]
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    return module.BratCorpusParser, module.MetricsCalculator


def _drift_guard(params: dict, corpus_dir: Path) -> None:
    """Fast path (local parser + features) == dashboard parser + MetricsCalculator."""
    core = _load_dashboard_core()
    if core is None:
        print("drift-guard skipped (dashboard/ absent: pip installation)")
        return
    parser_cls, calc_cls = core
    calc = calc_cls(params)
    ref_docs = parser_cls().parse_directory(str(corpus_dir))
    fast_docs = parse_brat_dir(corpus_dir)
    by_lower: dict[str, str] = {}
    for t in {a.entity_type for doc in fast_docs for a in doc.annotations}:
        by_lower.setdefault(t.lower(), t)
    worst = 0.0
    for ent_lower in load_gold(corpus_dir):
        etype = by_lower.get(ent_lower)
        if not etype:
            continue
        feat = extract_features(fast_docs, etype)
        if feat is None:
            continue
        ref = calc.compute_all_metrics(ref_docs, etype)
        fast = metrics_from_features(feat, params)
        for key in ("Te", "He", "R", "Freq", "Feas"):
            worst = max(worst, abs(ref[key] - fast[key]))
    flag = "OK" if worst <= 1e-6 else "⚠️  DRIFT"
    print(f"drift-guard {flag} on {corpus_dir.name} (max |Δ| = {worst:.1e} vs MetricsCalculator)")
    _cli_scorer_guard(calc)


def _cli_scorer_guard(ref_calc: Any) -> None:
    """Check that the CLI scorers (src/demne/*.py) implement the SAME formulas as
    MetricsCalculator.

    `main.py` routes on the CSVs written by these scorers while the dashboard goes
    through MetricsCalculator: without this guard the two paths could silently diverge
    and route differently.
    """
    checks: list[tuple[str, bool, str]] = []

    # 1. Te — normalisation de motif (casse-insensible + collapse)
    try:
        from demne.templatability import TemplatabilityScorer as _Ts

        # __init__ requires a corpus → test the method without instantiating.
        _ts = _Ts.__new__(_Ts)
        diffs = [
            (s, _Ts.normalize_pattern(_ts, s), _normalize_pattern(s))
            for s in ("HER2 3+", "her2 3+", "SEIN   DROIT", "Ki 67 10%")
        ]
        bad_s = [(s, a, b) for s, a, b in diffs if a != b]
        checks.append(
            (
                "templatability.normalize_pattern",
                not bad_s,
                "; ".join(f"{s!r}: {a!r} vs {b!r}" for s, a, b in bad_s),
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(("templatability.normalize_pattern", False, f"cannot be checked: {exc}"))

    # 2. R — scope-aware negation + identical lexicons
    try:
        from demne.risk_context import RiskContextScorer

        checks.append(
            (
                "risk_context.is_negation_risky",
                hasattr(RiskContextScorer, "is_negation_risky"),
                "method missing",
            )
        )
        cli = RiskContextScorer()  # no corpus is loaded by the constructor
        for name in ("NEGATION_PATTERNS", "UNCERTAINTY_PATTERNS"):
            ours, ref = getattr(cli, name), getattr(ref_calc, name)
            checks.append((f"risk_context.{name}", ours == ref, "lexiques divergents"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("risk_context", False, f"cannot be checked: {exc}"))

    # 3. Feas — saturating count (not the density)
    try:
        src = (Path(__file__).resolve().parent / "ner_feasibility.py").read_text(encoding="utf-8")
        checks.append(("ner_feasibility.sat_count", "sat_count" in src, "still uses the density"))
    except OSError as exc:
        checks.append(("ner_feasibility.sat_count", False, f"cannot be checked: {exc}"))

    bad = [(n, d) for n, ok, d in checks if not ok]
    if bad:
        print("⚠️  DRIFT in the CLI scorers (main.py will route differently from the dashboard):")
        for name, detail in bad:
            print(f"     - {name} : {detail}")
    else:
        print(f"cli-scorer-guard OK ({len(checks)} formulas aligned with MetricsCalculator)")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _print_layout(layout: dict, all_corpora: bool) -> None:
    root = _data_root()
    print(f"Corpora detected under {root}:")
    rows = [(OPT_DIR, layout["opt"]), (TEST_DIR, layout["test"])]
    if all_corpora:
        rows.append((OFF_DOMAIN_DIR, layout["off"]))
    elif layout["off"]:
        rows.append((OFF_DOMAIN_DIR + " (ignored without --all_corpora)", layout["off"]))
    rows.append(
        (ALL_DIR if layout["all_explicit"] else f"{ALL_DIR} (auto: OPT ∪ TEST)", layout["all"])
    )
    for label, corpora in rows:
        print(f"  {label:<45} {', '.join(corpora) or '—'}")
    print(f"  {'→ final fit on':<45} {', '.join(layout['final'])}")


def run(all_corpora: bool) -> dict:
    """Full recalibration; writes demne_params.json. Returns the run summary."""
    optuna = _require_optuna()
    mode = "all_corpora" if all_corpora else "in_domain"
    layout = resolve_layout(all_corpora)
    _print_layout(layout, all_corpora)

    pfile = params_file()
    base = load_params()
    print(f"\nCurrent parameters: {pfile}")

    print(f"\nExtraction des features (chunk_size={CHUNK_SIZE}) :")
    print(f" {OPT_DIR}")
    opt_samples = build_dataset(layout["opt"], chunk_size=CHUNK_SIZE)
    print(f" {TEST_DIR} (corpus entiers, held-out)")
    test_samples = build_dataset(layout["test"], chunk_size=None)
    print(" Fit final")
    final_samples = build_dataset(layout["final"], chunk_size=CHUNK_SIZE)
    if not final_samples:
        raise OptimizeParamsError(
            "No gold entity was found in the .ann files: check that the "
            "entity column of the CSV matches the BRAT entity types."
        )

    print()
    _drift_guard(base, next(iter(layout["opt"].values())))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 1. OPT → held-out TEST
    print(f"\n################  OPT → TEST held-out ({HOLDOUT_TRIALS} trials)  ################")
    test_before = report("TEST/current parameters", test_samples, base, detail=False)
    opt_best = run_study(
        opt_samples, base, seed=SEED, n_trials=HOLDOUT_TRIALS, study_name="holdout"
    )
    test_after = report("TEST/optimised on OPT", test_samples, opt_best)

    # 2. Leave-one-corpus-out on the final-fit corpora
    loco = loco_cv(final_samples, base, seed=SEED, n_trials=CV_TRIALS) if CV_TRIALS else None

    # 3. Final fit — persisted Optuna study (one per run, never mixed across corpora)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = _results_dir()
    storage = f"sqlite:///{(results / 'optuna_demne.db').as_posix()}"
    print(f"\n################  FIT FINAL ({mode}, {TRIALS} trials)  ################")
    final_before = report("FINAL FIT/current parameters", final_samples, base, detail=False)
    best = run_study(
        final_samples,
        base,
        seed=SEED,
        n_trials=TRIALS,
        study_name=f"demne_{mode}_{stamp}",
        storage=storage,
        show_pareto=True,
    )
    final_after = report("FINAL FIT/optimised", final_samples, best)
    print(
        f"  macro-recall (diagnostic) : {macro_recall(final_samples, base):.3f} → "
        f"{macro_recall(final_samples, best):.3f}"
    )

    # Write-back (always) — previous file backed up first.
    backup = results / f"demne_params_before_{stamp}.json"
    if pfile.exists():
        backup.write_text(pfile.read_text(encoding="utf-8"), encoding="utf-8")
    write_params(pfile, best)

    summary = {
        "mode": mode,
        "timestamp": stamp,
        "corpora": {k: sorted(layout[k]) for k in ("opt", "test", "off", "all", "final")},
        "settings": {
            "trials": TRIALS,
            "holdout_trials": HOLDOUT_TRIALS,
            "cv_trials": CV_TRIALS,
            "seed": SEED,
            "chunk_size": CHUNK_SIZE,
        },
        "holdout_test": {"before": test_before, "after": test_after},
        "final_fit": {"before": final_before, "after": final_after},
        "loco": loco,
        "params_before": _tunable_view(base),
        "params_after": {
            s: {k: _clean(v) for k, v in d.items()} for s, d in _tunable_view(best).items()
        },
    }
    report_file = results / f"optimize_params_{mode}_{stamp}.json"
    report_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print(f"{'':<28}{'concordance':>20}{'cost-score':>20}")
    print(f"{'':<28}{'before → after':>20}{'before → after':>20}")
    for label, (b, a) in (
        ("Held-out TEST (OPT→TEST)", (test_before, test_after)),
        ("Fit final", (final_before, final_after)),
    ):
        print(f"{label:<28}{f'{b[0]:.3f} → {a[0]:.3f}':>20}{f'{b[1]:.3f} → {a[1]:.3f}':>20}")
    if loco:
        print(
            f"{'Leave-one-corpus-out':<28}{loco['mean_concordance']:>20.3f}"
            f"{loco['mean_cost_score']:>20.3f}"
        )
    print("=" * 72)
    if final_after[0] < final_before[0] - 1e-9:
        print(
            "⚠️  The final fit is WORSE than the previous parameters "
            f"({final_before[0]:.3f} → {final_after[0]:.3f}): too few trials or corpora "
            "too small. The file is updated anyway; to go back, "
            "copy back the backup listed below."
        )
    print(f"✅ demne_params.json updated: {pfile}")
    print(f"   previous version backed up: {backup}")
    print(f"   report: {report_file}")
    print("   Thresholds, weights and presets (CALIBRATED/FRUGAL/QUALITY) are read from this file")
    print("   by main.py, the scorers, the Streamlit dashboard and the GUI. Left to do:")
    print("     1. re-run  python main.py evaluate --gs_dir <corpus>  : the metrics and")
    print("        data/decision_config.json date from the old weights;")
    print("     2. restart the Streamlit dashboard if it is open (parameters are read at startup).")
    return summary


def add_mode_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the mutually exclusive --in_domain / --all_corpora flags."""
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--in_domain",
        action="store_true",
        help="optimise on OPT_CORPORA, held-out TEST_CORPORA, final fit on ALL_CORPORA",
    )
    g.add_argument(
        "--all_corpora",
        action="store_true",
        help="same, the final fit also includes OFF_DOMAIN_CORPORA",
    )


def main(argv: list[str] | None = None) -> int:
    """`python -m demne.optimize_params`; returns the process exit code."""
    ap = argparse.ArgumentParser(description="Recalibrate demne_params.json (Optuna NSGA-II)")
    add_mode_arguments(ap)
    args = ap.parse_args(argv)
    try:
        run(all_corpora=args.all_corpora)
    except OptimizeParamsError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
