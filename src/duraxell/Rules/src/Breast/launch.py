from pprint import pprint

from biomarker_brat_annotator import (_parse_her2_ihc_value,
                                      annotate_txt_folder_new_scheme,
                                      annotate_txt_folder_regex_only)
from eco2ai import Tracker, set_params




import os
import sys

# Calcul du chemin racine du projet pour obtenir les vrais chemins de données
# _current_file = src/duraxell/Rules/src/Breast/lunch.py
_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../duraxell"))

# New scheme (with Value attributes)
annotate_txt_folder_new_scheme(
    os.path.join(_base_dir, "Breast/RCP/training_set_breast_cancer/"),
    out_dir=os.path.join(_base_dir, "Breast/RCP/training_set_breast_cancer_pred/"),
    recursive=True,
)


annotate_txt_folder_new_scheme(
    os.path.join(_base_dir, "Breast/RCP/evaluation_set_breast_cancer_GS/"),
    out_dir=os.path.join(_base_dir, "Breast/RCP/evaluation_set_breast_cancer_pred_rules/"),
    recursive=True,
)

annotate_txt_folder_new_scheme(
    os.path.join(_base_dir, "Breast/CHIR/evaluation_set_breast_cancer_chir_GS/"),
    out_dir=os.path.join(_base_dir, "Breast/CHIR/evaluation_set_breast_cancer_chir_pred_rules/"),
    recursive=True,
)


def evaluate_ann_folders(
    gold_dir,
    pred_dir,
    *,
    recursive=False,
    overlap_metric: str = "iou",  # "iou", "min", "gold", "pred", "exact"
    overlap_threshold: float = 0.90,
    collect_errors: bool = False,
    snippet_chars: int = 40,  # chars of context on each side if collect_errors=True
):
    """
    Compare BRAT .ann folders and compute metrics.
    Also (optionally) collect all False Positives / False Negatives with file/label/span/snippet.

    Labels: Estrogen_receptor, Progesterone_receptor, HER2_status, Ki67, FISH
    Matching:
      - "exact": label + exact [start,end]
      - "iou":   IoU(g,p) >= threshold
      - "min":   overlap / min(len(g),len(p)) >= threshold
      - "gold":  overlap / len(g) >= threshold
      - "pred":  overlap / len(p) >= threshold
    """
    import os
    import re
    from collections import Counter, defaultdict

    LABELS = (
        "Estrogen_receptor",
        "Progesterone_receptor",
        "HER2_status",
        "HER2_IHC",
        "Ki67",
        "HER2_FISH",
    )

    def _list_anns(folder):
        paths = {}
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith(".ann"):
                    rel = (
                        os.path.relpath(os.path.join(root, f), start=folder)
                        if recursive
                        else f
                    )
                    paths[rel] = os.path.join(root, f)
            if not recursive:
                break
        return paths

    def _read_T_spans(path):
        """
        Parse T-lines only:
          Tn\t<label> start end\t<text>
        Returns Counter keyed by (label, start, end).
        Discontinuous spans ('start end; start end') -> take the first segment.
        """
        c = Counter()
        if not os.path.exists(path):
            return c
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not line.startswith("T"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                m = re.match(r"^(\S+)\s+(\d+)\s+(\d+)", parts[1])
                if not m:
                    m = re.match(r"^(\S+)\s+(\d+)\s+(\d+);", parts[1])
                if not m:
                    continue
                label, s, e = m.group(1), int(m.group(2)), int(m.group(3))
                if label in LABELS:
                    c[(label, s, e)] += 1
        return c

    def _overlap_len(a, b):
        (s1, e1), (s2, e2) = a, b
        return max(0, min(e1, e2) - max(s1, s2))

    def _score_span(g_span, p_span, metric: str):
        """Return overlap score in [0,1] according to metric."""
        if metric == "exact":
            return 1.0 if g_span == p_span else 0.0
        inter = _overlap_len(g_span, p_span)
        glen = g_span[1] - g_span[0]
        plen = p_span[1] - p_span[0]
        if glen <= 0 or plen <= 0:
            return 0.0
        if metric == "iou":
            denom = glen + plen - inter
            return inter / denom if denom > 0 else 0.0
        elif metric == "min":
            return inter / min(glen, plen)
        elif metric == "gold":
            return inter / glen
        elif metric == "pred":
            return inter / plen
        else:
            raise ValueError(f"Unknown overlap_metric: {metric}")

    def _match_and_residuals(g: Counter, p: Counter, metric: str, thr: float):
        """
        Greedy one-to-one matching across duplicate spans.
        Returns:
          tp, fp, fn,
          rem_g (Counter of unmatched gold (start,end)),
          rem_p (Counter of unmatched pred (start,end)),
          matches (list of ((gs),(ps), n))
        """
        # exact fast path
        if metric == "exact" or thr >= 1.0:
            tp = sum(min(g[k], p[k]) for k in set(g) | set(p))
            rem_g = Counter()
            rem_p = Counter()
            for k, n in g.items():
                rem = n - min(n, p.get(k, 0))
                if rem > 0:
                    rem_g[k] = rem
            for k, n in p.items():
                rem = n - min(n, g.get(k, 0))
                if rem > 0:
                    rem_p[k] = rem
            fp = sum(rem_p.values())
            fn = sum(rem_g.values())
            return tp, fp, fn, rem_g, rem_p, []

        # Build all candidate pairs with score >= thr
        pairs = []
        for gs in g.keys():
            for ps in p.keys():
                sc = _score_span(gs, ps, metric)
                if sc >= thr:
                    pairs.append((sc, gs, ps))
        pairs.sort(
            key=lambda t: (-t[0], t[1][1] - t[1][0], t[2][1] - t[2][0])
        )  # best first

        rem_g = dict(g)
        rem_p = dict(p)
        matches = []
        tp = 0
        for sc, gs, ps in pairs:
            if rem_g.get(gs, 0) <= 0 or rem_p.get(ps, 0) <= 0:
                continue
            n = min(rem_g[gs], rem_p[ps])
            tp += n
            rem_g[gs] -= n
            rem_p[ps] -= n
            matches.append((gs, ps, n, sc))

        # convert dicts back to Counters with only positives
        rem_g = Counter({k: v for k, v in rem_g.items() if v > 0})
        rem_p = Counter({k: v for k, v in rem_p.items() if v > 0})
        fp = sum(rem_p.values())
        fn = sum(rem_g.values())
        return tp, fp, fn, rem_g, rem_p, matches

    def _read_txt(rel_path_from_gold_root):
        # try read .txt next to gold .ann for snippets
        try:
            txt_path = os.path.splitext(gold_files[rel_path_from_gold_root])[0] + ".txt"
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        except Exception:
            return ""

    gold_files = _list_anns(gold_dir)
    pred_files = _list_anns(pred_dir)

    per_label_counts = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in LABELS}
    errors = {"false_positives": [], "false_negatives": []} if collect_errors else None

    for rel, gold_path in gold_files.items():
        pred_path = pred_files.get(rel, None)
        gold = _read_T_spans(gold_path)
        pred = _read_T_spans(pred_path) if pred_path else Counter()

        # split by label
        gold_by_lbl = defaultdict(Counter)
        pred_by_lbl = defaultdict(Counter)
        for (lbl, s, e), n in gold.items():
            gold_by_lbl[lbl][(s, e)] += n
        for (lbl, s, e), n in pred.items():
            pred_by_lbl[lbl][(s, e)] += n

        # optional text for snippets (from gold side)
        text = (
            _read_txt(rel)
            if collect_errors and snippet_chars and rel in gold_files
            else ""
        )

        for lbl in LABELS:
            g = gold_by_lbl.get(lbl, Counter())
            p = pred_by_lbl.get(lbl, Counter())

            tp, fp, fn, rem_g, rem_p, _ = _match_and_residuals(
                g, p, overlap_metric, overlap_threshold
            )

            per_label_counts[lbl]["tp"] += tp
            per_label_counts[lbl]["fp"] += fp
            per_label_counts[lbl]["fn"] += fn

            if collect_errors:
                # Expand counters into per-instance rows, preserving multiplicity
                def _snippet(s, e):
                    if not text:
                        return ""
                    lo = max(0, s - snippet_chars)
                    hi = min(len(text), e + snippet_chars)
                    return text[lo:hi].replace("\n", " ")

                for (s, e), n in rem_p.items():
                    for _ in range(n):
                        errors["false_positives"].append(
                            {
                                "file": rel,
                                "label": lbl,
                                "start": s,
                                "end": e,
                                "text": _snippet(s, e),
                            }
                        )
                for (s, e), n in rem_g.items():
                    for _ in range(n):
                        errors["false_negatives"].append(
                            {
                                "file": rel,
                                "label": lbl,
                                "start": s,
                                "end": e,
                                "text": _snippet(s, e),
                            }
                        )

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    # per-label metrics
    per_label = {}
    micro_tp = micro_fp = micro_fn = 0
    for lbl, d in per_label_counts.items():
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        p, r, f1 = _prf(tp, fp, fn)
        per_label[lbl] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    P, R, F1 = _prf(micro_tp, micro_fp, micro_fn)
    micro = {
        "tp": micro_tp,
        "fp": micro_fp,
        "fn": micro_fn,
        "precision": P,
        "recall": R,
        "f1": F1,
    }

    macro_p = sum(per_label[l]["precision"] for l in LABELS) / len(LABELS)
    macro_r = sum(per_label[l]["recall"] for l in LABELS) / len(LABELS)
    macro_f = sum(per_label[l]["f1"] for l in LABELS) / len(LABELS)
    macro = {"precision": macro_p, "recall": macro_r, "f1": macro_f}

    out = {"per_label": per_label, "micro": micro, "macro": macro}
    if collect_errors:
        out["errors"] = errors
    return out


resRCPrules = evaluate_ann_folders(
    os.path.join(_base_dir, "Breast/RCP/evaluation_set_breast_cancer_GS"),
    os.path.join(_base_dir, "Breast/RCP/evaluation_set_breast_cancer_pred_rules"),
    recursive=True,
    overlap_metric="min",
    overlap_threshold=0.80,
    collect_errors=True,
    snippet_chars=100,
)

# pprint(resRCPrules["per_label"])

resRCPner = evaluate_ann_folders(
    os.path.join(_base_dir, "Breast/RCP/evaluation_set_breast_cancer_GS"),
    os.path.join(_base_dir, "Breast/RCP/evaluation_set_breast_cancer_pred_ner"),
    recursive=True,
    overlap_metric="min",
    overlap_threshold=0.80,
    collect_errors=True,
    snippet_chars=100,
)


resRCPllm = evaluate_ann_folders(
    os.path.join(_base_dir, "Breast/RCP/evaluation_set_breast_cancer_GS"),
    os.path.join(_base_dir, "Breast/RCP/evaluation_set_breast_cancer_pred_Mistral8x7b"),
    recursive=True,
    overlap_metric="min",
    overlap_threshold=0.20,
    collect_errors=True,
    snippet_chars=100,
)

resRCPllm_finetunned = evaluate_ann_folders(
    os.path.join(_base_dir, "Breast/RCP/evaluation_set_breast_cancer_GS"),
    os.path.join(_base_dir, "Breast/RCP/evaluation_set_breast_cancer_pred_Mistral8x7b_finetunned"),
    recursive=True,
    overlap_metric="min",
    overlap_threshold=0.20,
    collect_errors=True,
    snippet_chars=100,
)

# pprint(resRCPner["per_label"])

resCHIRrules = evaluate_ann_folders(
    os.path.join(_base_dir, "Breast/CHIR/evaluation_set_breast_cancer_chir_GS"),
    os.path.join(_base_dir, "Breast/CHIR/evaluation_set_breast_cancer_chir_pred_rules"),
    recursive=True,
    overlap_metric="min",
    overlap_threshold=0.80,
    collect_errors=True,
    snippet_chars=100,
)

# pprint(resCHIRrules["per_label"])

resCHIRner = evaluate_ann_folders(
    os.path.join(_base_dir, "Breast/CHIR/evaluation_set_breast_cancer_chir_GS"),
    os.path.join(_base_dir, "Breast/CHIR/evaluation_set_breast_cancer_chir_pred_ner"),
    recursive=True,
    overlap_metric="min",
    overlap_threshold=0.80,
    collect_errors=True,
    snippet_chars=100,
)


resCHIRllm = evaluate_ann_folders(
    os.path.join(_base_dir, "Breast/CHIR/evaluation_set_breast_cancer_chir_GS"),
    os.path.join(_base_dir, "Breast/CHIR/evaluation_set_breast_cancer_chir_pred_Mistral8x7b"),
    recursive=True,
    overlap_metric="min",
    overlap_threshold=0.80,
    collect_errors=True,
    snippet_chars=100,
)

resCHIRllm_finetunned = evaluate_ann_folders(
    os.path.join(_base_dir, "Breast/CHIR/evaluation_set_breast_cancer_chir_GS"),
    os.path.join(_base_dir, "Breast/CHIR/evaluation_set_breast_cancer_chir_pred_Mistral8x7b_finetunned"),
    recursive=True,
    overlap_metric="min",
    overlap_threshold=0.80,
    collect_errors=True,
    snippet_chars=100,
)

print("RCP _ GS vs RULES", resRCPrules["micro"])

print("RCP _ GS vs NER", resRCPner["micro"])

print("RCP _ GS vs LLM", resRCPllm["micro"])


print("RCP _ GS vs LLM_finetunned", resRCPllm_finetunned["micro"])

print("CHIR _ GS vs RULES", resCHIRrules["micro"])

print("CHIR _ GS vs NER", resCHIRner["micro"])


print("CHIR _ GS vs LLM", resCHIRllm["micro"])

print("CHIR _ GS vs LLM_finetunned", resCHIRllm_finetunned["micro"])

# pprint(resCHIRner["per_label"])


import matplotlib.pyplot as plt
import pandas as pd

# ---------- 1) Normalize your results into DataFrames ----------


def build_result_frames(result_map):
    """
    result_map: dict like {
        "BCPS_RULES": resRCPrules,
        "BCPS_NER":   resRCPner,
        "BCPS_LLM":   resRCPllm,
        "BCPS_LLM_Finetunned":   resRCPllm_finetunned,
        "SurgicalNote_RULES":resCHIRrules,
        "SurgicalNote_NER":  resCHIRner,
        "SurgicalNote_LLM":  resCHIRllm,
        "SurgicalNote_LLM_Finetunned":  resCHIRllm_finetunned,
    }

    Returns:
      df_overall: index=model (e.g. "BCPS_RULES"), columns=["precision","recall","f1"] from 'micro'
      df_perlabel: MultiIndex (model, label) -> precision/recall/f1
    """
    overall_rows = []
    perlabel_rows = []

    for name, res in result_map.items():
        # overall (micro)
        m = res["micro"]
        overall_rows.append(
            {
                "model": name,
                "precision": m.get("precision", 0.0),
                "recall": m.get("recall", 0.0),
                "f1": m.get("f1", 0.0),
                "tp": m.get("tp", 0),
                "fp": m.get("fp", 0),
                "fn": m.get("fn", 0),
            }
        )

        # per-label
        for lbl, d in res["per_label"].items():
            perlabel_rows.append(
                {
                    "model": name,
                    "label": lbl,
                    "precision": d.get("precision", 0.0),
                    "recall": d.get("recall", 0.0),
                    "f1": d.get("f1", 0.0),
                    "tp": d.get("tp", 0),
                    "fp": d.get("fp", 0),
                    "fn": d.get("fn", 0),
                }
            )

    df_overall = pd.DataFrame(overall_rows).set_index("model").sort_index()
    df_perlabel = pd.DataFrame(perlabel_rows).set_index(["model", "label"]).sort_index()
    return df_overall, df_perlabel


# ---------- 2) Overall comparison bar chart ----------


def plot_overall_comparison(df_overall, metric="f1", title="Overall (micro)"):
    """
    df_overall: from build_result_frames()[0]
    metric: "precision" | "recall" | "f1"
    """
    vals = df_overall[metric]
    ax = vals.plot(kind="bar", rot=45)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()


# ---------- 3) Per-entity grouped bars for one metric ----------


def plot_per_label_grouped(df_perlabel, metric="f1", subset_models=None, title=None):
    """
    df_perlabel: from build_result_frames()[1]
    subset_models: optional list to filter specific rows (e.g. ["BCPS_RULES","BCPS_NER","BCPS_LLM"])
    metric: "precision" | "recall" | "f1"
    """
    data = df_perlabel
    if subset_models:
        data = data.loc[data.index.get_level_values("model").isin(subset_models)]

    # pivot to labels x models
    pivot = data[metric].unstack(level="model")  # rows: label, cols: model
    ax = pivot.plot(kind="bar")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Entity label")
    ax.set_title(title or f"Per-entity {metric.upper()}")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# ---------- 4) Heatmap-style view (quick & simple without seaborn) ----------


def plot_per_label_heatmap(df_perlabel, metric="f1", subset_models=None, title=None):
    """
    Simple 'heatmap' using imshow (matplotlib only).
    Rows = labels, Cols = models.
    """
    import numpy as np

    data = df_perlabel
    if subset_models:
        data = data.loc[data.index.get_level_values("model").isin(subset_models)]

    pivot = data[metric].unstack(level="model").sort_index()  # labels x models
    arr = pivot.to_numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(arr, aspect="auto", vmin=0, vmax=1)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_title(title or f"{metric.upper()} by entity & model")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.upper())

    # annotate cells
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()


# ---------- 5) Example wiring with your variables ----------


def visualize_all():
    result_map = {
        "BCPS_LLM_Finetunned": resRCPllm_finetunned,
        "BCPS_RULES": resRCPrules,
        "BCPS_NER": resRCPner,
        "BCPS_LLM": resRCPllm,
        "SurgicalNote_RULES": resCHIRrules,
        "SurgicalNote_NER": resCHIRner,
        "SurgicalNote_LLM": resCHIRllm,
        "SurgicalNote_LLM_Finetunned": resCHIRllm_finetunned,
    }
    df_overall, df_perlabel = build_result_frames(result_map)

    # Overall micro-F1 across all 6 systems
    plot_overall_comparison(df_overall, metric="f1", title="Overall micro-F1")

    # Per-entity F1 for RCP set (3 systems)
    plot_per_label_grouped(
        df_perlabel,
        metric="f1",
        subset_models=["BCPS_RULES", "BCPS_NER", "BCPS_LLM"],
        title="RCP: Per-entity F1",
    )

    # Per-entity F1 for CHIR set (3 systems)
    plot_per_label_grouped(
        df_perlabel,
        metric="f1",
        subset_models=["SurgicalNote_RULES", "SurgicalNote_NER", "SurgicalNote_LLM"],
        title="CHIR: Per-entity F1",
    )

    # Optional: heatmap view for quick scanning
    plot_per_label_heatmap(
        df_perlabel,
        metric="f1",
        subset_models=["BCPS_RULES", "BCPS_NER", "BCPS_LLM"],
        title="RCP: F1 heatmap",
    )
    plot_per_label_heatmap(
        df_perlabel,
        metric="f1",
        subset_models=["SurgicalNote_RULES", "SurgicalNote_NER", "SurgicalNote_LLM"],
        title="CHIR: F1 heatmap",
    )


visualize_all()

# ---------- 6) Overall plots per dataset (RCP / CHIR) ----------


def _filter_overall_by_prefix(df_overall: pd.DataFrame, prefix: str) -> pd.Series:
    """Return a Series of the chosen metric for rows whose model starts with prefix (e.g., 'RCP_')."""
    # keep only models that start with the prefix
    sub = df_overall[df_overall.index.to_series().str.startswith(prefix)]
    # shorten index labels: RCP_RULES -> RULES, RCP_NER -> NER, ...
    sub = sub.copy()
    sub.index = sub.index.str.replace(f"^{prefix}", "", regex=True)
    return sub


def plot_overall_dataset(
    df_overall: pd.DataFrame,
    dataset_prefix: str = "BCPS_",
    metric: str = "f1",
    title: str | None = None,
):
    """
    Example:
      plot_overall_dataset(df_overall, "BCPS_", metric="f1", title="RCP: Overall micro-F1")
      plot_overall_dataset(df_overall, "SurgicalNote_", metric="f1", title="CHIR: Overall micro-F1")
    """
    sub = _filter_overall_by_prefix(df_overall, dataset_prefix)[metric]
    ax = sub.plot(kind="bar", rot=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Model")
    ax.set_title(
        title or f"{dataset_prefix.rstrip('_')}: Overall micro-{metric.upper()}"
    )
    for i, v in enumerate(sub):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()


df_overall, df_perlabel = build_result_frames(
    {
        "BCPS_RULES": resRCPrules,
        "BCPS_NER": resRCPner,
        "BCPS_LLM": resRCPllm,
        "BCPS_LLM_Finetunned": resRCPllm_finetunned,
        "SurgicalNote_RULES": resCHIRrules,
        "SurgicalNote_NER": resCHIRner,
        "SurgicalNote_LLM": resCHIRllm,
        "SurgicalNote_LLM_Finetunned": resCHIRllm_finetunned,
    }
)

# RCP overall (micro-F1)
plot_overall_dataset(df_overall, "BCPS_", metric="f1", title="RCP: Overall micro-F1")

# CHIR overall (micro-F1)
plot_overall_dataset(
    df_overall, "SurgicalNote_", metric="f1", title="CHIR: Overall micro-F1"
)

# If you want precision or recall instead:
# plot_overall_dataset(df_overall, "BCPS_", metric="precision", title="RCP: Overall micro-Precision")
# plot_overall_dataset(df_overall, "SurgicalNote_", metric="recall",    title="CHIR: Overall micro-Recall")
import pandas as pd


def build_summary_table(result_map, metric="f1", digits=3):
    """
    Build a table (DataFrame) summarizing results across datasets & methods.

    result_map: dict like {
        "BCPS_RULES": resRCPrules,
        "BCPS_NER":   resRCPner,
        "BCPS_LLM":   resRCPllm,
        "BCPS_LLM_Finetunned":   resRCPllm_finetunned,
        "SurgicalNote_RULES":resCHIRrules,
        "SurgicalNote_NER":  resCHIRner,
        "SurgicalNote_LLM":  resCHIRllm,
        "SurgicalNote_LLM_Finetunned":  resCHIRllm_finetunned,
    }

    Returns:
      DataFrame with columns: Dataset, Method, Precision, Recall, F1, TP, FP, FN
    """
    rows = []
    for name, res in result_map.items():
        dataset, method = name.split("_", 1)
        m = res["micro"]
        rows.append(
            {
                "Dataset": dataset,
                "Method": method,
                "Precision": round(m.get("precision", 0.0), digits),
                "Recall": round(m.get("recall", 0.0), digits),
                "F1": round(m.get("f1", 0.0), digits),
                "TP": m.get("tp", 0),
                "FP": m.get("fp", 0),
                "FN": m.get("fn", 0),
            }
        )
    df = pd.DataFrame(rows).set_index(["Dataset", "Method"]).sort_index()
    return df


result_map = {
    "BCPS_RULES": resRCPrules,
    "BCPS_NER": resRCPner,
    "BCPS_LLM": resRCPllm,
    "BCPS_LLM_Finetunned": resRCPllm_finetunned,
    "SurgicalNote_RULES": resCHIRrules,
    "SurgicalNote_NER": resCHIRner,
    "SurgicalNote_LLM": resCHIRllm,
    "SurgicalNote_LLM_Finetunned": resCHIRllm_finetunned,
}

summary = build_summary_table(result_map)
print(summary)

# If you want to export to CSV for reporting:
summary.to_csv("Results/breast_cancer_biomarker_eval_summary.csv")


"""
res["micro"]
pprint(res["per_label"])

fps = res["errors"]["false_positives"]
fns = res["errors"]["false_negatives"]

her2_status = []
er = []
pr = []
ihc = []
fish = []

for i in fns :
    if i["label"] == "Estrogen_receptor" : 
        er.append(i)
    elif i["label"] == "HER2_IHC" : 
        ihc.append(i)
    elif i["label"] == "Progesterone_receptor" : 
        pr.append(i)
    elif i["label"] == "HER2_status" : 
        her2_status.append(i)
    elif i["label"] == "HER2_FISH" : 
        fish.append(i)

len(pr),len(er),len(her2_status),len(ihc),len(fish)

her2_status = []
er = []
pr = []
ihc = []
fish = []

for i in fps :
    if i["label"] == "Estrogen_receptor" : 
        er.append(i)
    elif i["label"] == "HER2_IHC" : 
        ihc.append(i)
    elif i["label"] == "Progesterone_receptor" : 
        pr.append(i)
    elif i["label"] == "HER2_status" : 
        her2_status.append(i)
    elif i["label"] == "HER2_FISH" : 
        fish.append(i)

len(pr),len(er),len(her2_status),len(ihc),len(fish)

pprint(ihc)
"""

try:
    tracker.stop()
except Exception as e:
    print(
        f"\nWarning: Generalized error in Eco2AI tracking (likely 'N/A' vs float dtype issue): {e}"
    )
    print(
        "Carbon emission tracking data could not be saved, but analysis results are preserved."
    )
