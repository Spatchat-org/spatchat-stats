# app.py

# =========================
# Imports
# =========================
import os
import re
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr  # used for gr.Error and gr.update return values
from dotenv import load_dotenv

# LLM
from llm_utils import UnifiedLLM

try:
    from spatchat_gate import metered_chat
except ImportError:  # launched without spatchat-gateway's client/ on PYTHONPATH
    def metered_chat(**_kwargs):
        return lambda fn: fn

# Parsing & prompts
from parsing_utils import (
    set_llm,
    ask_llm,
    sanitize_args,
    local_parse,
    clarify_message,
)

# Report / outputs
from report_utils import (
    write_report,
    save_zip,
    clear_outputs,
    save_image_np,
)

# Plot/theme
from plot_helpers import set_house_style, bar_with_error_plot

# Core utils for schema-ish decisions
from core_utils import usable_numeric_cols, is_integer_like

# Stats API (from stats/ package)
from stats import (
    # two-group & paired
    ttest_summary,
    run_ttest,
    mann_whitney,
    wilcoxon_signed,
    point_biserial,
    # k-group & variance + posthocs
    run_anova,
    kruskal_wallis,
    levene_test,
    tukey_hsd,
    dunn_posthoc,
    # correlations
    corr_pair,
    corr_matrix_plot,
    partial_corr,
    # categorical associations
    chisq_test,
    # modeling
    run_ols,
    run_glm,
    # descriptives & checks
    plot_hist,
    plot_box,
    plot_violin,
    check_normality,
    # power
    power_ttest_ind,
    power_anova_oneway,
    # recommendations & summaries
    quick_summary,
    recommend_text_and_examples,
)

print("Starting SpatChat: Stats Room (chat-first)")

# =========================
# Globals
# =========================
cached_df: Optional[pd.DataFrame] = None
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# Track if we asked a follow-up
pending: Dict[str, Optional[str]] = {"action": None, "need": None, "args": None}

# =========================
# LLM init & styling
# =========================
load_dotenv()
llm = UnifiedLLM()
set_llm(llm)  # provide the LLM client to parsing_utils
set_house_style()  # apply house plotting theme once


# =========================
# Small helpers (incl. robust column resolution)
# =========================
def _fence(text: str) -> str:
    """Wrap multi-line stats text to prevent Markdown table parsing in the chat UI."""
    from textwrap import dedent
    return "```text\n" + dedent(str(text)).strip() + "\n```"


def _safe_last(paths: List[str]) -> Tuple[Optional[str], List[str], int]:
    if paths:
        return paths[-1], paths, len(paths) - 1
    return None, [], -1


def _step(paths: List[str], idx: int, delta: int) -> Tuple[Optional[str], int]:
    if not paths:
        return None, -1
    n = len(paths)
    new_idx = (idx + delta) % n
    return paths[new_idx], new_idx


def _col_lookup_map(df: pd.DataFrame) -> Dict[str, str]:
    """Map lowercase stripped column name -> actual column name (first match)."""
    m = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key not in m:
            m[key] = c
    return m


_SPECIAL_WHOLE_DATA = {"data", "dataset", "everything", "all", "*"}


def _resolve_colname(df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
    """Resolve a requested column name case-insensitively; pass through 'data/dataset' tokens."""
    if name is None:
        return None
    lower = str(name).strip().lower()
    if lower in _SPECIAL_WHOLE_DATA:
        return "data"
    m = _col_lookup_map(df)
    return m.get(lower, name)


def _resolve_many(df: pd.DataFrame, names: Optional[List[str]]) -> List[str]:
    if not names:
        return []
    out = []
    for n in names:
        r = _resolve_colname(df, n)
        if r is not None:
            out.append(r)
    return out


# Lightweight resolver for patsy-style formulas: replace identifiers with real-cased columns.
# This is conservative: we replace only bare-word tokens that match a column ignoring case.
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _resolve_formula(df: pd.DataFrame, formula: str) -> str:
    m = _col_lookup_map(df)
    # tokens that we should not rewrite (common patsy keywords/functions/operators)
    stop = {
        "C", "I", "np", "log", "exp", "sin", "cos", "BS", "bs", "Poly", "poly",
        "year", "month", "day", "Q", "CIs", "True", "False"
    }
    def repl(tok: re.Match) -> str:
        w = tok.group(0)
        lw = w.lower()
        if w in stop:
            return w
        return m.get(lw, w)
    return _IDENT_RE.sub(repl, formula)


def need_value_and_group(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Helper used when the user omits value/group; prefers common group-like columns.
    """
    nums = usable_numeric_cols(df)
    groups = []
    for c in df.columns:
        if df[c].nunique(dropna=True) >= 2:
            if not pd.api.types.is_numeric_dtype(df[c]) or is_integer_like(df[c]):
                groups.append(c)
    prefer = ["sex", "group", "treatment", "class", "category"]
    ordered_g = [c for c in prefer if c in [g.lower() for g in groups]]
    # keep original casing from df, but sort by our preference list first
    ordered = []
    for pref in prefer:
        for g in groups:
            if str(g).strip().lower() == pref and g not in ordered:
                ordered.append(g)
    ordered += [g for g in groups if g not in ordered]
    return nums, ordered


# =========================
# Data & chat handlers (used by UI)
# =========================
def handle_upload(file):
    """
    UI wires: file_input.change(handle_upload, ...)
    Returns: (chat_history, data_preview_component, download_btn_component)
    """
    global cached_df
    clear_outputs()
    pending.update({"action": None, "need": None, "args": None})
    try:
        df = pd.read_csv(file)
        cached_df = df
        cols = ", ".join(map(str, df.columns))
        preview = df.head(200)
        return (
            [
                {
                    "role": "assistant",
                    "content": (
                        "CSV uploaded. Columns detected: "
                        f"{cols}. Ask me for t-test, ANOVA (+ Tukey), Kruskal (+ Dunn), "
                        "correlations/partial/pbiserial, chi-square/Fisher, rank tests, Levene’s, "
                        "OLS/GLM, histogram, box/violin/bar, normality, power, quick summaries, "
                        "or 'what can I do with my data?'"
                    ),
                }
            ],
            gr.update(value=preview, visible=True),
            gr.update(visible=True),
        )
    except Exception as e:
        return (
            [{"role": "assistant", "content": f"Failed to read CSV: {e}"}],
            gr.update(visible=False),
            gr.update(visible=False),
        )


def _chat_denied(markdown, chat_history, user_message, data_preview):
    history = list(chat_history or []) + [
        {"role": "user", "content": str(user_message or "")},
        {"role": "assistant", "content": markdown},
    ]
    return history, gr.skip(), gr.skip(), data_preview, gr.skip(), gr.skip()


def handle_chat(chat_history, user_message, data_preview):
    """
    Core router for user requests. The UI passes (chat_history, user_input, data_preview)
    and expects:
      (chat_history, preview_plot_component, download_btn_component,
       data_preview, gallery_paths_state, gallery_index_state)
    """
    global cached_df, pending
    chat_history = list(chat_history)
    text = str(user_message or "").strip()

    parsed = None
    llm_error_text = None

    # ---- NEW: Local parse first to avoid LLM overriding simple summaries ----
    lp = local_parse(text)
    if lp:
        parsed = lp

    # If no local parse match, try the LLM
    if not parsed:
        try:
            tool, _ = ask_llm(chat_history, text)
            parsed = tool
        except Exception as e:
            llm_error_text = f"(LLM unavailable: {e})"

    if pending["action"] and parsed and parsed.get("tool") == "stats":
        pending = {"action": None, "need": None, "args": None}

    if not parsed or parsed.get("tool") != "stats":
        msg = clarify_message(cached_df, llm_error_text, text)
        chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])
        preview, paths, idx = _safe_last([])
        return (
            chat_history,
            gr.update(value=preview),
            gr.update(value=None, visible=False),
            data_preview,
            paths,
            idx,
        )

    if cached_df is None:
        msg = "Please upload a CSV first."
        chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])
        preview, paths, idx = _safe_last([])
        return (
            chat_history,
            gr.update(value=preview),
            gr.update(value=None, visible=False),
            data_preview,
            paths,
            idx,
        )

    action = parsed.get("action")
    args = sanitize_args(action, parsed.get("args", {}))
    df = cached_df.copy()

    # --- central resolution for all arg column names ---
    def _rg(name: Optional[str]) -> Optional[str]:
        return _resolve_colname(df, name)
    def _rg_many(names: Optional[List[str]]) -> List[str]:
        return _resolve_many(df, names)

    sections: List[Tuple[str, str, Optional[np.ndarray]]] = []
    image_paths: List[str] = []

    try:
        # -------- Recommendations --------
        if action == "recommend":
            tech, examples = recommend_text_and_examples(df)
            msg = "Recommendations\n" + tech + "\n\nSay it like this\n" + examples
            sections.append(("Recommendations", msg, None))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])
            write_report(sections)
            zip_fp = save_zip()
            preview, paths, idx = _safe_last([])
            return (chat_history, gr.update(value=preview), gr.update(value=zip_fp, visible=True), data_preview, paths, idx)

        # -------- Summary --------
        if action == "summary":
            col = _rg(args.get("col"))
            by  = _rg(args.get("by"))
            if col is None or str(col).strip().lower() in _SPECIAL_WHOLE_DATA:
                col = "data"
            print(f"[DEBUG] quick_summary called with col={col}, by={by}")  # <--- keep your debug
            msg = quick_summary(df, col, by)
            sections.append(("Summary", msg, None))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(msg)}])
            write_report(sections)
            zip_fp = save_zip()
            preview, paths, idx = _safe_last([])
            return (chat_history, gr.update(value=preview), gr.update(value=zip_fp, visible=True), data_preview, paths, idx)

        # -------- t-test --------
        if action == "ttest":
            value = _rg(args.get("value"))
            group = _rg(args.get("group"))
            if not value or not group:
                nums, groups = need_value_and_group(df)
                pending.update({"action": "ttest", "need": "value" if not value else "group", "args": {"value": value, "group": group}})
                ask = ("Which numeric outcome should I test? Candidates: " + ", ".join(map(str, nums))
                       if not value else "Which group column? Candidates: " + ", ".join(map(str, groups)))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = run_ttest(
                df, value=value, group=group,
                paired=bool(args.get("paired", False)),
                equal_var=bool(args.get("equal_var", False)),
            )
            sections.append(("t-test", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"ttest_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- ANOVA --------
        elif action == "anova":
            value = _rg(args.get("value")); group = _rg(args.get("group"))
            if not value or not group:
                nums, groups = need_value_and_group(df)
                pending.update({"action": "anova", "need": "value" if not value else "group", "args": {"value": value, "group": group}})
                ask = ("Which numeric outcome for ANOVA? Candidates: " + ", ".join(map(str, nums))
                       if not value else "Which group column for ANOVA? Candidates: " + ", ".join(map(str, groups)))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = run_anova(df, value=value, group=group)
            sections.append(("ANOVA", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"anova_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- POSTHOC: TUKEY --------
        elif action == "posthoc_tukey":
            value = _rg(args.get("value")); group = _rg(args.get("group"))
            if not value or not group:
                nums, groups = need_value_and_group(df)
                ask = ("Which numeric outcome for Tukey? Candidates: " + ", ".join(map(str, nums))
                       if not value else "Which group column (≥2 levels)? Candidates: " + ", ".join(map(str, groups)))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = tukey_hsd(df, value=value, group=group)
            sections.append(("Post-hoc: Tukey HSD", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"tukey_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- KRUSKAL --------
        elif action == "kruskal":
            value = _rg(args.get("value")); group = _rg(args.get("group"))
            if not value or not group:
                nums, groups = need_value_and_group(df)
                ask = ("Which numeric outcome for Kruskal–Wallis? Candidates: " + ", ".join(map(str, nums))
                       if not value else "Which group column (≥2 levels)? Candidates: " + ", ".join(map(str, groups)))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = kruskal_wallis(df, value=value, group=group)
            sections.append(("Kruskal–Wallis", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"kruskal_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- POSTHOC: DUNN --------
        elif action == "posthoc_dunn":
            value = _rg(args.get("value")); group = _rg(args.get("group")); padj = args.get("p_adjust", "holm")
            if not value or not group:
                nums, groups = need_value_and_group(df)
                ask = ("Which numeric outcome for Dunn’s test? Candidates: " + ", ".join(map(str, nums))
                       if not value else "Which group column (≥2 levels)? Candidates: " + ", ".join(map(str, groups)))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, _ = dunn_posthoc(df, value=value, group=group, p_adjust=padj)
            sections.append(("Post-hoc: Dunn’s", txt, None))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- CHI-SQUARE / FISHER --------
        elif action == "chisq":
            row = _rg(args.get("row")); col = _rg(args.get("col")); exact = bool(args.get("exact", False))
            if not row or not col:
                cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
                ask = "Which two categorical columns for chi-square? Candidates: " + ", ".join(map(str, cats))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = chisq_test(df, row=row, col=col, exact=exact)
            sections.append(("Chi-square / Fisher", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"chisq_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- CORRELATION(S) --------
        elif action == "corr":
            if bool(args.get("matrix", False)):
                method = args.get("method", "pearson")
                cols = _rg_many(args.get("cols"))
                txt, imgs = corr_matrix_plot(df, cols=cols, method=method)
                sections.append(("Correlation matrix", txt, None))
                for i, im in enumerate(imgs):
                    image_paths.append(save_image_np(im, f"corr_matrix_{method}_{i+1}.png"))
                chat_history.extend(
                    [{"role": "user", "content": text}, {"role": "assistant", "content": "Correlation matrix generated."}]
                )
            else:
                x = _rg(args.get("x")); y = _rg(args.get("y")); method = args.get("method", "pearson")
                if not x or not y:
                    nums = usable_numeric_cols(df)
                    ask = "Which two numeric columns for correlation? Candidates: " + ", ".join(map(str, nums))
                    chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                    preview, paths, idx = _safe_last([])
                    return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

                txt, imgs = corr_pair(df, x=x, y=y, method=method)
                sections.append(("Correlation", txt, None))
                for i, im in enumerate(imgs):
                    image_paths.append(save_image_np(im, f"corr_pair_{method}_{i+1}.png"))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- PARTIAL CORRELATION --------
        elif action == "pcorr":
            x = _rg(args.get("x")); y = _rg(args.get("y")); ctrls = _rg_many(args.get("controls", [])); method = args.get("method", "pearson")
            if not x or not y or not ctrls:
                nums = usable_numeric_cols(df)
                ask = "Specify: partial correlation <x> and <y> controlling for <a, b>. Numeric candidates: " + ", ".join(map(str, nums))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = partial_corr(df, x=x, y=y, controls=ctrls, method=method)
            sections.append(("Partial correlation", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"pcorr_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- POINT-BISERIAL --------
        elif action == "pbiserial":
            value = _rg(args.get("value")); group = _rg(args.get("group"))
            if not value or not group:
                nums = usable_numeric_cols(df)
                groups = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
                ask = ("Which numeric outcome for point-biserial? Candidates: " + ", ".join(map(str, nums))
                       if not value else "Which binary group? Candidates: " + ", ".join(map(str, groups)))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = point_biserial(df, value=value, group=group)
            sections.append(("Point-biserial correlation", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"pbiserial_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- MANN–WHITNEY --------
        elif action == "mwutest":
            value = _rg(args.get("value")); group = _rg(args.get("group"))
            if not value or not group:
                nums, groups = need_value_and_group(df)
                ask = ("Which numeric outcome for Mann–Whitney? Candidates: " + ", ".join(map(str, nums))
                       if not value else "Which group column (2 levels needed)? Candidates: " + ", ".join(map(str, groups)))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = mann_whitney(df, value=value, group=group)
            sections.append(("Mann–Whitney U", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"mwutest_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- WILCOXON SIGNED-RANK --------
        elif action == "wilcoxon":
            a = _rg(args.get("a")); b = _rg(args.get("b"))
            if not a or not b:
                nums = usable_numeric_cols(df)
                ask = "Which two columns are paired? Example: 'wilcoxon pre vs post'. Candidates: " + ", ".join(map(str, nums))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, imgs = wilcoxon_signed(df, a=a, b=b)
            sections.append(("Wilcoxon signed-rank", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"wilcoxon_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- LEVENE --------
        elif action == "levene":
            value = _rg(args.get("value")); group = _rg(args.get("group")); center = args.get("center", "median")
            if not value or not group:
                nums, groups = need_value_and_group(df)
                ask = ("Which numeric outcome for Levene? Candidates: " + ", ".join(map(str, nums))
                       if not value else "Which group column? Candidates: " + ", ".join(map(str, groups)))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            txt, _ = levene_test(df, value=value, group=group, center=center)
            sections.append(("Levene’s test", txt, None))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- OLS --------
        elif action == "ols":
            formula = args.get("formula")
            if not formula or "~" not in formula:
                ask = "Please specify a formula like: ols score ~ age + weight."
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            formula = _resolve_formula(df, formula)
            txt, imgs = run_ols(df, formula=formula)
            sections.append(("OLS", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"ols_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- GLM --------
        elif action == "glm":
            formula = args.get("formula"); family = args.get("family", "gaussian")
            if not formula or "~" not in formula:
                ask = "Please specify a formula like: glm count ~ age with family=poisson."
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                preview, paths, idx = _safe_last([])
                return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

            formula = _resolve_formula(df, formula)
            txt, imgs = run_glm(df, formula=formula, family=family)
            sections.append((f"GLM ({family})", txt, None))
            for i, im in enumerate(imgs):
                image_paths.append(save_image_np(im, f"glm_plot_{i+1}.png"))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- PLOTS --------
        elif action == "plot":
            # Histogram
            if "hist" in args:
                p = args["hist"]
                col = _rg(p.get("col"))
                title, im = plot_hist(df, col, int(p.get("bins", 30)))
                image_paths.append(save_image_np(im, f"plot_hist_{str(col)}.png"))
                sections.append((title, "", im))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": f"{title} generated."}])

            # Box
            if "box" in args:
                p = args["box"]; val = _rg(p.get("value")); grp = _rg(p.get("group"))
                if grp is None:
                    _, groups = need_value_and_group(df)
                    ask = "Which group column for the box plot? Candidates: " + ", ".join(map(str, groups))
                    chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                    preview, paths, idx = _safe_last([])
                    return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)
                if val == grp:
                    raise gr.Error("Outcome and group must be different.")
                title, im = plot_box(df, val, grp)
                image_paths.append(save_image_np(im, f"plot_box_{str(val)}_{str(grp)}.png"))
                sections.append((title, "", im))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": f"{title} generated."}])

            # Violin
            if "violin" in args:
                p = args["violin"]; val = _rg(p.get("value")); grp = _rg(p.get("group"))
                if grp is None:
                    _, groups = need_value_and_group(df)
                    ask = "Which group column for the violin plot? Candidates: " + ", ".join(map(str, groups))
                    chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                    preview, paths, idx = _safe_last([])
                    return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)
                if val == grp:
                    raise gr.Error("Outcome and group must be different.")
                title, im = plot_violin(df, val, grp)
                image_paths.append(save_image_np(im, f"plot_violin_{str(val)}_{str(grp)}.png"))
                sections.append((title, "", im))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": f"{title} generated."}])

            # Bar
            if "bar" in args:
                p = args["bar"]; val = _rg(p.get("value")); grp = _rg(p.get("group"))
                if grp is None:
                    _, groups = need_value_and_group(df)
                    ask = "Which group column for the bar chart? Candidates: " + ", ".join(map(str, groups))
                    chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": ask}])
                    preview, paths, idx = _safe_last([])
                    return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)
                if val == grp:
                    raise gr.Error("Outcome and group must be different.")
                err = p.get("error", "sem")
                # Use stable ordering inferred by helper
                gorder = df[grp].dropna().astype(str).unique().tolist()
                im = bar_with_error_plot(df, val, grp, error=err, gorder=gorder)
                image_paths.append(save_image_np(im, f"plot_bar_{str(val)}_{str(grp)}_{err}.png"))
                sections.append((f"Bar {str(val)}~{str(grp)}", "", im))
                chat_history.extend(
                    [{"role": "user", "content": text}, {"role": "assistant", "content": f"Bar ± {str(err).upper()} generated."}]
                )

        # -------- CHECKS --------
        elif action == "check":
            if "normality" in args:
                p = args["normality"]
                col = _rg(p.get("col"))
                msg, im = check_normality(df, col)
                image_paths.append(save_image_np(im, f"plot_qq_{str(col)}.png"))
                sections.append(("Normality", msg, im))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(msg)}])

        # -------- POWER --------
        elif action == "power":
            if "ttest_ind" in args:
                p = args["ttest_ind"]
                msg = power_ttest_ind(
                    p.get("effect_size", 0.5),
                    float(p.get("alpha", 0.05)),
                    p.get("power", 0.8),
                    float(p.get("ratio", 1.0)),
                    p.get("solve_for", "n_total"),
                )
                sections.append(("Power – t-test (ind)", msg, None))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(msg)}])
            if "anova_oneway" in args:
                p = args["anova_oneway"]
                msg = power_anova_oneway(
                    p.get("effect_size", 0.25),
                    int(p.get("k_groups", 3)),
                    float(p.get("alpha", 0.05)),
                    p.get("power", 0.8),
                    p.get("solve_for", "n_per_group"),
                )
                sections.append(("Power – ANOVA (one-way)", msg, None))
                chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(msg)}])

        else:
            msg = clarify_message(df, llm_error_text, text)
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])

    except gr.Error as e:
        msg = f"Sorry — {e}"
        chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])
        preview, paths, idx = _safe_last([])
        return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

    except Exception as e:
        msg = f"Sorry — something went wrong: {e}"
        chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])
        preview, paths, idx = _safe_last([])
        return (chat_history, gr.update(value=preview), gr.update(value=None, visible=False), data_preview, paths, idx)

    # Finalize report & zip, prep gallery preview
    write_report(sections)
    zip_fp = save_zip()
    preview, paths, idx = _safe_last(image_paths)
    return (chat_history, gr.update(value=preview), gr.update(value=zip_fp, visible=True), data_preview, paths, idx)


# =========================
# Optional: gallery nav helpers the UI may bind to buttons
# =========================
def on_prev(paths, idx):
    preview, new_idx = _step(paths, idx, -1)
    return gr.update(value=preview), new_idx


def on_next(paths, idx):
    preview, new_idx = _step(paths, idx, +1)
    return gr.update(value=preview), new_idx

# --------------------------
# UI (unchanged layout)
# --------------------------
def _logo_html() -> str:
    import base64
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo_small.png")
    try:
        with open(path, "rb") as f:
            src = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return '<span style="font-weight:700;color:#e8527a">Spatchat</span>'
    return f'<img src="{src}" alt="Spatchat" draggable="false">'


_LOGO_HTML = _logo_html()

# Same look as the Move room: green palette, pill header bar, dark-mode toggle.
_THEME_HEAD = """
<script>
window.spatchatApplyTheme = function(theme) {
  const dark = theme === "dark";
  document.documentElement.classList.toggle("spatchat-dark", dark);
  document.body.classList.toggle("spatchat-dark", dark);
  document.body.classList.toggle("dark", dark);
  try { localStorage.setItem("spatchat-theme", dark ? "dark" : "light"); } catch (e) {}
};
window.spatchatToggleTheme = function() {
  window.spatchatApplyTheme(document.body.classList.contains("spatchat-dark") ? "light" : "dark");
};
(function () {
  let saved = null;
  try { saved = localStorage.getItem("spatchat-theme"); } catch (e) {}
  const apply = () => window.spatchatApplyTheme(saved === "dark" ? "dark" : "light");
  if (document.body) apply(); else document.addEventListener("DOMContentLoaded", apply, { once: true });
})();
</script>
"""

_STATS_CSS = """
<style>
:root {
  --spatchat-bg: #f5f8f5; --spatchat-surface: #ffffff; --spatchat-surface-alt: #eef3ee;
  --spatchat-border: rgba(16, 36, 26, 0.1); --spatchat-text: #10241a; --spatchat-text-secondary: #5c6b63;
  --spatchat-accent: #1f7a4f; --spatchat-accent-hover: #17603d; --spatchat-accent-soft: #e3f2e8;
  --spatchat-shadow: 0 1px 3px rgba(16, 36, 26, 0.08), 0 1px 2px rgba(16, 36, 26, 0.06);
}
body.spatchat-dark {
  --spatchat-bg: #0f1713; --spatchat-surface: #17211b; --spatchat-surface-alt: #1f2c24;
  --spatchat-border: rgba(230, 240, 233, 0.12); --spatchat-text: #e6f0e9; --spatchat-text-secondary: #a3b3a9;
  --spatchat-accent: #4fbf86; --spatchat-accent-hover: #6fd3a0; --spatchat-accent-soft: #1d3a2b;
  --spatchat-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
}
html, body, gradio-app, .gradio-container, .gradio-container .main {
  background: var(--spatchat-bg) !important; color: var(--spatchat-text) !important;
}
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
#spatchat-header-row {
  display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  background: var(--spatchat-surface); border: 1px solid var(--spatchat-border); box-shadow: var(--spatchat-shadow);
  border-radius: 12px; padding: 6px 18px; margin: 4px 0 8px;
}
#spatchat-header-row img { height: 32px; margin: 6px 0; border-radius: 6px; display: block; }
body.spatchat-dark #spatchat-header-row img { background: #fff; border-radius: 8px; padding: 2px 6px; }
#spatchat-header-row nav { display: flex; gap: 22px; align-items: center; font-size: 15px; font-weight: 600; }
#spatchat-header-row nav a { color: var(--spatchat-text); text-decoration: none; }
#spatchat-header-row nav a:hover, #spatchat-header-row nav a.accent { color: var(--spatchat-accent); }
.spatchat-title-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin: 8px 0 14px; }
.spatchat-title { margin: 0; font-size: 26px; font-weight: 700; color: var(--spatchat-text); }
.spatchat-subtitle { margin: 4px 0 0; font-size: 14px; color: var(--spatchat-text-secondary); }
.spatchat-title-actions { display: flex; gap: 8px; align-items: center; }
.spatchat-title-actions button {
  padding: 8px 16px; font-size: 13px; font-weight: 600; border-radius: 8px; cursor: pointer;
  border: 1px solid var(--spatchat-border); background: transparent; color: var(--spatchat-text-secondary);
}
.spatchat-title-actions button:hover { background: var(--spatchat-surface-alt); color: var(--spatchat-text); }
.spatchat-title-actions #spatchat-share-btn { border-color: var(--spatchat-accent); color: var(--spatchat-accent); }
.spatchat-theme-dark { display: none; }
body.spatchat-dark .spatchat-theme-light { display: none; }
body.spatchat-dark .spatchat-theme-dark { display: inline; }
.spatchat-footer { font-size: 12px; color: var(--spatchat-text-secondary); margin-top: 18px; }
.gradio-container .block, .gradio-container .form { background: var(--spatchat-surface) !important; border-color: var(--spatchat-border) !important; border-radius: 12px !important; }
.gradio-container button.primary, .gradio-container .primary { background: var(--spatchat-accent) !important; border-color: var(--spatchat-accent) !important; color: #fff !important; }
.gradio-container button.secondary { background: var(--spatchat-surface-alt) !important; color: var(--spatchat-text) !important; border: 1px solid var(--spatchat-border) !important; }
.gradio-container button.secondary:hover { background: var(--spatchat-accent-soft) !important; }
.gradio-container textarea, .gradio-container input[type=text] { background: var(--spatchat-surface) !important; color: var(--spatchat-text) !important; }
.dataframe-wrap { max-height: 340px; overflow: auto; border: 1px solid var(--spatchat-border); border-radius: 6px; padding: 6px; }
#preview-plot { margin-bottom: 6px; }
#plot-nav { margin: 0 0 10px 0; display: flex; gap: 8px; justify-content: flex-end; }
</style>
"""

_STATS_HEADER = """
<div id="spatchat-header-row">
  <div>""" + _LOGO_HTML + """</div>
  <nav>
    <a href="https://spatchat.org/" target="_blank" rel="noopener">Home</a>
    <a href="https://spatchat.org/about" target="_blank" rel="noopener">About</a>
    <a href="https://spatchat.org/browse" target="_blank" rel="noopener" class="accent">Browse Rooms</a>
    <a href="https://spatchat.org/join" target="_blank" rel="noopener">Join</a>
  </nav>
</div>
<div class="spatchat-title-row">
  <div>
    <h1 class="spatchat-title">Stats Room</h1>
    <p class="spatchat-subtitle">General statistics, data exploration, and plots through conversation.</p>
  </div>
  <div class="spatchat-title-actions">
    <button id="spatchat-theme-btn" type="button" onclick="window.spatchatToggleTheme && window.spatchatToggleTheme()">
      <span class="spatchat-theme-light">Dark mode</span><span class="spatchat-theme-dark">Light mode</span>
    </button>
    <button id="spatchat-share-btn" type="button" onclick="
      navigator.clipboard.writeText('https://spatchat.org/browse/?room=stats');
      this.textContent = 'Link copied';
      setTimeout(() => { this.textContent = 'Share'; }, 1500);
    ">Share</button>
  </div>
</div>
"""

_STATS_FOOTER = """
<div class="spatchat-footer">
© 2025 Ho Yi Wan &amp; Logan Hysen. All rights reserved. If you use SpatChat in research, please cite:
<b>Wan, H.Y.</b> &amp; <b>Hysen, L.</b> (2025). <i>SpatChat: Stats Room.</i>
</div>
"""


with gr.Blocks(title="SpatChat: Stats Room", head=_THEME_HEAD) as demo:
    gr.HTML(_STATS_CSS + _STATS_HEADER)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat",
                show_label=True,
                type="messages",
                value=[
                    {
                        "role": "assistant",
                        "content": "Welcome! Upload a CSV, then ask: t-test, ANOVA (+Tukey), Kruskal (+Dunn), correlations (Pearson/Spearman), partial/pbiserial, chi-square/Fisher, rank tests (Mann-Whitney, Wilcoxon), Levene’s, OLS/GLM, histogram, box/violin/bar, normality, power, quick summaries, or 'what can I do with my data?'.",
                    }
                ],
            )
            user_input = gr.Textbox(label="Ask SpatChat", placeholder="e.g., Tukey HSD score by group", lines=1)
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])

        with gr.Column(scale=3):
            preview_plot = gr.Image(label="Preview (last figure)", value=None, type="filepath", elem_id="preview-plot")
            with gr.Row(elem_id="plot-nav"):
                prev_btn = gr.Button("◀️ Prev", variant="secondary")
                next_btn = gr.Button("Next ▶️", variant="secondary")
            data_preview = gr.Dataframe(label="Data Preview (first 200 rows)", interactive=False, visible=False)
            download_btn = gr.DownloadButton("📥 Download Results", value=None, visible=False)
            gallery_paths = gr.State([])
            gallery_index = gr.State(-1)

    gr.HTML(_STATS_FOOTER)

    demo.queue(max_size=16)

    file_input.change(
        handle_upload,
        inputs=file_input,
        outputs=[chatbot, data_preview, download_btn],
    )

    user_input.submit(
        metered_chat(room="stats", message_index=1, deny=_chat_denied)(handle_chat),
        inputs=[chatbot, user_input, data_preview],
        outputs=[chatbot, preview_plot, download_btn, data_preview, gallery_paths, gallery_index],
    )
    user_input.submit(lambda *args: "", inputs=None, outputs=user_input)

    def on_prev(paths, idx):
        preview, new_idx = _step(paths, idx, -1)
        return gr.update(value=preview), new_idx

    def on_next(paths, idx):
        preview, new_idx = _step(paths, idx, +1)
        return gr.update(value=preview), new_idx

    prev_btn.click(on_prev, inputs=[gallery_paths, gallery_index], outputs=[preview_plot, gallery_index])
    next_btn.click(on_next, inputs=[gallery_paths, gallery_index], outputs=[preview_plot, gallery_index])


if __name__ == "__main__":
    demo.launch(ssr_mode=False)