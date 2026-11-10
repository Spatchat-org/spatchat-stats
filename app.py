# app.py

# =========================
# Imports
# =========================
import os
import re
import json
import shutil
import threading
import time
import uuid
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr  # used for gr.Error and gr.update return values
import spatchat_ui
import spatchat_files as sfiles
import spatchat_layout
from gradio.context import LocalContext
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
# Per-visitor state: one entry per browser tab, with its own sessions/<id>/{uploads,outputs} folders
# (removed when the tab closes). Nothing a visitor uploads or produces is visible to anyone else.
# =========================
SESSIONS_ROOT = "sessions"
shutil.rmtree(SESSIONS_ROOT, ignore_errors=True)  # leftovers of a previous process belong to nobody
os.makedirs(SESSIONS_ROOT, exist_ok=True)
_SESSIONS: Dict[str, dict] = {}
_HASH_TO_SID: Dict[str, str] = {}
_LAST_SEEN: Dict[str, float] = {}
_UNLOAD_GRACE_SECONDS = 15 * 60


def _request_hash() -> Optional[str]:
    try:
        request = LocalContext.request.get()
    except LookupError:
        return None
    return str(getattr(request, "session_hash", "") or "") or None


def _session(sid: Optional[str]) -> Tuple[str, dict]:
    """(session id, state) for this visitor, created on first use."""
    sid = (sid or "").strip()
    if not sid:
        # an event that arrives before the new session id reached the browser joins the session this connection has
        rh = _request_hash()
        sid = _HASH_TO_SID.get(rh, "") if rh else ""
    sid = sid or uuid.uuid4().hex
    _LAST_SEEN[sid] = time.time()
    S = _SESSIONS.get(sid)
    if S is None:
        S = {"df": None, "active": None, "pending": {"action": None, "need": None, "args": None}}
        _SESSIONS[sid] = S
    S["root"] = os.path.join(SESSIONS_ROOT, sid)
    S["upload_dir"] = os.path.join(S["root"], "uploads")
    S["output_dir"] = os.path.join(S["root"], "outputs")
    os.makedirs(S["upload_dir"], exist_ok=True)
    os.makedirs(S["output_dir"], exist_ok=True)
    rh = _request_hash()
    if rh:
        _HASH_TO_SID[rh] = sid
    return sid, S


def _cleanup_if_still_idle(sid: str, closed_at: float) -> None:
    if _LAST_SEEN.get(sid, 0.0) > closed_at:
        return  # the visitor came back: the "close" was a dropped connection
    _SESSIONS.pop(sid, None)
    shutil.rmtree(os.path.join(SESSIONS_ROOT, sid), ignore_errors=True)


def _cleanup_browser_session() -> None:
    """Gradio says a tab closed. Behind proxies a dropped keep-alive looks the same, so only delete the data if
    nothing touches the session for a while."""
    rh = _request_hash()
    sid = _HASH_TO_SID.pop(rh, None) if rh else None
    if sid:
        timer = threading.Timer(_UNLOAD_GRACE_SECONDS, _cleanup_if_still_idle, args=(sid, time.time()))
        timer.daemon = True
        timer.start()


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
def _message_text(message) -> str:
    """Plain text of a chat message. Gradio 6 delivers content as a list of blocks, Gradio 5 as a string."""
    content = message.get("content") if isinstance(message, dict) else message
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(
            _message_text(part.get("text") if isinstance(part, dict) and "text" in part else part)
            for part in content if part)
    return "" if content is None else str(content)


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
_SAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "demo_stats.csv")


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")


def _activate(S: dict, path: Optional[str]) -> None:
    """Make `path` the dataset the analyses run on (or none)."""
    S["pending"].update({"action": None, "need": None, "args": None})
    if path and os.path.isfile(path):
        S["df"], S["active"] = _read_csv(path), path
    else:
        S["df"], S["active"] = None, None


def handle_upload(file, sid, chat_history):
    """Copy an uploaded CSV into this visitor's folder and make it the active dataset.
    Returns (chat_history, download_btn, session id)."""
    sid, S = _session(sid)
    chat = list(chat_history or [])
    src = getattr(file, "name", file)
    try:
        name = os.path.basename(str(src))
        stem, ext = os.path.splitext(name)
        dst, n = os.path.join(S["upload_dir"], name), 2
        while os.path.exists(dst):
            dst = os.path.join(S["upload_dir"], f"{stem}_{n}{ext}")
            n += 1
        shutil.copy2(src, dst)
        _activate(S, dst)
        cols = ", ".join(map(str, S["df"].columns))
        chat.append({"role": "assistant", "content": (
            f"Loaded **{os.path.basename(dst)}**. Columns detected: {cols}. "
            "Ask me for t-test, ANOVA (+ Tukey), Kruskal (+ Dunn), "
            "correlations/partial/pbiserial, chi-square/Fisher, rank tests, Levene’s, "
            "OLS/GLM, histogram, box/violin/bar, normality, power, quick summaries, "
            "or 'what can I do with my data?'")})
    except Exception as e:
        try:
            os.remove(dst)  # do not keep a file we could not read
        except Exception:
            pass
        chat.append({"role": "assistant", "content": f"Failed to read CSV: {e}"})
    return chat, gr.update(), sid


def _chat_denied(markdown, chat_history, user_message, sid):
    history = list(chat_history or []) + [
        {"role": "user", "content": str(user_message or "")},
        {"role": "assistant", "content": markdown},
    ]
    return history, gr.skip(), sid


_GROUP_ACTIONS = {"ttest", "anova", "posthoc_tukey", "kruskal", "posthoc_dunn", "levene", "mwutest", "pbiserial"}


def _fill_default_columns(df: pd.DataFrame, action: Optional[str], args: Dict) -> Optional[str]:
    """Fill value/group (or plot columns) the user left out. Returns a note describing what was chosen."""
    nums, groups = need_value_and_group(df)
    if not nums:
        return None
    nums = [str(c) for c in nums]
    groups = [str(g) for g in groups if str(g) not in nums]
    prefer_value = next((c for c in nums if c.lower() in {"score", "value", "outcome", "y"}), nums[-1])

    def pick_group(levels=None):
        ok = [g for g in groups if levels is None or df[g].nunique(dropna=True) == levels]
        if levels == 3:
            ok = [g for g in groups if df[g].nunique(dropna=True) >= 3]
        return (ok or groups or [None])[0]

    chosen = []
    if action in _GROUP_ACTIONS:
        if not args.get("value"):
            args["value"] = prefer_value; chosen.append(f"outcome **{prefer_value}**")
        if not args.get("group"):
            levels = 2 if action in {"ttest", "mwutest", "pbiserial"} else 3 if action in {"anova", "posthoc_tukey", "kruskal", "posthoc_dunn"} else None
            g = pick_group(levels)
            if g:
                args["group"] = g; chosen.append(f"group **{g}**")
    elif action == "plot":
        if isinstance(args.get("hist"), dict) and not args["hist"].get("col"):
            args["hist"]["col"] = prefer_value; chosen.append(f"column **{prefer_value}**")
        for kind in ("box", "violin", "bar"):
            spec = args.get(kind)
            if isinstance(spec, dict):
                if not spec.get("value"):
                    spec["value"] = prefer_value; chosen.append(f"outcome **{prefer_value}**")
                if not spec.get("group"):
                    g = pick_group()
                    if g:
                        spec["group"] = g; chosen.append(f"group **{g}**")
    elif action == "check":
        spec = args.get("normality")
        if isinstance(spec, dict) and not spec.get("col"):
            spec["col"] = prefer_value; chosen.append(f"column **{prefer_value}**")
    elif action == "corr" and not args.get("matrix") and not (args.get("x") and args.get("y")):
        args["matrix"] = True; args["cols"] = []
        chosen.append("all numeric columns")
    elif action == "wilcoxon" and not (args.get("a") and args.get("b")) and len(nums) >= 2:
        args["a"], args["b"] = nums[0], nums[1]; chosen.append(f"**{nums[0]}** vs **{nums[1]}**")
    elif action == "chisq" and not (args.get("row") and args.get("col")) and len(groups) >= 2:
        args["row"], args["col"] = groups[0], groups[1]; chosen.append(f"**{groups[0]}** by **{groups[1]}**")
    return ("Using " + ", ".join(chosen) + " (say which columns you want to change it).") if chosen else None



def _handle_chat_core(chat_history, user_message, S):
    """
    Core router for user requests. The UI passes (chat_history, user_input, data_preview)
    and expects:
      (chat_history, preview_plot_component, download_btn_component,
       data_preview, gallery_paths_state, gallery_index_state)
    """
    od = S["output_dir"]
    pending = S["pending"]
    data_preview = None
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
        pending.update({"action": None, "need": None, "args": None})

    if not parsed or parsed.get("tool") != "stats":
        msg = clarify_message(S["df"], llm_error_text, text)
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

    if S["df"] is None:
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
    df = S["df"].copy()

    # --- central resolution for all arg column names ---
    def _rg(name: Optional[str]) -> Optional[str]:
        return _resolve_colname(df, name)
    def _rg_many(names: Optional[List[str]]) -> List[str]:
        return _resolve_many(df, names)

    # Chips and short requests ("Histogram", "ANOVA") omit columns: pick sensible defaults and say so.
    defaults_note = _fill_default_columns(df, action, args)

    sections: List[Tuple[str, str, Optional[np.ndarray]]] = []
    image_paths: List[str] = []

    try:
        # -------- Recommendations --------
        if action == "recommend":
            tech, examples = recommend_text_and_examples(df)
            msg = "Recommendations\n" + tech + "\n\nSay it like this\n" + examples
            sections.append(("Recommendations", msg, None))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": msg}])
            write_report(sections, od)
            zip_fp = save_zip(od)
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
            write_report(sections, od)
            zip_fp = save_zip(od)
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
                image_paths.append(save_image_np(im, f"ttest_plot_{i+1}.png", od))
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
                image_paths.append(save_image_np(im, f"anova_plot_{i+1}.png", od))
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
                image_paths.append(save_image_np(im, f"tukey_plot_{i+1}.png", od))
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
                image_paths.append(save_image_np(im, f"kruskal_plot_{i+1}.png", od))
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
                image_paths.append(save_image_np(im, f"chisq_plot_{i+1}.png", od))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- CORRELATION(S) --------
        elif action == "corr":
            if bool(args.get("matrix", False)):
                method = args.get("method", "pearson")
                cols = _rg_many(args.get("cols"))
                txt, imgs = corr_matrix_plot(df, cols=cols, method=method)
                sections.append(("Correlation matrix", txt, None))
                for i, im in enumerate(imgs):
                    image_paths.append(save_image_np(im, f"corr_matrix_{method}_{i+1}.png", od))
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
                    image_paths.append(save_image_np(im, f"corr_pair_{method}_{i+1}.png", od))
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
                image_paths.append(save_image_np(im, f"pcorr_plot_{i+1}.png", od))
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
                image_paths.append(save_image_np(im, f"pbiserial_plot_{i+1}.png", od))
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
                image_paths.append(save_image_np(im, f"mwutest_plot_{i+1}.png", od))
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
                image_paths.append(save_image_np(im, f"wilcoxon_plot_{i+1}.png", od))
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
                image_paths.append(save_image_np(im, f"ols_plot_{i+1}.png", od))
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
                image_paths.append(save_image_np(im, f"glm_plot_{i+1}.png", od))
            chat_history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": _fence(txt)}])

        # -------- PLOTS --------
        elif action == "plot":
            # Histogram
            if "hist" in args:
                p = args["hist"]
                col = _rg(p.get("col"))
                title, im = plot_hist(df, col, int(p.get("bins", 30)))
                image_paths.append(save_image_np(im, f"plot_hist_{str(col)}.png", od))
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
                image_paths.append(save_image_np(im, f"plot_box_{str(val)}_{str(grp)}.png", od))
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
                image_paths.append(save_image_np(im, f"plot_violin_{str(val)}_{str(grp)}.png", od))
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
                image_paths.append(save_image_np(im, f"plot_bar_{str(val)}_{str(grp)}_{err}.png", od))
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
                image_paths.append(save_image_np(im, f"plot_qq_{str(col)}.png", od))
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
    if defaults_note and chat_history and chat_history[-1].get("role") == "assistant":
        chat_history[-1] = {**chat_history[-1], "content": f"{_message_text(chat_history[-1])}\n\n_{defaults_note}_"}
    write_report(sections, od)
    zip_fp = save_zip(od)
    preview, paths, idx = _safe_last(image_paths)
    return (chat_history, gr.update(value=preview), gr.update(value=zip_fp, visible=True), data_preview, paths, idx)


# =========================
# What the Plots and Files tabs show
# =========================
_EMPTY_PLOTS_HTML = spatchat_ui.empty_state_html(
    "plot", "No plots yet",
    "Load a CSV, then ask for a histogram, box plot, t-test or any analysis. "
    "Every figure stays here so you can compare them.",
)
_SAMPLE_PATH = _SAMPLE_DATA_PATH


def _entries(S: dict):
    """(uploaded, produced) file entries for the Files tab."""
    active = os.path.abspath(S.get("active") or "")

    def up_note(path):
        return "Dataset" + (" (active)" if os.path.abspath(path) == active else "")

    def out_note(path):
        ext = os.path.splitext(path)[1].lower()
        return {".png": "Plot", ".html": "Report", ".zip": "Results bundle", ".csv": "Table"}.get(ext, "")

    uploaded = sfiles.scan(S["upload_dir"], sfiles.UPLOADED, note_for=up_note)
    produced = sfiles.scan(S["output_dir"], sfiles.PRODUCED, note_for=out_note)
    sfiles.stamp_sources(produced, os.path.join(S["root"], ".provenance.json"),
                         os.path.basename(S["active"]) if S.get("active") else "")
    return uploaded, produced


def _map_html(S: dict) -> str:
    """Map tab: the active dataset's points when it has coordinates, otherwise the standard empty map."""
    df = S.get("df")
    if df is None:
        return spatchat_layout.empty_map(
            "Drop a CSV here, or click + to start",
            "Data with latitude and longitude columns appears on this map; any other data is analysed in the Plots tab.")
    xy = spatchat_layout.detect_lonlat(df)
    if xy is None:
        return spatchat_layout.empty_map(
            "No coordinates in this dataset",
            "Add latitude and longitude columns to see it on the map. Every analysis still works without them.")
    return spatchat_layout.points_map(df, xy[0], xy[1],
                                      label=os.path.basename(S["active"]) if S.get("active") else "Points")


def map_for(sid):
    return _map_html(_session(sid)[1])


def _views(sid, open_active=False):
    """Plots gallery + Files tab, from what is on disk. `open_active` opens the dataset as a datasheet."""
    sid, S = _session(sid)
    gal, gal_empty = spatchat_ui.gallery_updates(gr, spatchat_ui.list_images(S["output_dir"]), _EMPTY_PLOTS_HTML)
    uploaded, produced = _entries(S)
    return (gal, gal_empty,
            *files_panel.updates(gr, uploaded, produced, preview_path=S.get("active") if open_active else None))


def _views_open(sid):
    return _views(sid, open_active=True)


def _download_for(S: dict):
    """Show the results zip when there is something to download."""
    if sfiles.scan(S["output_dir"], sfiles.PRODUCED, exts=(".png", ".html", ".csv")):
        return gr.update(value=save_zip(S["output_dir"]), visible=True)
    return gr.update(value=None, visible=False)


def on_upload(file, chat_history, sid):
    chat, _, sid = handle_upload(file, sid, chat_history)
    return chat, sid


def on_sample(chat_history, sid):
    return on_upload(_SAMPLE_PATH, chat_history, sid)


def chat_step(chat_history, user_message, sid):
    sid, S = _session(sid)
    out = _handle_chat_core(chat_history, user_message, S)
    return out[0], out[2], sid


def files_remove(rows, paths, chat_history, sid):
    """Delete the ticked files; if the active dataset goes, fall back to the newest one left (or none)."""
    sid, S = _session(sid)
    removed, problems = sfiles.remove_paths(files_panel.ticked(rows, paths), [S["root"]])
    gone = {os.path.abspath(p) for p in removed}
    chat = list(chat_history or [])
    note = ""
    if S.get("active") and os.path.abspath(S["active"]) in gone:
        left = sfiles.scan(S["upload_dir"], sfiles.UPLOADED)
        _activate(S, left[0]["path"] if left else None)
        note = (f" Now using **{os.path.basename(S['active'])}**." if S.get("active") else " No dataset is loaded now.")
    if gone:
        chat.append({"role": "assistant", "content": sfiles.removal_message(removed) + "." + note})
        if any(p.startswith(os.path.abspath(S["output_dir"])) for p in gone):
            _download = _download_for(S)  # keep the zip in step with what is left
        else:
            _download = gr.skip()
    else:
        _download = gr.skip()
    return (*files_panel.updates(gr, *_entries(S), message=sfiles.removal_message(removed), problems=problems,
                                 reset_preview=bool(removed)),
            chat, _download, sid)


def files_download(rows, paths, sid):
    sid, S = _session(sid)
    return files_panel.download_update(gr, rows, paths, [S["root"]])


_WELCOME_MESSAGE = (
    "Welcome! Upload a CSV, then ask: t-test, ANOVA (+Tukey), Kruskal (+Dunn), correlations (Pearson/Spearman), "
    "partial/pbiserial, chi-square/Fisher, rank tests (Mann-Whitney, Wilcoxon), Levene’s, OLS/GLM, "
    "histogram, box/violin/bar, normality, power, quick summaries, or 'what can I do with my data?'. "
    "If your data has latitude and longitude columns it also shows on the map."
)

_SUGGESTION_CHIPS = [
    "What can I do with my data?",
    "Quick summary",
    "Histogram",
    "Correlations",
    "ANOVA (+Tukey)",
]


with gr.Blocks(title="SpatChat: Stats Room") as demo:
    ui = spatchat_layout.build(
        gr,
        title="Stats Room",
        subtitle="General statistics, data exploration, and plots through conversation.",
        room="stats",
        cite="Wan, H.Y. &amp; Hysen, L. (2025). <i>SpatChat: Stats Room.</i>",
        welcome=_WELCOME_MESSAGE,
        chips=_SUGGESTION_CHIPS,
        placeholder="Message Spatchat...",
        file_types=[".csv"],
        map_html=_map_html({}),
        plots_empty_html=_EMPTY_PLOTS_HTML,
    )
    chatbot, sample_btn, file_input = ui.chatbot, ui.sample_btn, ui.file_input
    user_input, send_btn, download_btn = ui.user_input, ui.send_btn, ui.download_btn
    preview_plot, plots_empty, files_panel = ui.gallery, ui.gallery_empty, ui.files
    session_state, map_out = ui.session_state, ui.map_out

    demo.queue(max_size=16)
    demo.unload(_cleanup_browser_session)

    VIEW_OUTPUTS = [preview_plot, plots_empty, *files_panel.outputs]

    file_input.upload(on_upload, inputs=[file_input, chatbot, session_state], outputs=[chatbot, session_state]
    ).then(_views_open, inputs=session_state, outputs=VIEW_OUTPUTS
    ).then(map_for, inputs=session_state, outputs=map_out)
    sample_btn.click(on_sample, inputs=[chatbot, session_state], outputs=[chatbot, session_state]
    ).then(_views_open, inputs=session_state, outputs=VIEW_OUTPUTS
    ).then(map_for, inputs=session_state, outputs=map_out)

    files_panel.refresh_btn.click(_views, inputs=session_state, outputs=VIEW_OUTPUTS)
    files_panel.remove_btn.click(
        files_remove,
        inputs=[*files_panel.selections, chatbot, session_state],
        outputs=[*files_panel.outputs, chatbot, download_btn, session_state],
    ).then(_views, inputs=session_state, outputs=VIEW_OUTPUTS, show_progress="hidden"
    ).then(map_for, inputs=session_state, outputs=map_out)
    files_panel.table.input(files_download, inputs=[*files_panel.selections, session_state],
                            outputs=files_panel.download_btn, show_progress="hidden")

    gr.on(
        triggers=[user_input.submit, send_btn.click],
        fn=metered_chat(room="stats", message_index=1, deny=_chat_denied)(chat_step),
        inputs=[chatbot, user_input, session_state],
        outputs=[chatbot, download_btn, session_state],
    ).then(_views, inputs=session_state, outputs=VIEW_OUTPUTS, show_progress="hidden")
    gr.on(triggers=[user_input.submit, send_btn.click], fn=lambda *args: "", inputs=None, outputs=user_input)


if __name__ == "__main__":
    demo.launch(ssr_mode=False, head=spatchat_ui.HEAD)
