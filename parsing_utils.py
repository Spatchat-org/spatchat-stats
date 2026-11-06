# parsing_utils.py

from typing import Dict, List, Optional
import json
import re
import pandas as pd

# Pull dataset-aware guidance for clarify_message()
from stats.recommendations import recommend_text_and_examples

# --------------------------
# PROMPTS
# --------------------------
SYSTEM_PROMPT = """
You are SpatChat, an expert statistics assistant for basic analyses.
When the user asks for an analysis, respond ONLY in compact JSON using this schema:
{"tool":"stats","action":"ttest|anova|ols|glm|plot|check|power|summary|recommend|chisq|corr|mwutest|wilcoxon|kruskal|levene|posthoc_tukey|posthoc_dunn|pcorr|pbiserial","args":{...}}

Actions and args:
- ttest: {"value":"colY","group":"colG", "paired": false, "equal_var": false}
- anova: {"value":"colY","group":"colG"}
- posthoc_tukey: {"value":"colY","group":"colG"}
- kruskal: {"value":"colY","group":"colG"}
- posthoc_dunn: {"value":"colY","group":"colG","p_adjust":"holm"}
- ols: {"formula":"y ~ x1 + x2"}
- glm: {"formula":"y ~ x1 + x2", "family":"gaussian|binomial|poisson|gamma"}
- chisq: {"row":"catA","col":"catB","exact": false}
- corr: {"x":"col1","y":"col2","method":"pearson|spearman","matrix": false, "cols": ["colA","colB", "..."]}
- pcorr: {"x":"col1","y":"col2","controls":["c1","c2"],"method":"pearson|spearman"}
- pbiserial: {"value":"numericCol","group":"binaryCol"}
- mwutest: {"value":"colY","group":"colG"}
- wilcoxon: {"a":"colA","b":"colB"}
- levene: {"value":"colY","group":"colG","center":"median|mean|trimmed"}
- plot: one of {"hist":{"col":"c","bins":30}, "box":{"value":"y","group":"g"}, "violin":{"value":"y","group":"g"}, "bar":{"value":"y","group":"g","error":"sem|sd|ci95"}}
- check: {"normality":{"col":"y"}}
- power: one of
  {"ttest_ind": {"effect_size": 0.5, "alpha": 0.05, "power": 0.8, "ratio": 1.0, "solve_for":"n_total|power|effect_size"}}
  {"anova_oneway": {"effect_size": 0.25, "k_groups": 3, "alpha": 0.05, "power": 0.8, "solve_for":"n_per_group|power|effect_size"}}
- summary: {"col":"y", "by":"group_col_or_null"}
- recommend: {}

If unclear, pick sensible defaults from the dataset.
NEVER include prose; JSON only.
For general questions, answer in <=3 sentences of plain text.
""".strip()

FALLBACK_PROMPT = """
You are SpatChat, a concise statistics tutor. If you can't map to a tool call, answer naturally in <=3 sentences.
""".strip()

# --------------------------
# LLM bridge
# --------------------------
_LLM = None  # set once by app.py via set_llm()

def set_llm(llm_obj) -> None:
    """Called by app.py after creating the UnifiedLLM."""
    global _LLM
    _LLM = llm_obj

def ask_llm(chat_history, user_input):
    if _LLM is None:
        raise RuntimeError("LLM client is not set. Call parsing_utils.set_llm(llm) in app.py after creating the client.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(chat_history) + [{"role": "user", "content": user_input}]
    resp = _LLM.chat(messages=messages, temperature=0.0, max_tokens=256, stream=False)
    try:
        call = json.loads(resp)
        return call, resp
    except Exception:
        conv = _LLM.chat(
            messages=[{"role": "system", "content": FALLBACK_PROMPT}] + messages,
            temperature=0.7,
            max_tokens=256,
            stream=False,
        )
        return None, conv

# --------------------------
# Argument sanitization
# --------------------------
_NULLY_STRINGS = {"", "null", "none", "na", "n/a", "nil"}

def clean_null_like(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    return None if s in _NULLY_STRINGS else v

def sanitize_args(action: Optional[str], args: Dict) -> Dict:
    if not isinstance(args, dict):
        return {}
    if action == "summary":
        args["by"] = clean_null_like(args.get("by"))
    elif action in {"ttest", "anova", "mwutest", "kruskal", "levene", "posthoc_tukey", "posthoc_dunn", "pbiserial"}:
        args["group"] = clean_null_like(args.get("group"))
    elif action == "plot":
        for k in ("box", "violin", "bar"):
            if k in args and isinstance(args[k], dict):
                args[k]["group"] = clean_null_like(args[k].get("group"))
    elif action == "chisq":
        args["row"] = clean_null_like(args.get("row"))
        args["col"] = clean_null_like(args.get("col"))
    elif action == "corr":
        args["x"] = clean_null_like(args.get("x"))
        args["y"] = clean_null_like(args.get("y"))
    elif action == "pcorr":
        args["x"] = clean_null_like(args.get("x"))
        args["y"] = clean_null_like(args.get("y"))
        if isinstance(args.get("controls"), list):
            args["controls"] = [c for c in args["controls"] if clean_null_like(c)]
    elif action == "wilcoxon":
        args["a"] = clean_null_like(args.get("a"))
        args["b"] = clean_null_like(args.get("b"))
    return args

# --------------------------
# Local regex parser (UPDATED)
# --------------------------
def _split_controls(s: str) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in re.split(r"[,+]", s) if p.strip()]

_COL = r"([A-Za-z0-9_.]+)"
_STOP = {"a", "an", "the", "of", "on", "for", "my", "me", "data", "dataset", "please", "test", "plot", "chart"}
_NOT_COLS = {"bins", "with", "and", "by", "show", "make", "draw", "give", "get", "run", "do", "i", "want", "to",
             "check", "qq", "shapiro", "normality", "is", "are", "histogram", "hist", "in", "at", "using", "use"}


def _value_group(rest: str):
    """'on score by sex' -> ('score', 'sex'); tolerant of filler words."""
    m = re.search(rf"(?:\b(?:on|of|for)\s+)?{_COL}\s+(?:by|across|per|between|vs\.?|versus|and|grouped by)\s+{_COL}", rest)
    if m and m.group(1) not in _STOP and m.group(2) not in _STOP:
        return m.group(1), m.group(2)
    return None, None


def _one_col(rest: str):
    for tok in re.findall(r"[A-Za-z0-9_.]+", rest):
        if tok in _STOP or tok in _NOT_COLS or re.fullmatch(r"\d+(?:\.\d+)?", tok):
            continue
        return tok
    return None


def _tail(s: str, pattern: str) -> str:
    m = re.search(pattern, s)
    return s[m.end():] if m else ""


def _stats(action, **args):
    return {"tool": "stats", "action": action, "args": args}


def _local_parse_analyses(s: str) -> Optional[Dict]:
    """Rule-based intent parser, so every analysis works without an LLM. Column names are resolved
    case-insensitively against the dataset by the caller."""
    # ---------- plots ----------
    if re.search(r"\bhist(?:ogram)?\b", s):
        col = _one_col(_tail(s, r"\bhist(?:ogram)?\b"))
        bins = re.search(r"\b(\d+)\s*bins?\b|\bbins?\s*(?:=|of)?\s*(\d+)", s)
        n = int(next(g for g in bins.groups() if g)) if bins else 30
        return _stats("plot", hist={"col": col, "bins": n})
    for kind, pat in (("violin", r"\bviolin(?:\s*plot)?\b"),
                      ("box", r"\bbox\s*plot\b|\bboxplot\b|\bbox\b"),
                      ("bar", r"\bbar\s*(?:plot|chart|graph)?\b")):
        if re.search(pat, s):
            rest = _tail(s, pat)
            v, g = _value_group(rest)
            spec = {"value": v or _one_col(rest), "group": g}
            if kind == "bar":
                spec["error"] = "sd" if re.search(r"\bsd\b|std", s) else "ci95" if re.search(r"\bci\b|confidence", s) else "sem"
            return _stats("plot", **{kind: spec})
    if re.search(r"\bnormality\b|\bqq\b|\bq-q\b|\bshapiro\b|\bnormally distributed\b", s):
        return _stats("check", normality={"col": _one_col(_tail(s, r"\b(?:normality|qq(?:\s*plot)?|q-q(?:\s*plot)?|shapiro)\b"))})

    # ---------- power ----------
    if re.search(r"\bpower\b|\bsample size\b", s):
        es = re.search(r"(?:effect size|cohen'?s d|\bd|\bf)\s*(?:=|of|is)?\s*(\d*\.?\d+)", s)
        al = re.search(r"alpha\s*(?:=|of|is)?\s*(\d*\.?\d+)", s)
        pw = re.search(r"power\s*(?:=|of|is)\s*(\d*\.?\d+)", s)
        if re.search(r"anova", s):
            k = re.search(r"(\d+)\s*groups?", s)
            return _stats("power", anova_oneway={
                "effect_size": float(es.group(1)) if es else 0.25, "k_groups": int(k.group(1)) if k else 3,
                "alpha": float(al.group(1)) if al else 0.05, "power": float(pw.group(1)) if pw else 0.8,
                "solve_for": "n_per_group"})
        return _stats("power", ttest_ind={
            "effect_size": float(es.group(1)) if es else 0.5, "alpha": float(al.group(1)) if al else 0.05,
            "power": float(pw.group(1)) if pw else 0.8, "ratio": 1.0, "solve_for": "n_total"})

    # ---------- models ----------
    m = re.search(r"\b(ols|linear regression|lm|glm)\b[^~]*?([A-Za-z0-9_.]+\s*~[^,;]+)", s)
    if m:
        formula = m.group(2).strip()
        if m.group(1) == "glm":
            fam = re.search(r"famil(?:y|ies)\s*=?\s*(gaussian|binomial|poisson|gamma)|\b(binomial|poisson|gamma|gaussian)\b", s)
            family = next((g for g in fam.groups() if g), "gaussian") if fam else "gaussian"
            formula = re.sub(r"\s*(?:with\s+)?famil(?:y|ies)\s*=?\s*\w+\s*$", "", formula)
            return _stats("glm", formula=formula, family=family)
        return _stats("ols", formula=formula)

    # ---------- correlations ----------
    if re.search(r"\bpartial\s+corr", s):
        rest = _tail(s, r"\bpartial\s+corr\w*")
        m = re.search(rf"{_COL}\s+(?:and|vs\.?|with)\s+{_COL}\s+(?:controlling|adjusting|holding|while controlling)\s*(?:for)?\s*(.+)", rest)
        method = "spearman" if "spearman" in s else "pearson"
        if m:
            ctrls = [c for c in re.findall(r"[A-Za-z0-9_.]+", m.group(3)) if c not in _STOP and c != "and"]
            return _stats("pcorr", x=m.group(1), y=m.group(2), controls=ctrls, method=method)
        return _stats("pcorr")
    if re.search(r"point[-\s]?biserial", s):
        v, g = _value_group(_tail(s, r"point[-\s]?biserial"))
        return _stats("pbiserial", value=v, group=g)
    if re.search(r"\bcorrelations?\b|\bcorr\b|\bpearson\b|\bspearman\b", s):
        method = "spearman" if "spearman" in s else "pearson"
        rest = _tail(s, r"\b(?:correlation|corr|pearson|spearman)\b")
        m = re.search(rf"(?:\b(?:of|between)\s+)?{_COL}\s+(?:and|vs\.?|with|versus)\s+{_COL}", rest)
        if m and m.group(1) not in _STOP and not re.search(r"matrix|heatmap", s):
            return _stats("corr", x=m.group(1), y=m.group(2), method=method, matrix=False)
        return _stats("corr", matrix=True, method=method, cols=[])

    # ---------- group comparisons ----------
    for pat, action, extra in (
        (r"\banova\b|\banalysis of variance\b", "anova", {}),
        (r"\btukey\b", "posthoc_tukey", {}),
        (r"\bdunn\b", "posthoc_dunn", {"p_adjust": "holm"}),
        (r"\bkruskal\b", "kruskal", {}),
        (r"\blevene\b", "levene", {"center": "median"}),
        (r"\bmann[-\s]?whitney\b|\bmwu\b|\brank[-\s]?sum\b", "mwutest", {}),
    ):
        if re.search(pat, s):
            v, g = _value_group(_tail(s, pat))
            return _stats(action, value=v, group=g, **extra)
    if re.search(r"\bwilcoxon\b|\bsigned[-\s]?rank\b", s):
        m = re.search(rf"{_COL}\s+(?:vs\.?|versus|and|with)\s+{_COL}", _tail(s, r"\bwilcoxon\b|\bsigned[-\s]?rank\b"))
        return _stats("wilcoxon", a=m.group(1) if m else None, b=m.group(2) if m else None)
    if re.search(r"\bt[-\s]?test\b|\bttest\b", s):
        v, g = _value_group(_tail(s, r"\bt[-\s]?test\b|\bttest\b"))
        return _stats("ttest", value=v, group=g, paired=bool(re.search(r"\bpaired\b", s)),
                      equal_var=bool(re.search(r"\bequal var|\bstudent", s)))
    if re.search(r"\bchi[-\s]?(?:square|squared|2)\b|\bfisher\b|\bcontingency\b", s):
        m = re.search(rf"{_COL}\s+(?:by|and|vs\.?|versus|with|against)\s+{_COL}",
                      _tail(s, r"\bchi[-\s]?(?:square|squared|2)\b|\bfisher\b|\bcontingency\b"))
        ok = bool(m) and m.group(1) not in _STOP and m.group(2) not in _STOP
        return _stats("chisq", row=m.group(1) if ok else None, col=m.group(2) if ok else None,
                      exact=bool(re.search(r"fisher|exact", s)))
    return None


def local_parse(user_input: str) -> Optional[Dict]:
    s = user_input.strip().lower()

    def _summary_args(by_val=None):
        return {"tool": "stats", "action": "summary", "args": {"col": "data", "by": by_val}}

    # Bare "summary" / "quick summary" -> whole-dataset summary
    if re.fullmatch(r"(?:a\s+|the\s+)?(?:quick\s+|brief\s+)?(?:summary|summari[sz]e|describe)(?:\s+please)?[.!?]*", s):
        return _summary_args()

    # --- Summarize data/dataset/all ---
    m = re.search(r"\b(summarize|summary|describe)\s+(?:the\s+)?(data|dataset|everything|\*)\s+(?:by|across)\s+([a-zA-Z0-9_]+)", s)
    if m:
        return _summary_args(by_val=m.group(3))

    # Just "summarize data/dataset" (no grouping)
    m = re.search(r"\b(summarize|summary|describe)\s+(?:the\s+)?(data|dataset|everything|\*)\b", s)
    if m:
        return _summary_args()

    # "summarize by sex"
    m = re.search(r"\b(summarize|summary|describe)\s+(?:by|across)\s+([a-zA-Z0-9_]+)", s)
    if m:
        return _summary_args(by_val=m.group(2))

    # Recommendation
    if re.search(r"\b(what can i do|how should i analyze|recommend|suggestion|what analyses)\b", s):
        return {"tool": "stats", "action": "recommend", "args": {}}

    # Summarize a specific column
    m = re.search(r"(?:mean|average|summary|summarize|describe)\s+([a-zA-Z0-9_]+)(?:\s+by\s+([a-zA-Z0-9_]+))?", s)
    if m:
        return {"tool": "stats", "action": "summary", "args": {"col": m.group(1), "by": m.group(2) if m.group(2) else None}}

    parsed = _local_parse_analyses(s)
    if parsed:
        return parsed

    # Finally, allow raw JSON if user typed it
    try:
        maybe = json.loads(user_input)
        if isinstance(maybe, dict) and "tool" in maybe:
            return maybe
    except Exception:
        pass
    return None

# --------------------------
# Clarification helper
# --------------------------
def clarify_message(df: Optional[pd.DataFrame], llm_error_text: Optional[str], user_text: str) -> str:
    prefix = "Sorry — I didn’t understand that request."
    if llm_error_text:
        prefix = f"{prefix} {llm_error_text}"

    examples = [
        "What is the average score?",
        "Show a histogram of height.",
        "I want to do a t-test on score by sex.",
        "Kruskal-Wallis score by group.",
        "Tukey HSD score by group.",
        "Dunn's test score by group.",
        "Pearson correlation height and weight.",
        "Partial correlation height and score controlling for age, weight.",
        "Point-biserial score by sex.",
        "Chi-square sex by group.",
        "Check normality of score (QQ plot).",
    ]

    if df is not None:
        try:
            tech, ex = recommend_text_and_examples(df)
            return f"{prefix}\n\nCould you clarify what you want to do?\n\n{tech}\n\nSay it like this\n{ex}"
        except Exception:
            pass

    cols = list(map(str, df.columns)) if df is not None else []
    col_text = f"\nDetected columns: {', '.join(cols)}" if cols else ""
    return f"{prefix}\n\nTry one of these examples:\n• " + "\n• ".join(examples) + col_text
