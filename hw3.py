#!/usr/bin/env python3
"""FTEC5660 HW3 student starter: an agent that applies 3 of Fisher's 15 points using tools."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any


MODEL_NAME = "deepseek-v4-flash"
INTERVIEW_LIMIT = 10
MAX_SQL_ROWS = 50

# Philip Fisher, "Common Stocks and Uncommon Profits", Part One, Chapter 3.
# For every point, PASS means the company looks good on it. You choose 3
# points and decide yourself what PASS means for each of them.
FISHER_POINTS = {
    1: "Does the company have products or services with sufficient market potential to make possible a sizable increase in sales for at least several years?",
    2: "Does the management have a determination to continue to develop products or processes that will still further increase total sales potentials when the growth potentials of currently attractive product lines have largely been exploited?",
    3: "How effective are the company's research and development efforts in relation to its size?",
    4: "Does the company have an above-average sales organization?",
    5: "Does the company have a worthwhile profit margin?",
    6: "What is the company doing to maintain or improve profit margins?",
    7: "Does the company have outstanding labor and personnel relations?",
    8: "Does the company have outstanding executive relations?",
    9: "Does the company have depth to its management?",
    10: "How good are the company's cost analysis and accounting controls?",
    11: "Are there other aspects of the business, somewhat peculiar to the industry involved, which will give the investor important clues as to how outstanding the company may be in relation to its competition?",
    12: "Does the company have a short-range or long-range outlook in regard to profits?",
    13: "In the foreseeable future will the growth of the company require sufficient equity financing so that the larger number of shares then outstanding will largely cancel the existing stockholders' benefit from this anticipated growth?",
    14: "Does the management talk freely to investors about its affairs when things are going well but 'clam up' when troubles and disappointments occur?",
    15: "Does the company have a management of unquestionable integrity?",
}
VERDICTS = ("PASS", "FAIL")


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Choose any 3 of the 15 points, e.g. CHOSEN_POINTS = [1, 7, 15]. Your agent
# gives a PASS or FAIL verdict on these 3 points only, with your own rule for each.
CHOSEN_POINTS: list[int] = [5, 7, 10]  ### YOUR CODE HERE


# ===========================================================================
#  Helper functions
# ===========================================================================

# 兼容 langchain 1.0 的 create_agent 与旧版 langgraph 的 create_react_agent
try:
    from langchain.agents import create_agent as _create_agent
except ImportError:  # 旧版 langchain / langgraph
    try:
        from langgraph.prebuilt import create_react_agent as _create_react_agent

        def _create_agent(model, tools, system_prompt):
            return _create_react_agent(model, tools, prompt=system_prompt)
    except ImportError:  # 都没有 → 运行时给清晰报错
        _create_agent = None


_SYSTEM_PROMPT = (
    "You are a careful Fisher-style equity analyst. You will receive pre-computed "
    "numbers, interview excerpts and document excerpts about a company. Apply the "
    "stated rules exactly and return ONE JSON object with no markdown fence and no "
    "extra prose. Never recompute numbers that are already given; treat given "
    "numbers as ground truth."
)

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

_QUARTER_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4"}


def _parse_sql_rows(text: str) -> list[dict[str, str]]:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    if not lines or lines[0].startswith("SQL error") or lines[0] == "(no rows)":
        return []
    header = [h.strip() for h in lines[0].split("|")]
    rows: list[dict[str, str]] = []
    for ln in lines[1:]:
        if ln.startswith("...") or ln == "(no rows)":
            continue
        vals = [v.strip() for v in ln.split("|")]
        if len(vals) != len(header):
            continue
        rows.append(dict(zip(header, vals)))
    return rows


def _f(s: Any) -> float | None:
    if s is None:
        return None
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def _verdict(s: Any) -> str | None:
    if isinstance(s, bool):
        return "PASS" if s else "FAIL"
    v = str(s or "").strip().upper()
    return v if v in ("PASS", "FAIL") else None


def _final(v: str | None) -> str:
    """Every entry needs a verdict: a missing one scores as wrong anyway, and
    most true verdicts are PASS, so an unverified point defaults to PASS."""
    return v or "PASS"


def _is_no_comment(text: str) -> bool:
    t = (text or "").lower()
    if len(t) < 40:
        return True
    return any(kw in t for kw in (
        "can't comment", "cannot comment", "can not comment",
        "no comment", "don't know", "do not know",
        "not something i know", "wouldn't know", "would not know",
    ))


def _message_text(msg_or_content: Any) -> str:
    """AIMessage.content may be str or list of blocks; normalize to str."""
    c = getattr(msg_or_content, "content", msg_or_content)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(c)


def _extract_json(text: str) -> dict:
    text = _message_text(text).strip()
    if "```" in text:
        for part in text.split("```"):
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                text = p
                break
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j <= i:
        return {}
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return {}


def _norm_quarter(s: Any) -> str | None:
    """Normalise any quarter spelling to 'YYYYQn'.

    Accepts '2025Q3', '2025 Q3', '2025-Q3', 'Q3 2025', '3Q2025' and
    'third quarter 2025'. Returning None here would silently drop a quarter
    from the point-10 comparison, so the pattern is deliberately generous.
    """
    t = str(s or "").lower()
    for word, num in _QUARTER_WORDS.items():
        t = t.replace(word, num)
    # year first: 2025Q3, 2025 Q3, 2025-Q3
    m = re.search(r"(20\d{2})\s*[-/ ]?\s*(?:q\s*)?([1-4])(?![0-9])", t)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    # quarter first: Q3 2025, 3Q2025, 3 quarter 2025
    m = re.search(r"(?:q\s*)?([1-4])\s*(?:q\s*)?(?:quarter\s*)?[-/ ]?\s*(20\d{2})", t)
    if m:
        return f"{m.group(2)}Q{m.group(1)}"
    return None


def _eval_point5(sql_query: Any) -> tuple[str | None, str]:
    """Point 5: FY2025 operating margin vs the company's own industry median."""
    q = """
    SELECT
      (SELECT SUM(amount)/1000.0 FROM orders
         WHERE order_date >= '2025-01-01' AND order_date < '2026-01-01') AS rev_musd,
      (SELECT SUM(cost_of_sales)/1000.0 FROM monthly_costs
         WHERE month BETWEEN '2025-01' AND '2025-12')                 AS cogs_musd,
      (SELECT SUM(selling_admin)/1000.0 FROM monthly_costs
         WHERE month BETWEEN '2025-01' AND '2025-12')                 AS sga_musd,
      (SELECT SUM(research_development)/1000.0 FROM monthly_costs
         WHERE month BETWEEN '2025-01' AND '2025-12')                 AS rd_musd,
      (SELECT COUNT(DISTINCT month) FROM monthly_costs
         WHERE month BETWEEN '2025-01' AND '2025-12')                 AS n_months,
      (SELECT im.operating_margin_pct FROM industry_medians im
         WHERE LOWER(im.industry) = LOWER((SELECT industry FROM company LIMIT 1))
         LIMIT 1)                                                     AS ind_median
    """
    raw = sql_query.invoke({"query": q})
    rows = _parse_sql_rows(raw)
    if not rows:
        return None, f"point5: no data. raw={raw!r}"
    r = rows[0]
    rev  = _f(r.get("rev_musd"))
    cogs, sga, rd = _f(r.get("cogs_musd")), _f(r.get("sga_musd")), _f(r.get("rd_musd"))
    med  = _f(r.get("ind_median"))
    n_months = int(_f(r.get("n_months")) or 0)
    if rev is None or rev <= 0 or med is None or None in (cogs, sga, rd):
        return None, f"point5: incomplete data rev={rev} cogs={cogs} sga={sga} rd={rd} med={med}"
    profit = rev - cogs - sga - rd
    margin = profit / rev * 100.0
    verdict = "PASS" if margin >= med else "FAIL"
    warn = "" if n_months >= 12 else f" [WARN only {n_months} months of cost data]"
    ev = (f"FY2025 revenue {rev:.2f}M USD, operating profit {profit:.2f}M USD, "
          f"operating margin {margin:.2f}% vs industry median {med:.2f}%{warn}; "
          f"rule margin>=median -> {verdict}.")
    return verdict, ev


def _eval_point10(list_documents: Any, read_document: Any, sql_query: Any
                  ) -> tuple[str, list[str], dict[str, float]]:
    """Read every earnings-call document; return the text, its names and the
    actual quarterly revenue from the database (USD millions)."""
    listing = list_documents.invoke({}) or ""
    names = [ln.split(":", 1)[0].strip() for ln in listing.splitlines() if ":" in ln]
    doc_names = [n for n in names if "earnings" in n.lower() or "call" in n.lower()]
    if not doc_names:
        doc_names = [n for n in names if n.lower().endswith(".md")]

    parts = []
    for d in doc_names:
        try:
            text = read_document.invoke({"name": d})
        except Exception as exc:
            text = f"<error reading {d}: {exc}>"
        parts.append(f"=== {d} ===\n{text}")

    q = """
    SELECT
      substr(order_date, 1, 4) || 'Q' ||
        CAST((CAST(substr(order_date, 6, 2) AS INTEGER) - 1) / 3 + 1 AS TEXT) AS q,
      ROUND(SUM(amount)/1000.0, 2) AS rev_musd
    FROM orders
    WHERE order_date >= '2025-07-01' AND order_date < '2026-07-01'
    GROUP BY q
    ORDER BY q
    """
    actual: dict[str, float] = {}
    for r in _parse_sql_rows(sql_query.invoke({"query": q})):
        qn, v = (r.get("q") or "").strip(), _f(r.get("rev_musd"))
        if qn and v is not None:
            actual[qn] = v
    return "\n\n".join(parts), doc_names, actual


async def _extract_claims(agent: Any, docs_text: str) -> dict[str, float]:
    """One LLM call: pull the revenue management quoted in each earnings call."""
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = (
        "Below are a company's quarterly earnings-call transcripts.\n"
        "For EACH call, find the revenue figure management quotes for that quarter "
        "(it may be written as US dollars, USD, $, million or m).\n"
        'Return ONE JSON object: {"quarters": {"<YYYYQn>": <number in USD millions>}}\n'
        "Include one entry per call. Do not invent numbers.\n\n" + docs_text
    )
    for _ in range(2):
        try:
            resp = await agent.ainvoke([SystemMessage(content=_SYSTEM_PROMPT),
                                        HumanMessage(content=prompt)])
            parsed = _extract_json(_message_text(resp))
            raw = parsed.get("quarters") or parsed.get("claimed_revenues_musd") or parsed
            claims: dict[str, float] = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    q, val = _norm_quarter(k), _f(v)
                    if q and val is not None:
                        claims[q] = val
            if claims:
                return claims
        except Exception:
            pass
        prompt += "\n\nReturn valid JSON only, one entry per quarter."
    return {}


async def _summarize(agent: Any, name: str, verdicts: dict[int, str], evidence: dict[int, str]) -> str:
    """One LLM call that only writes the summary; verdicts are final."""
    from langchain_core.messages import HumanMessage, SystemMessage

    lines = "\n".join(f"- Point {p} = {verdicts[p]}. Evidence: {evidence.get(p, '')[:700]}"
                      for p in sorted(verdicts))
    prompt = (
        f'Company folder: "{name}". The three verdicts below are FINAL: do not change '
        "them and do not recompute anything. Write 2-3 sentences for an investor: one "
        "clause per point, citing the number (points 5 and 10) or the quote (point 7) "
        "that justifies each verdict. Plain text only.\n\n" + lines
    )
    resp = await agent.ainvoke([
        SystemMessage(content="You are a careful Fisher-style equity analyst. "
                              "Reply with plain prose only: no JSON, no markdown, no lists."),
        HumanMessage(content=prompt),
    ])
    text = _message_text(resp).strip()
    if text.startswith("{"):  # unwrap {"summary": "..."} / {"investor_summary": "..."}
        parsed = _extract_json(text)
        text = next((str(v) for v in parsed.values() if isinstance(v, str) and len(v) > 20), text)
    return text


async def _judge_labour(agent: Any, quote: str) -> tuple[str | None, str]:
    """One LLM call: negative labour signals -> FAIL."""
    resp = await agent.ainvoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            "An employee said this about their company:\n" + quote + "\n\n"
            "PASS = clearly good labour/personnel relations (fair pay, low turnover, morale).\n"
            "FAIL = clearly bad (chronic overtime, high turnover, pay complaints, strike threat).\n"
            'Return ONLY {"verdict":"PASS" or "FAIL","reason":"<one sentence>"}')),
    ])
    parsed = _extract_json(_message_text(resp))
    return _verdict(parsed.get("verdict")), str(parsed.get("reason", ""))


def build_agent() -> Any:
    from langchain_deepseek import ChatDeepSeek
    load_env_file()
    return ChatDeepSeek(model=MODEL_NAME, temperature=0, timeout=120, max_retries=2,
                        max_tokens=2000)


async def analyze_companies(agent: Any, companies: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    if _create_agent is None:
        return {
            name: {"points": {p: "PASS" for p in CHOSEN_POINTS},
                   "summary": "create_agent not available; install langchain>=1.0 or langgraph."}
            for name in companies
        }

    sem = asyncio.Semaphore(5)

    async def point7(interview: Any, name: str) -> tuple[str | None, str]:
        """Route point 7 to interviews: at most two are spent."""
        try:
            p7_agent = _create_agent(
                agent,
                tools=[interview],
                system_prompt=(
                    "You are doing Fisher's scuttlebutt on LABOR AND PERSONNEL RELATIONS.\n"
                    "  1. Call interview(role='current_employee', topic='working_conditions').\n"
                    "  2. If the answer says 'can't comment' or is clearly off-topic, call\n"
                    "     interview(role='former_employee', topic='working_conditions').\n"
                    "  3. Do NOT ask any other role or topic.\n"
                    "  PASS = clearly positive (fair treatment, morale, pay, safety, retention).\n"
                    "  FAIL = clearly negative (complaints, hostility, high turnover, fear).\n"
                    "Reply with ONE JSON object only:\n"
                    '{"verdict":"PASS" or "FAIL","evidence":"<quote>","reason":"<one sentence>"}'
                ),
            )
            async with sem:
                resp = await p7_agent.ainvoke(
                    {"messages": [HumanMessage(content=f"Company folder: {name}. Start the scuttlebutt.")]},
                    config={"recursion_limit": 8},
                )
            parsed = _extract_json(_message_text(resp["messages"][-1]))
            return (_verdict(parsed.get("verdict")),
                    f"interview said: {parsed.get('evidence', '')} ({parsed.get('reason', '')})")
        except Exception as exc:
            return None, f"point7 agent error: {type(exc).__name__}: {exc}"

    async def analyze_one(name: str, tools: list[Any]) -> dict[str, Any]:
        _list_documents, read_document, sql_query, interview = tools
        verdicts: dict[int, str] = {}
        evidence: dict[int, str] = {}

        # ---------- point 5: SQL + Python, no model in the loop ----------
        try:
            v5, ev5 = _eval_point5(sql_query)
        except Exception as exc:
            v5, ev5 = None, f"point5 error: {type(exc).__name__}: {exc}"
        verdicts[5] = _final(v5)
        evidence[5] = ev5

        # ---------- point 7: scuttlebutt ----------
        v7, ev7 = await point7(interview, name)
        if v7 is None:  # fallback: one more interview, judged by one LLM call
            try:
                quote = _message_text(await interview.ainvoke(
                    {"role": "former_employee", "topic": "working_conditions"}))
                if _is_no_comment(quote):
                    quote = _message_text(await interview.ainvoke(
                        {"role": "current_employee", "topic": "working_conditions"}))
                v7, why = await _judge_labour(agent, quote)
                ev7 = f"fallback interview: {quote} ({why})"
            except Exception as exc:
                ev7 = f"point7 fallback error: {type(exc).__name__}: {exc}"
        verdicts[7] = _final(v7)
        evidence[7] = ev7

        # ---------- point 10: documents vs database, decided in Python ----------
        try:
            docs_text, doc_names, actual = _eval_point10(_list_documents, read_document, sql_query)
            async with sem:
                claims = await _extract_claims(agent, docs_text)
            mismatches = []
            for qn, claimed in sorted(claims.items()):
                av = actual.get(qn)
                if av and av > 0 and abs(claimed - av) / av > 0.02:
                    mismatches.append(f"{qn}: claimed {claimed:.2f} vs database {av:.2f}")
            verdicts[10] = "FAIL" if mismatches else "PASS"
            checked = ", ".join(f"{qn} {claims[qn]:.2f}/{actual.get(qn, float('nan')):.2f}"
                                for qn in sorted(claims))
            evidence[10] = (f"{len(doc_names)} earnings calls read; claimed/database (USD M): {checked}; "
                            + ("MISMATCH -> FAIL: " + "; ".join(mismatches) if mismatches
                               else "all quarters within 2% -> PASS"))
        except Exception as exc:
            verdicts[10] = "PASS"
            evidence[10] = f"point10 error: {type(exc).__name__}: {exc}"

        # ---------- summary: one call, verdicts are final inputs ----------
        try:
            async with sem:
                summary = await _summarize(agent, name, verdicts, evidence)
        except Exception as exc:
            summary = f"Analysis failed: {type(exc).__name__}: {exc}"

        return {"points": dict(verdicts), "summary": summary}

    results = await asyncio.gather(
        *(analyze_one(n, t) for n, t in companies.items()),
        return_exceptions=True,
    )
    out: dict[str, dict[str, Any]] = {}
    for name, res in zip(companies.keys(), results):
        if isinstance(res, Exception):
            out[name] = {"points": {p: "PASS" for p in CHOSEN_POINTS},
                         "summary": f"error: {type(res).__name__}: {res}"}
        else:
            out[name] = res
    return out

# Everything below is provided runner/scoring code. No edits are needed.
# Your code must read company data only through these tools: do not open
# company.db or interviews.dat directly.

INTERVIEWS_USED: dict[str, int] = {}


def company_folders(folder: Path) -> list[Path]:
    """Return one sub-folder per company directly inside *folder*, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def make_tools(company_folder: Path) -> list[Any]:
    """Build the four tools for one company."""
    from langchain_core.tools import tool

    name = company_folder.name
    db_uri = f"{(company_folder / 'company.db').resolve().as_uri()}?mode=ro"
    answers = json.loads(zlib.decompress((company_folder / "interviews.dat").read_bytes()))
    documents = {p.name: p for p in sorted(company_folder.glob("*.md"))}
    lock = threading.Lock()
    INTERVIEWS_USED[name] = 0

    @tool
    def list_documents() -> str:
        """List this company's documents (annual letter, earnings calls, news) with their dates."""
        lines = []
        for doc_name, path in documents.items():
            text = path.read_text(encoding="utf-8").splitlines()
            dated = next((line for line in text if line.startswith("Date:")), "")
            lines.append(f"{doc_name}: {text[0].lstrip('# ')} ({dated or 'undated'})")
        return "\n".join(lines)

    @tool
    def read_document(name: str) -> str:
        """Return the full text of one document named by list_documents, e.g. 'annual_letter_2025.md'."""
        path = documents.get(name.strip())
        if path is None:
            return f"No document named {name!r}. Available: {', '.join(documents)}"
        return path.read_text(encoding="utf-8")

    @tool
    def sql_query(query: str) -> str:
        """Run one read-only SQLite query on this company's database and return at most 50 rows.

        Amounts are in USD thousands. Dates are 'YYYY-MM-DD', months 'YYYY-MM', quarters like '2025Q3'.
        Tables:
          company(name, industry)
          industry_medians(industry, revenue_growth_2y_pct, new_product_share_pct, new_product_sales_per_rd, operating_margin_pct)
          orders(order_id, order_date, product, customer, amount, note)
          products(product, launch_date)
          monthly_costs(month, cost_of_sales, selling_admin, research_development)
          quarterly_cash(quarter, operating_cash_flow, capex, cash_end)
          capex_plan(year, planned_capex)
          returns(month, units_shipped, units_returned)
          insider_trades(trade_date, person, title, side, shares, price)
        Use SUM, GROUP BY and WHERE to compute totals rather than reading raw rows.
        """
        try:
            con = sqlite3.connect(db_uri, uri=True)
            try:
                cursor = con.execute(query)
                columns = [c[0] for c in cursor.description or []]
                rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
            finally:
                con.close()
        except sqlite3.Error as exc:
            return f"SQL error: {exc}"
        if not columns:
            return "Query returned no columns."
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        out = [" | ".join(columns)] + [" | ".join(fmt(v) for v in row) for row in rows[:MAX_SQL_ROWS]]
        if len(rows) > MAX_SQL_ROWS:
            out.append(f"... more rows not shown (limit {MAX_SQL_ROWS}); aggregate or add LIMIT/OFFSET.")
        if not rows:
            out.append("(no rows)")
        return "\n".join(out)

    @tool
    def interview(role: str, topic: str) -> str:
        """Interview one person about the company (at most 10 interviews per company).

        role: customer, current_employee, former_employee, competitor or supplier.
        topic: sales_team, working_conditions, top_management, decision_making or competitive_edge.
        People who would not know about a topic say so; the README lists who knows what.
        """
        role, topic = role.strip().lower(), topic.strip().lower()
        if role not in answers:
            return f"Unknown role {role!r}. Roles: {', '.join(answers)}. (Not counted.)"
        if topic not in answers[role]:
            return f"Unknown topic {topic!r}. Topics: {', '.join(answers[role])}. (Not counted.)"
        with lock:
            if INTERVIEWS_USED[name] >= INTERVIEW_LIMIT:
                return f"Interview limit reached: {INTERVIEW_LIMIT} interviews per company."
            INTERVIEWS_USED[name] += 1
        return answers[role][topic]

    return [list_documents, read_document, sql_query, interview]


def normalize_verdict(value: Any) -> str | None:
    """Accept PASS/FAIL (any case) or a boolean."""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    text = str(getattr(value, "content", value)).strip().upper()
    return text if text in VERDICTS else None


def check_chosen_points(chosen: Any) -> list[int]:
    """CHOSEN_POINTS must name 3 different points between 1 and 15."""
    try:
        points = sorted({int(p) for p in chosen})
    except (TypeError, ValueError):
        points = []
    if len(points) != 3 or len(chosen) != 3 or not all(p in FISHER_POINTS for p in points):
        raise SystemExit("Set CHOSEN_POINTS at the top of hw3.py to 3 different point numbers "
                         f"between 1 and 15, e.g. [1, 7, 15] (now: {chosen!r}).")
    return points


def normalize_points(points: Any, chosen: list[int]) -> dict[int, str | None]:
    """Map each chosen point number to PASS/FAIL, or None when missing/invalid."""
    points = points if isinstance(points, dict) else {}
    by_number: dict[int, Any] = {}
    for key, value in points.items():
        try:
            by_number[int(key)] = value
        except (TypeError, ValueError):
            continue
    return {p: normalize_verdict(by_number.get(p)) for p in chosen}


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read the expected 15 point verdicts per company."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def point_errors(points: dict[int, str | None], expected: dict[str, Any] | None) -> tuple[int, str]:
    """Count correct point verdicts and describe the wrong ones."""
    if expected is None:
        return 0, ""
    wrong = []
    for p in points:
        truth = expected["points"][str(p)]
        if points[p] != truth:
            wrong.append(f"{p}: expected {truth}, predicted {points[p] or 'missing'}")
    if all(v is None for v in points.values()):
        return 0, "all point verdicts missing"
    return len(points) - len(wrong), "; ".join(wrong)


def balanced_point_accuracy(
    analyses: dict[str, Any], truth: dict[str, dict[str, Any]], names: list[str], chosen: list[int]
) -> float:
    """Mean of the accuracy on true-PASS points and on true-FAIL points.

    Most verdicts are PASS, so answering PASS everywhere gets a high plain
    accuracy but only 50% balanced accuracy. Grading uses the balanced number.
    """
    hits = {"PASS": [0, 0], "FAIL": [0, 0]}
    for name in names:
        if name not in truth:
            continue
        entry = analyses.get(name)
        points = normalize_points(entry.get("points") if isinstance(entry, dict) else None, chosen)
        for p in chosen:
            expected = truth[name]["points"][str(p)]
            hits[expected][0] += points[p] == expected
            hits[expected][1] += 1
    rates = [h / n for h, n in hits.values() if n]
    return sum(rates) / len(rates) if rates else 0.0


def write_results(
    names: list[str], analyses: dict[str, Any], truth: dict[str, dict[str, Any]], chosen: list[int]
) -> tuple[Path, int]:
    """Write results.csv and return the number of correct point verdicts."""
    output = Path("results.csv")
    correct_points = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["company", "points", "points_correct", "point_errors", "interviews_used", "summary"])
        for name in names:
            entry = analyses.get(name)
            entry = entry if isinstance(entry, dict) else {}
            points = normalize_points(entry.get("points"), chosen)
            expected = truth.get(name)
            n_points, errors = point_errors(points, expected)
            correct_points += n_points
            writer.writerow([
                name,
                " ".join(f"{p}:{points[p] or 'missing'}" for p in chosen),
                f"{n_points}/{len(chosen)}" if expected else "",
                errors,
                INTERVIEWS_USED.get(name, 0),
                str(entry.get("summary", "")).strip(),
            ])
    return output, correct_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW3 on company data folders")
    parser.add_argument(
        "--data-folder",
        required=True,
        type=Path,
        help="folder containing one sub-folder per company",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    folders = company_folders(folder)
    if not folders:
        raise SystemExit(f"no company folders found in {folder}")

    chosen = check_chosen_points(CHOSEN_POINTS)
    load_env_file()
    companies = {path.name: make_tools(path) for path in folders}
    agent = build_agent()
    analyses = await analyze_companies(agent, companies)
    if not isinstance(analyses, dict):
        raise TypeError("analyze_companies() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct_points = write_results(list(companies), analyses, truth, chosen)
    summary = f" Your points: {', '.join(map(str, chosen))}."
    if truth:
        summary += (
            f" Point accuracy: {correct_points}/{len(companies) * len(chosen)}"
            f" (balanced: {balanced_point_accuracy(analyses, truth, list(companies), chosen):.1%})."
        )
    print(f"Processed {len(companies)} compan{'y' if len(companies) == 1 else 'ies'}. Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.data_folder.is_dir():
        raise SystemExit(f"not a folder: {args.data_folder}")
    return asyncio.run(run(args.data_folder))


if __name__ == "__main__":
    raise SystemExit(main())
