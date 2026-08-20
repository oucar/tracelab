"""Study + calibration artifacts for the README: markdown tables and the tradeoff chart."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.tracing.store import Store


def study_rows(st: Store) -> list[dict]:
    latest: dict[str, dict] = {}
    for r in st.list_eval_runs():  # newest first
        if not r["label"].startswith("study:"):
            continue
        latest.setdefault(r["label"].removeprefix("study:"), r)
    out = []
    for config, r in sorted(latest.items()):
        n = max(r["questions_total"], 1)
        out.append({
            "config": config, "eval_run_id": r["id"],
            "tier1_pass_rate": (r["tier1_passed"] / r["tier1_scorable"]
                                if r["tier1_scorable"] else None),
            "judge_avg": r["judge_avg"],
            "cost_per_question": round(r["cost_usd"] / n, 4),
            "latency_s_per_question": round(r["duration_ms"] / n / 1000, 1),
            "cost_usd": r["cost_usd"],
        })
    return out


def study_markdown(rows: list[dict]) -> str:
    lines = ["| Config | Tier-1 pass | Judge avg (1-5) | $/question | s/question |",
             "|---|---|---|---|---|"]
    for r in rows:
        t1 = "—" if r["tier1_pass_rate"] is None else f"{r['tier1_pass_rate']:.0%}"
        judge = "—" if r["judge_avg"] is None else f"{r['judge_avg']:.2f}"
        lines.append(f"| {r['config']} | {t1} | {judge} "
                     f"| ${r['cost_per_question']:.4f} | {r['latency_s_per_question']} |")
    return "\n".join(lines)


def calibration_markdown(report: dict) -> str:
    if not report.get("available"):
        return "_No calibration labels yet._"
    lines = ["| Dimension | n | Exact % | Within-1 % | Cohen's κ |", "|---|---|---|---|---|"]
    for d in report["dimensions"]:
        lines.append(f"| {d['dimension'].replace('_', ' ')} | {d['n']} "
                     f"| {d['exact_pct']} | {d['within1_pct']} | {d['kappa']} |")
    o = report["overall"]
    lines.append(f"| **overall (pooled)** | {o['n']} | {o['exact_pct']} "
                 f"| {o['within1_pct']} | {o['kappa']} |")
    return "\n".join(lines)


def tradeoff_png(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    xs = [r["cost_per_question"] for r in rows]
    ys = [r["judge_avg"] if r["judge_avg"] is not None
          else (r["tier1_pass_rate"] or 0) * 5 for r in rows]
    sizes = [40 + 12 * r["latency_s_per_question"] for r in rows]
    ax.scatter(xs, ys, s=sizes, alpha=0.75)
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(r["config"], (x, y), xytext=(6, 6), textcoords="offset points")
    if max(xs) / max(min(xs), 1e-9) > 10:
        ax.set_xscale("log")
    ax.set_xlabel("cost per question (USD)")
    ax.set_ylabel("quality (judge avg, 1-5)")
    ax.set_title("Quality vs cost vs latency (point size = latency)")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
