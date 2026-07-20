"""CLI: python -m app.evals <subcommand>. Subcommands grow over M4/M5."""
import argparse
import json
import sys

from app.evals.derivations import write_golden
from app.evals.golden import GOLDEN_DIR, load_golden


def _cmd_run(args) -> int:
    from app.agents.llm import GraphDeps
    from app.deps import store
    from app.evals.harness import run_eval
    from app.evals.judge import real_judge

    golden = load_golden(GOLDEN_DIR)
    if args.datasets:
        keep = set(args.datasets.split(","))
        golden = [g for g in golden if g.name in keep]
    st = store()
    eval_id = run_eval(
        st, golden, GraphDeps.default, judge=real_judge() if args.judge else None,
        label=args.label,
    )
    run = next(r for r in st.list_eval_runs() if r["id"] == eval_id)
    rate = run["tier1_passed"] / max(run["tier1_scorable"], 1)
    print(
        f"eval {eval_id}: tier1 {run['tier1_passed']}/{run['tier1_scorable']} "
        f"({rate:.0%}), judge_avg={run['judge_avg']}, cost=${run['cost_usd']:.3f}"
    )
    if args.gate:
        baseline = json.loads((GOLDEN_DIR.parent / "baseline.json").read_text())
        floor = baseline["tier1_pass_rate"] - args.gate_margin
        if rate < floor:
            print(f"GATE FAIL: pass rate {rate:.2%} < floor {floor:.2%}")
            return 1
        print(f"gate ok (floor {floor:.2%})")
    if args.write_baseline:
        (GOLDEN_DIR.parent / "baseline.json").write_text(
            json.dumps({"tier1_pass_rate": round(rate, 4), "eval_run_id": eval_id}, indent=2)
        )
        print("baseline.json updated")
    return 0


def _cmd_study(args) -> None:
    from app.agents.llm import GraphDeps, _structured_fn
    from app.agents.schemas import JudgeTurn
    from app.deps import store
    from app.evals.study import CONFIGS_DIR, load_study_config, run_study

    configs = [
        load_study_config(CONFIGS_DIR / f"{n.strip()}.yaml")
        for n in args.configs.split(",")
    ]
    run_study(
        store(), configs,
        deps_factory_for=lambda cfg: (lambda: GraphDeps.default(models=cfg.models)),
        judge_for=lambda cfg: None if args.no_judge
        else _structured_fn("judge", JudgeTurn, model=cfg.judge_model),
        golden_sets=load_golden(GOLDEN_DIR),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.evals")
    sub = parser.add_subparsers(dest="cmd", required=True)

    golden = sub.add_parser("golden", help="golden-set maintenance")
    golden.add_argument("--write", action="store_true",
                        help="recompute expected values from data/samples")

    run = sub.add_parser("run", help="run the golden set live (spends API budget)")
    run.add_argument("--label", default="")
    run.add_argument("--judge", action="store_true", help="also run the tier-2 judge")
    run.add_argument("--datasets", default="", help="comma list, e.g. taxi,weather")
    run.add_argument("--gate", action="store_true",
                     help="exit 1 if pass rate drops below baseline - margin")
    run.add_argument("--gate-margin", type=float, default=0.05)
    run.add_argument("--write-baseline", action="store_true")

    tmpl = sub.add_parser("label-template",
                          help="print a YAML labeling template for an eval run")
    tmpl.add_argument("eval_run_id")

    sub.add_parser("calibration", help="print the calibration report as JSON")

    study = sub.add_parser("study", help="run the golden set across model configs")
    study.add_argument("--configs", default="mini,strong-planner,strong-critic,strong")
    study.add_argument("--no-judge", action="store_true")

    args = parser.parse_args()
    if args.cmd == "golden":
        if args.write:
            write_golden()
            print("golden YAMLs updated from derivations")
        return
    if args.cmd == "run":
        sys.exit(_cmd_run(args))
    if args.cmd == "label-template":
        from app.deps import store
        from app.evals.calibration import label_template
        print(label_template(store(), args.eval_run_id))
        return
    if args.cmd == "calibration":
        from app.deps import store
        from app.evals.calibration import calibration_report, load_labels
        labels = load_labels()
        if labels is None:
            print("no labels file at backend/app/evals/labels/human_labels.yaml")
            sys.exit(1)
        print(json.dumps(calibration_report(store(), labels), indent=2))
        return
    if args.cmd == "study":
        _cmd_study(args)
        return


if __name__ == "__main__":
    main()
