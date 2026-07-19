"""CLI: python -m app.evals <subcommand>. Subcommands grow over M4/M5."""
import argparse

from app.evals.derivations import write_golden


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.evals")
    sub = parser.add_subparsers(dest="cmd", required=True)
    golden = sub.add_parser("golden", help="golden-set maintenance")
    golden.add_argument("--write", action="store_true",
                        help="recompute expected values from data/samples")
    args = parser.parse_args()
    if args.cmd == "golden" and args.write:
        write_golden()
        print("golden YAMLs updated from derivations")


if __name__ == "__main__":
    main()
