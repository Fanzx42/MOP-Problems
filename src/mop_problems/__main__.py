"""Small CLI for inspecting the catalogue and checking installation."""

from __future__ import annotations

import argparse
import json

from mop_problems import get_problem, get_problem_info, list_problems, list_suites


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mop-problems")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list implemented problems")
    listing.add_argument("--suite")
    listing.add_argument("--category")
    listing.add_argument("--json", action="store_true")
    suites = commands.add_parser("suites", help="list all suites, including planned entries")
    suites.add_argument("--status", choices=["implemented", "planned"])
    suites.add_argument("--json", action="store_true")
    info = commands.add_parser("info", help="show problem metadata")
    info.add_argument("name")
    sample = commands.add_parser("sample", help="evaluate a reproducible decision-box sample")
    sample.add_argument("name")
    sample.add_argument("--n", type=int, default=3)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--n-var", type=int)
    sample.add_argument("--n-obj", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            names = list_problems(suite=args.suite, category=args.category)
            print(json.dumps(names) if args.json else "\n".join(names))
        elif args.command == "suites":
            rows = list_suites(status=args.status)
            if args.json:
                print(json.dumps(rows, indent=2, ensure_ascii=False))
            else:
                for row in rows:
                    print(f"{row['name']:20} {row['status']:12} {row['category']}")
        elif args.command == "info":
            print(json.dumps(get_problem_info(args.name), indent=2, ensure_ascii=False))
        else:
            kwargs = {
                k: v for k, v in {"n_var": args.n_var, "n_obj": args.n_obj}.items() if v is not None
            }
            problem = get_problem(args.name, **kwargs)
            X = problem.sample(args.n, seed=args.seed)
            result = problem.evaluate(X)
            print(
                json.dumps(
                    {
                        "problem": problem.name,
                        "X": X.tolist(),
                        "F": result.F.tolist(),
                        "G": result.G.tolist(),
                        "H": result.H.tolist(),
                        "CV": result.CV.tolist(),
                        "feasible": result.feasible.tolist(),
                    },
                    indent=2,
                    allow_nan=False,
                )
            )
    except (KeyError, ValueError, TypeError, NotImplementedError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
