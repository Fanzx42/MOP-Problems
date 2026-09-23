"""Run after installing the checkout: python examples/basic_usage.py."""

from mop_problems import get_problem, list_problems


def main():
    print("Implemented benchmarks:", ", ".join(list_problems()))
    for name in ["ZDT1", "DTLZ2", "BNH", "OSY"]:
        p = get_problem(name)
        out = p.evaluate(p.sample(4, seed=42))
        print(f"\n{p}\nF =\n{out.F}\nCV = {out.CV}")
    p = get_problem("BNH")
    print("\nKnown feasible BNH point:", p.evaluate([1, 1]).feasible[0])


if __name__ == "__main__":
    main()
