import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from eval_cases import EVAL_CASES


class EvalResult:
    def __init__(self, case: dict, passed: bool, actual: str,
                 reason: str, duration_s: float, path: list):
        self.case       = case
        self.passed     = passed
        self.actual     = actual
        self.reason     = reason
        self.duration_s = duration_s
        self.path       = path


def check_exact(actual: str, expected) -> tuple[bool, str]:
    expected_str = str(expected).replace(",", "")
    actual_clean = actual.replace(",", "").replace(".", "")
    expected_clean = expected_str.replace(".", "")

    if expected_clean in actual_clean:
        return True, f"found '{expected}' in answer"
    try:
        num = int(expected)
        for fmt in [f"{num:,}", f"{num}", str(num)]:
            if fmt.replace(",","") in actual.replace(",",""):
                return True, f"found '{fmt}' in answer"
    except (ValueError, TypeError):
        pass
    return False, f"expected '{expected}' not found in answer"


def check_contains(actual: str, expected: str) -> tuple[bool, str]:
    if expected.lower() in actual.lower():
        return True, f"found '{expected}' in answer"
    return False, f"'{expected}' not found in answer"


def check_numeric(actual: str, expected: float, tolerance: float) -> tuple[bool, str]:
    numbers = re.findall(r'-?\d+\.?\d*', actual.replace(",", ""))
    for num_str in numbers:
        try:
            num = float(num_str)
            if abs(num - expected) <= tolerance:
                return True, f"found {num} which is within {tolerance} of {expected}"
        except ValueError:
            continue
    return False, f"no number within {tolerance} of {expected} found (expected ~{expected})"


def evaluate_response(case: dict, response: str) -> tuple[bool, str]:
    check_type = case["check_type"]
    expected   = case["expected"]

    if check_type == "exact":
        return check_exact(response, expected)
    elif check_type == "contains":
        return check_contains(response, str(expected))
    elif check_type == "numeric":
        tolerance = case.get("tolerance", 0.1)
        return check_numeric(response, float(expected), tolerance)
    else:
        return False, f"unknown check_type: {check_type}"


def run_case(case: dict, graph, schema: dict, csv_path: str) -> EvalResult:
    from agent.state import create_initial_state

    state = create_initial_state(
        user_query=case["question"],
        schema_profile=schema,
        csv_file_path=csv_path,
    )

    start = time.time()
    try:
        result    = graph.invoke(state)
        duration  = round(time.time() - start, 1)
        response  = result.get("final_response") or ""
        path      = result.get("execution_path", [])
        decision  = result.get("router_decision", "unknown")

        if decision in ("red", "yellow") and not response:
            response = result.get("refusal_reason") or \
                       result.get("clarification_question") or \
                       f"Pipeline decision: {decision}"

        passed, reason = evaluate_response(case, response)

    except Exception as e:
        duration = round(time.time() - start, 1)
        response = f"EXCEPTION: {str(e)}"
        path     = []
        passed   = False
        reason   = f"pipeline raised exception: {str(e)}"

    return EvalResult(
        case=case,
        passed=passed,
        actual=response[:300],
        reason=reason,
        duration_s=duration,
        path=path,
    )


def print_report(results: list[EvalResult], run_date: str):
    total  = len(results)
    passed = sum(1 for r in results if r.passed)

    by_diff = {}
    for r in results:
        d = r.case["difficulty"]
        by_diff.setdefault(d, []).append(r)

    print()
    print("=" * 65)
    print("EVALUATION REPORT — Personal Data Analyst")
    print("=" * 65)
    print(f"Run date:  {run_date}")
    print(f"Dataset:   olist_master.csv (99,441 rows)")
    print()

    print("ACCURACY BY DIFFICULTY:")
    for diff in ["simple", "medium", "complex"]:
        group = by_diff.get(diff, [])
        if not group:
            continue
        n_pass = sum(1 for r in group if r.passed)
        pct    = round(n_pass / len(group) * 100)
        bar    = "#" * n_pass + "." * (len(group) - n_pass)
        print(f"  {diff.capitalize():<8} [{bar}]  {n_pass}/{len(group)}  {pct}%")

    pct_total = round(passed / total * 100)
    print()
    print(f"OVERALL: {passed}/{total}  ({pct_total}%)")
    print()

    durations = [r.duration_s for r in results]
    print("TIMING:")
    print(f"  Average: {round(sum(durations)/len(durations), 1)}s per query")
    print(f"  Slowest: {max(durations)}s")
    print(f"  Fastest: {min(durations)}s")
    print()

    print("DETAILED RESULTS:")
    print("-" * 65)
    for r in results:
        status = "[PASS]" if r.passed else "[FAIL]"
        print(f"{status}  [{r.case['id']}] {r.case['question'][:55]}")
        if not r.passed:
            print(f"         Expected: {r.case['expected']}")
            print(f"         Got:      {r.actual[:120]}")
            print(f"         Reason:   {r.reason}")
        else:
            print(f"         {r.reason}  ({r.duration_s}s)")

    print("=" * 65)
    print()

    report = {
        "run_date":    run_date,
        "total":       total,
        "passed":      passed,
        "accuracy_pct": pct_total,
        "by_difficulty": {
            d: {
                "passed": sum(1 for r in g if r.passed),
                "total":  len(g),
                "pct":    round(sum(1 for r in g if r.passed) / len(g) * 100),
            }
            for d, g in by_diff.items()
        },
        "results": [
            {
                "id":       r.case["id"],
                "question": r.case["question"],
                "passed":   r.passed,
                "reason":   r.reason,
                "duration": r.duration_s,
                "path":     r.path,
            }
            for r in results
        ],
    }
    report_path = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Personal Data Analyst pipeline")
    parser.add_argument("--difficulty", choices=["simple", "medium", "complex"],
                        help="Run only cases of this difficulty")
    parser.add_argument("--id", type=str, help="Run a single case by ID (e.g. S01)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print cases without running the pipeline")
    args = parser.parse_args()

    cases = EVAL_CASES
    if args.difficulty:
        cases = [c for c in cases if c["difficulty"] == args.difficulty]
    if args.id:
        cases = [c for c in cases if c["id"] == args.id]

    if args.dry_run:
        print(f"\nDRY RUN — {len(cases)} cases selected:\n")
        for c in cases:
            print(f"  [{c['id']}] ({c['difficulty']}) {c['question']}")
            print(f"       Expected: {c['expected']}  Check: {c['check_type']}")
        return

    print(f"\nRunning {len(cases)} eval cases...")
    print("This will make real API calls. Estimated time: "
          f"{len(cases) * 18}–{len(cases) * 30}s\n")

    from agent.graph_builder import compile_graph
    import json as _json
    from pathlib import Path as _Path

    graph = compile_graph(checkpointer=False)

    CSV_PATH    = str(_Path(__file__).parent / "data" / "olist_master.csv")
    schema_path = _Path(__file__).parent / "data" / "schema_profile.json"

    if not schema_path.exists():
        print("ERROR: data/schema_profile.json not found.")
        print("Run: python scanner.py olist_master.csv --output data/schema_profile.json")
        return

    with open(schema_path) as f:
        schema = _json.load(f)

    results = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']}: {case['question'][:55]}...")
        result = run_case(case, graph, schema, CSV_PATH)
        status = "PASS" if result.passed else "FAIL"
        print(f"          [{status}]  {result.duration_s}s")
        results.append(result)

    print_report(results, datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
