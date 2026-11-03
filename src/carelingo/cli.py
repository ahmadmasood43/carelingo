"""Command-line entry points for running and scoring experiments."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from carelingo.dataset import load_cases
from carelingo.evaluation import percentile, score_output
from carelingo.extraction import extract_facts
from carelingo.facts import Fact
from carelingo.llm import Completion, MistralClient
from carelingo.translate import translate_dictation
from carelingo.verification import Discrepancy, compare_facts, needs_review, verify

DEFAULT_CASES_DIR = Path(__file__).parents[2] / "data" / "cases"
CLINICAL_ERRORS = {"unsupported_claim", "status_changed", "hedge_lost", "site_changed", "prn_lost"}

_FACTS = TypeAdapter(list[Fact])


def _usage(completion: Completion) -> dict:
    return {
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "reasoning_tokens": completion.reasoning_tokens,
        "latency_s": round(completion.latency_s, 3),
        "cost_usd": completion.cost_usd,
    }


def _read_jsonl(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for path in paths for line in path.read_text().splitlines()]


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()


def run_baseline(args: argparse.Namespace) -> None:
    cases = [case for case in load_cases(args.cases) if case.split == args.split]
    client = MistralClient()

    def records():
        for case in cases:
            for language in args.languages:
                completion = translate_dictation(client, args.model, case.sources[language])
                print(f"{case.id} {language} {completion.latency_s:.1f}s")
                yield {
                    "case_id": case.id,
                    "language": language,
                    "model": completion.model,
                    "output": completion.text,
                    **_usage(completion),
                }

    _write_jsonl(args.out, records())


def score_runs(args: argparse.Namespace) -> None:
    cases = {case.id: case for case in load_cases(args.cases)}
    groups = defaultdict(list)
    for path in args.runs:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            score = score_output(cases[record["case_id"]], record["language"], record["output"])
            groups[(record["model"], record["language"])].append((record, score))

    print(
        "| model | lang | n | numeric facts kept | drugs kept | resident kept "
        "| notes w/ unsupported number | latency p50 / p95 (s) | cost / note (USD) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for (model, language), rows in sorted(groups.items()):
        scores = [score for _, score in rows]
        latencies = [record["latency_s"] for record, _ in rows]
        numeric_total = sum(s.numeric_facts for s in scores)
        drug_total = sum(s.drugs for s in scores)
        print(
            f"| {model} | {language} | {len(rows)} "
            f"| {sum(s.numeric_facts_preserved for s in scores)}/{numeric_total} "
            f"| {sum(s.drugs_preserved for s in scores)}/{drug_total} "
            f"| {sum(s.resident_preserved for s in scores)}/{len(rows)} "
            f"| {sum(bool(s.unsupported_numbers) for s in scores)}/{len(rows)} "
            f"| {percentile(latencies, 50):.2f} / {percentile(latencies, 95):.2f} "
            f"| {sum(r['cost_usd'] for r, _ in rows) / len(rows):.6f} |"
        )

    if args.details:
        for (model, language), rows in sorted(groups.items()):
            for record, score in rows:
                missing = score.numeric_facts - score.numeric_facts_preserved
                if missing or score.unsupported_numbers or score.drugs_preserved < score.drugs:
                    print(
                        f"\n[{model} {language} {record['case_id']}] missing numeric facts: "
                        f"{missing}, unsupported: {list(score.unsupported_numbers)}\n"
                        f"{record['output']}"
                    )


def run_extraction(args: argparse.Namespace) -> None:
    """Extract facts from each source dictation and from the gold German note."""
    cases = [case for case in load_cases(args.cases) if case.split == args.split]
    client = MistralClient()

    def records():
        for case in cases:
            texts = {lang.value: text for lang, text in case.sources.items()}
            texts["de"] = case.german_reference
            for language, text in texts.items():
                extraction = extract_facts(client, args.model, text)
                print(f"{case.id} {language} {len(extraction.facts)} facts")
                yield {
                    "case_id": case.id,
                    "language": language,
                    "model": args.model,
                    "facts": [fact.model_dump(mode="json") for fact in extraction.facts],
                    "rejected": list(extraction.rejected),
                    **_usage(extraction.completion),
                }

    _write_jsonl(args.out, records())


def score_extraction(args: argparse.Namespace) -> None:
    cases = {case.id: case for case in load_cases(args.cases)}
    groups = defaultdict(list)
    for record in _read_jsonl(args.runs):
        gold = list(cases[record["case_id"]].facts)
        extracted = _FACTS.validate_python(record["facts"])
        groups[(record["model"], record["language"])].append(
            (record, gold, extracted, compare_facts(gold, extracted))
        )

    print("| extractor | text | gold facts | omitted | added | attribute errors | rejected |")
    print("|---|---|---|---|---|---|---|")
    for (model, language), rows in sorted(groups.items()):
        found = Counter(d.type for *_, ds in rows for d in ds)
        print(
            f"| {model} | {language} | {sum(len(gold) for _, gold, _, _ in rows)} "
            f"| {found['omitted']} | {found['added']} | {found['changed']} "
            f"| {sum(len(r['rejected']) for r, *_ in rows)} |"
        )
    if args.details:
        for (model, language), rows in sorted(groups.items()):
            for record, _, _, discrepancies in rows:
                for d in discrepancies:
                    print(f"[{model} {language} {record['case_id']}] {d.type} {d.detail}")


def run_verification(args: argparse.Namespace) -> None:
    """Extract facts from each baseline output and compare them with the source facts."""
    cases = {case.id: case for case in load_cases(args.cases)}
    source_facts = {
        (r["case_id"], r["language"]): _FACTS.validate_python(r["facts"])
        for r in _read_jsonl([args.extractions])
    }
    client = MistralClient()

    def records():
        for output in _read_jsonl(args.outputs):
            case, language = cases[output["case_id"]], output["language"]
            sources = source_facts[case.id, language]
            extraction = extract_facts(client, args.model, output["output"])
            discrepancies = verify(
                sources,
                list(extraction.facts),
                case.sources[language],
                language,
                output["output"],
            )
            print(f"{output['model']} {case.id} {language} {len(discrepancies)} discrepancies")
            yield {
                "case_id": case.id,
                "language": language,
                "generator": output["model"],
                "output": output["output"],
                "extractor": args.model,
                "source_facts": [fact.model_dump(mode="json") for fact in sources],
                "output_facts": [fact.model_dump(mode="json") for fact in extraction.facts],
                "discrepancies": [vars(d) for d in discrepancies],
                "needs_review": needs_review(discrepancies),
                **_usage(extraction.completion),
            }

    _write_jsonl(args.out, records())


def score_verification(args: argparse.Namespace) -> None:
    labels = {
        (a["model"], a["case"], a["language"]): set(a["errors"])
        for a in yaml.safe_load(args.annotations.read_text())
    }
    cases = {case.id: case for case in load_cases(args.cases)}
    rows = _read_jsonl(args.runs)
    # Re-run the comparison on the stored facts, so rule changes need no new API calls.
    for r in rows:
        case = cases[r["case_id"]]
        discrepancies = verify(
            _FACTS.validate_python(r["source_facts"]),
            _FACTS.validate_python(r["output_facts"]),
            case.sources[r["language"]],
            r["language"],
            r["output"],
        )
        r["discrepancies"] = [vars(d) for d in discrepancies]
        r["needs_review"] = needs_review(discrepancies)
    for name, relevant in [("clinical errors", CLINICAL_ERRORS), ("any labelled error", None)]:
        counts = Counter()
        for r in rows:
            errors = labels.get((r["generator"], r["case_id"], r["language"]), set())
            actual = bool(errors if relevant is None else errors & relevant)
            counts[(actual, r["needs_review"])] += 1
        tp, fp, fn = counts[True, True], counts[False, True], counts[True, False]
        tn = counts[False, False]
        print(
            f"{name}: flagged {tp + fp}/{len(rows)} | caught {tp}/{tp + fn} "
            f"| false alarms {fp}/{fp + tn}"
        )
    if args.details:
        for r in rows:
            errors = labels.get((r["generator"], r["case_id"], r["language"]), set())
            if bool(errors & CLINICAL_ERRORS) != r["needs_review"] or args.details == "all":
                serious = [
                    Discrepancy(**d).detail for d in r["discrepancies"] if d["severity"] != "minor"
                ]
                print(
                    f"\n[{r['generator']} {r['case_id']} {r['language']}] labelled "
                    f"{sorted(errors) or '-'}; flagged {r['needs_review']}: {serious}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(prog="carelingo")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_DIR)
    commands = parser.add_subparsers(required=True)

    baseline = commands.add_parser("baseline", help="translate dictations with one model")
    baseline.add_argument("--model", required=True)
    baseline.add_argument("--split", choices=["dev", "test"], default="dev")
    baseline.add_argument("--languages", nargs="+", default=["en", "pl"])
    baseline.add_argument("--out", type=Path, required=True)
    baseline.set_defaults(handler=run_baseline)

    score = commands.add_parser("score", help="score run files against gold facts")
    score.add_argument("runs", type=Path, nargs="+")
    score.add_argument("--details", action="store_true", help="print outputs with problems")
    score.set_defaults(handler=score_runs)

    extract = commands.add_parser("extract", help="extract facts from sources and gold notes")
    extract.add_argument("--model", required=True)
    extract.add_argument("--split", choices=["dev", "test"], default="dev")
    extract.add_argument("--out", type=Path, required=True)
    extract.set_defaults(handler=run_extraction)

    score_extract = commands.add_parser("score-extraction", help="compare extractions to gold")
    score_extract.add_argument("runs", type=Path, nargs="+")
    score_extract.add_argument("--details", action="store_true")
    score_extract.set_defaults(handler=score_extraction)

    check = commands.add_parser("verify", help="verify baseline outputs against source facts")
    check.add_argument("--model", required=True, help="extractor model")
    check.add_argument("--extractions", type=Path, required=True)
    check.add_argument("--outputs", type=Path, nargs="+", required=True)
    check.add_argument("--out", type=Path, required=True)
    check.set_defaults(handler=run_verification)

    score_check = commands.add_parser("score-verification", help="compare flags with labels")
    score_check.add_argument("runs", type=Path, nargs="+")
    score_check.add_argument("--annotations", type=Path, required=True)
    score_check.add_argument(
        "--details", nargs="?", const="mismatches", choices=["mismatches", "all"]
    )
    score_check.set_defaults(handler=score_verification)

    args = parser.parse_args()
    args.handler(args)
