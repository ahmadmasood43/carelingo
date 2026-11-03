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
from carelingo.injection import variants
from carelingo.llm import Completion, MistralClient, OllamaClient
from carelingo.narrative import check_narrative
from carelingo.pipeline import ModelChoice, PipelineConfig, process
from carelingo.simulation import Step, run_policy, summarize
from carelingo.translate import translate_dictation
from carelingo.verification import (
    UNVERIFIABLE,
    compare_facts,
    needs_review,
    verify,
)

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
    client = MistralClient(reasoning_effort=args.reasoning_effort)

    def records():
        for case in cases:
            for language in args.languages:
                completion = translate_dictation(client, args.model, case.sources[language])
                print(f"{case.id} {language} {completion.latency_s:.1f}s")
                yield {
                    "case_id": case.id,
                    "language": language,
                    # Keep runs at different effort levels apart in later scoring.
                    "model": completion.model
                    + (f"@{args.reasoning_effort}" if args.reasoning_effort else ""),
                    "reasoning_effort": args.reasoning_effort,
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


def run_injection(args: argparse.Namespace) -> None:
    """Write single-error variants of the gold notes, one record per source language."""
    cases = [case for case in load_cases(args.cases) if case.split == args.split]
    records = [
        {
            "case_id": case.id,
            "language": language.value,
            "model": f"injected:{variant.error}",
            "error": variant.error,
            "description": variant.description,
            "output": variant.text,
        }
        for case in cases
        for variant in variants(case)
        for language in case.sources
    ]
    _write_jsonl(args.out, records)
    print(f"{len(records)} records from {len(cases)} cases")


def score_injection(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.runs)
    cases = {case.id: case for case in load_cases(args.cases)}
    by_error = defaultdict(list)
    for r in rows:
        case = cases[r["case_id"]]
        discrepancies = verify(
            _FACTS.validate_python(r["source_facts"]),
            _FACTS.validate_python(r["output_facts"]),
            case.sources[r["language"]],
            r["language"],
            r["output"],
        )
        if r.get("extraction_failed"):
            discrepancies.insert(0, UNVERIFIABLE)
        r["discrepancies"] = discrepancies
        flagged = needs_review(discrepancies)
        if args.narrative:
            flagged = flagged or bool(r.get("narrative_findings"))
        by_error[r["generator"].removeprefix("injected:")].append((r, flagged))

    print("| injected error | n | flagged | with a critical discrepancy |")
    print("|---|---|---|---|")
    for error, items in sorted(by_error.items(), key=lambda item: item[0] == "none"):
        n = len(items)
        flagged = sum(flag for _, flag in items)
        critical = sum(any(d.severity == "critical" for d in r["discrepancies"]) for r, _ in items)
        label = "none (controls: false alarms)" if error == "none" else error
        print(f"| {label} | {n} | {flagged}/{n} ({flagged / n:.0%}) | {critical}/{n} |")
    if args.details:
        for error, items in sorted(by_error.items()):
            for r, flag in items:
                if flag == (error == "none"):
                    serious = [d.detail for d in r["discrepancies"] if d.severity != "minor"]
                    print(f"[{error} {r['case_id']} {r['language']}] flagged {flag}: {serious}")


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
    # The same German note is often checked against several source languages.
    extractions = {}

    def records():
        for output in _read_jsonl(args.outputs):
            case, language = cases[output["case_id"]], output["language"]
            sources = source_facts[case.id, language]
            if output["output"] not in extractions:
                extractions[output["output"]] = extract_facts(client, args.model, output["output"])
            extraction = extractions[output["output"]]
            discrepancies = verify(
                sources,
                list(extraction.facts),
                case.sources[language],
                language,
                output["output"],
            )
            if extraction.failed:
                discrepancies.insert(0, UNVERIFIABLE)
            print(f"{output['model']} {case.id} {language} {len(discrepancies)} discrepancies")
            yield {
                "case_id": case.id,
                "language": language,
                "generator": output["model"],
                "output": output["output"],
                "extractor": args.model,
                "source_facts": [fact.model_dump(mode="json") for fact in sources],
                "output_facts": [fact.model_dump(mode="json") for fact in extraction.facts],
                "extraction_failed": extraction.failed,
                "discrepancies": [d.as_record() for d in discrepancies],
                "needs_review": needs_review(discrepancies),
                **_usage(extraction.completion),
            }

    _write_jsonl(args.out, records())


def run_narrative(args: argparse.Namespace) -> None:
    """Add a narrative check to stored verification records."""
    cases = {case.id: case for case in load_cases(args.cases)}
    client = OllamaClient(think=args.think)
    checks = {}
    done = []
    if args.resume and args.out.exists():
        done = _read_jsonl([args.out])
    finished = {(r["generator"], r["case_id"], r["language"]) for r in done}

    def records():
        yield from done
        for r in _read_jsonl(args.inputs):
            if (r["generator"], r["case_id"], r["language"]) in finished:
                continue
            dictation = cases[r["case_id"]].sources[r["language"]]
            key = (dictation, r["output"])
            if key not in checks:
                checks[key] = check_narrative(client, args.model, dictation, r["output"])
            check = checks[key]
            print(f"{r['generator']} {r['case_id']} {r['language']} {len(check.unsupported)}")
            yield {
                **r,
                "narrative_model": args.model,
                "narrative_think": args.think,
                "narrative_findings": list(check.unsupported),
                "narrative_latency_s": round(check.completion.latency_s, 3),
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
        if r.get("extraction_failed"):
            discrepancies.insert(0, UNVERIFIABLE)
        r["discrepancies"] = [d.as_record() for d in discrepancies]
        r["needs_review"] = needs_review(discrepancies) or (
            args.narrative and bool(r.get("narrative_findings"))
        )
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
                serious = [d["detail"] for d in r["discrepancies"] if d["severity"] != "minor"]
                print(
                    f"\n[{r['generator']} {r['case_id']} {r['language']}] labelled "
                    f"{sorted(errors) or '-'}; flagged {r['needs_review']}: {serious}"
                )


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full pipeline live on a split and record what happened to each note."""
    cases = [case for case in load_cases(args.cases) if case.split == args.split]
    config = PipelineConfig(
        draft=ModelChoice(MistralClient(reasoning_effort=args.draft_effort), args.draft_model),
        extractor=ModelChoice(MistralClient(), args.extractor),
        strong=ModelChoice(MistralClient(reasoning_effort=args.strong_effort), args.strong_model)
        if args.strong_model
        else None,
    )

    def records():
        for case in cases:
            for language in args.languages:
                result = process(case.sources[language], language, config)
                print(f"{case.id} {language} {result.decision.tier} {result.latency_s:.1f}s")
                yield {
                    "case_id": case.id,
                    "language": language,
                    "draft_model": args.draft_model,
                    "draft_effort": args.draft_effort,
                    "strong_model": args.strong_model,
                    "strong_effort": args.strong_effort,
                    "output": result.text,
                    "tier": result.decision.tier,
                    "escalated": result.escalated,
                    "discrepancies": [d.as_record() for d in result.discrepancies],
                    "checklist": result.checklist,
                    "cost_usd": result.cost_usd,
                    "latency_s": round(result.latency_s, 3),
                }

    _write_jsonl(args.out, records())


def simulate_routing(args: argparse.Namespace) -> None:
    cases = {case.id: case for case in load_cases(args.cases)}
    labels = defaultdict(set)
    for path in args.annotations:
        for a in yaml.safe_load(path.read_text()):
            labels[a["model"], a["case"], a["language"]] |= set(a["errors"])
    sources = {(r["case_id"], r["language"]): r for r in _read_jsonl([args.extractions])}
    drafts = {(r["model"], r["case_id"], r["language"]): r for r in _read_jsonl(args.baselines)}
    steps = {}
    for v in _read_jsonl(args.verifications):
        key = (v["generator"], v["case_id"], v["language"])
        case = cases[v["case_id"]]
        discrepancies = verify(
            _FACTS.validate_python(v["source_facts"]),
            _FACTS.validate_python(v["output_facts"]),
            case.sources[v["language"]],
            v["language"],
            v["output"],
        )
        if v.get("extraction_failed"):
            discrepancies.insert(0, UNVERIFIABLE)
        steps[key] = Step(
            drafts[key]["cost_usd"],
            drafts[key]["latency_s"],
            v["cost_usd"],
            v["latency_s"],
            discrepancies,
            frozenset(labels.get(key, set())),
        )
    if args.narrative:
        findings = {
            (r["generator"], r["case_id"], r["language"]): tuple(r["narrative_findings"])
            for r in _read_jsonl(args.narrative)
        }
        steps = {
            key: Step(**{**vars(step), "narrative": findings.get(key, ())})
            for key, step in steps.items()
        }
    dictations = sorted({(case_id, lang) for _, case_id, lang in steps})

    configs = []
    for first in args.first:
        configs += [(first, None, "unverified"), (first, None, "verified")]
        configs += [
            (first, strong, p)
            for strong in args.strong
            for p in ("escalate_critical", "escalate_any")
        ]
    configs += [(strong, None, "unverified") for strong in args.strong]
    configs += [(strong, None, "verified") for strong in args.strong]

    print(
        "| first draft | escalation | policy | escalated | notes w/ clinical error "
        "| **unflagged errors** | needs attention | blocked | cost/note | latency p50 / max |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for first, strong, policy in configs:
        outcomes = [
            run_policy(
                policy,
                steps[first, case_id, lang],
                steps[strong, case_id, lang] if strong else None,
                sources[case_id, lang]["cost_usd"],
                sources[case_id, lang]["latency_s"],
            )
            for case_id, lang in dictations
        ]
        m = summarize(outcomes)
        n = m["n"]
        print(
            f"| {first} | {strong or '-'} | {policy} | {m['escalated']}/{n} "
            f"| {m['clinical_errors']}/{n} | **{m['unflagged_errors']}/{n}** "
            f"| {m['needs_attention']}/{n} | {m['blocked']}/{n} "
            f"| ${m['cost_per_note_usd']:.5f} "
            f"| {m['latency_p50_s']:.1f} s / {m['latency_max_s']:.1f} s |"
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
    baseline.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
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
    score_check.add_argument("--narrative", action="store_true", help="include narrative check")
    score_check.set_defaults(handler=score_verification)

    narrative = commands.add_parser("narrative", help="add a narrative check to records")
    narrative.add_argument("--model", default="qwen3:8b")
    narrative.add_argument("--think", action="store_true", help="enable the model's reasoning")
    narrative.add_argument("--resume", action="store_true", help="keep records already in --out")
    narrative.add_argument("--inputs", type=Path, nargs="+", required=True)
    narrative.add_argument("--out", type=Path, required=True)
    narrative.set_defaults(handler=run_narrative)

    inject = commands.add_parser("inject", help="write single-error variants of gold notes")
    inject.add_argument("--split", choices=["dev", "test"], default="dev")
    inject.add_argument("--out", type=Path, required=True)
    inject.set_defaults(handler=run_injection)

    score_inject = commands.add_parser("score-injection", help="detection rate per error type")
    score_inject.add_argument("runs", type=Path, nargs="+")
    score_inject.add_argument("--narrative", action="store_true", help="include narrative check")
    score_inject.add_argument("--details", action="store_true", help="print misses and alarms")
    score_inject.set_defaults(handler=score_injection)

    live = commands.add_parser("pipeline", help="run the full pipeline live")
    # Default: a strong first draft, verified, with no second model.
    live.add_argument("--draft-model", default="mistral-large-4")
    live.add_argument("--draft-effort", choices=["low", "medium", "high"], default="low")
    live.add_argument("--strong-model")
    live.add_argument("--strong-effort", choices=["low", "medium", "high"])
    live.add_argument("--extractor", default="ministral-14b-2512")
    live.add_argument("--split", choices=["dev", "test"], default="dev")
    live.add_argument("--languages", nargs="+", default=["en", "pl"])
    live.add_argument("--out", type=Path, required=True)
    live.set_defaults(handler=run_pipeline)

    simulate = commands.add_parser("simulate-routing", help="replay runs under routing policies")
    simulate.add_argument("--baselines", type=Path, nargs="+", required=True)
    simulate.add_argument("--verifications", type=Path, nargs="+", required=True)
    simulate.add_argument("--extractions", type=Path, required=True)
    simulate.add_argument("--annotations", type=Path, nargs="+", required=True)
    simulate.add_argument("--narrative", type=Path, nargs="+", help="narrative check records")
    simulate.add_argument("--first", nargs="+", default=["ministral-8b-2512", "ministral-14b-2512"])
    simulate.add_argument("--strong", nargs="+", default=["mistral-large-4", "mistral-large-4@low"])
    simulate.set_defaults(handler=simulate_routing)

    args = parser.parse_args()
    args.handler(args)
