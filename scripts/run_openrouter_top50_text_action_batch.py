#!/usr/bin/env python3
"""Run a resumable OpenRouter top-N text_to_action batch plus optional judging."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
RUNNER = ROOT / "scripts" / "run_openvoicecs.py"
DEFAULT_MODELS = ROOT / "openrouter-top-100-models.md"


def require_openrouter_credits(minimum_usd: float) -> None:
    """Refuse to start a sweep that cannot be paid for.

    The v0.1 sweep ran its OpenRouter balance to zero partway through and
    recorded 5,680 HTTP 402 trials as model scores of 0.0 across 26 models.
    Failing loudly here costs one API call; failing later costs the whole run.
    """
    if minimum_usd <= 0:
        return
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except Exception as exc:  # network, auth, or schema failure
        raise SystemExit(f"could not verify OpenRouter credits: {exc}") from exc
    data = payload.get("data") or {}
    remaining = float(data.get("total_credits", 0)) - float(data.get("total_usage", 0))
    if remaining < minimum_usd:
        raise SystemExit(
            f"OpenRouter balance ${remaining:.2f} is below the ${minimum_usd:.2f} floor "
            f"required to start this sweep; top up at "
            f"https://openrouter.ai/settings/credits or lower --min-credits-usd"
        )
    print(f"OpenRouter balance: ${remaining:.2f} (floor ${minimum_usd:.2f})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OpenRouter top-N text_to_action benchmark reports."
    )
    parser.add_argument("--models-file", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/openvoicecs/runs")
    parser.add_argument("--run-id", default="openrouter_top50_text_action_3trial")
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--adapter-mode",
        choices=["json-trace", "native-tools"],
        default="json-trace",
        help="json-trace uses the stepwise JSON action-loop fallback; native-tools uses provider tool calls.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--model-timeout-seconds",
        type=int,
        default=1800,
        help="Wall-clock timeout per raw model report; timed out models are summarized and skipped.",
    )
    parser.add_argument(
        "--judge-timeout-seconds",
        type=int,
        default=1200,
        help="Wall-clock timeout per judged model report; timed out judge jobs are summarized.",
    )
    parser.add_argument("--judge", action="append", default=[])
    parser.add_argument("--adjudicator", default=None)
    parser.add_argument("--skip-judging", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--min-credits-usd",
        type=float,
        default=5.0,
        help="refuse to start unless the OpenRouter balance is at least this; 0 disables",
    )
    args = parser.parse_args()

    if not args.skip_judging and len(args.judge) < 2:
        raise SystemExit(
            "audited model judging needs at least two --judge specs "
            "(for example --judge openrouter:openai/gpt-4o-mini "
            "--judge openrouter:google/gemini-2.5-flash), or pass --skip-judging"
        )

    require_openrouter_credits(args.min_credits_usd)

    models = parse_openrouter_models(args.models_file)[: args.top]
    if len(models) < args.top:
        raise SystemExit(f"only found {len(models)} models in {args.models_file}")

    run_root = args.output_root / args.run_id
    report_dir = run_root / "reports"
    judged_report_dir = run_root / "judged_reports"
    judging_dir = run_root / "judging"
    report_dir.mkdir(parents=True, exist_ok=True)
    judged_report_dir.mkdir(parents=True, exist_ok=True)
    judging_dir.mkdir(parents=True, exist_ok=True)

    raw_summary_path = run_root / "raw_summary.csv"
    raw_rows = [] if args.rerun else read_summary_csv(raw_summary_path)
    completed_raw_models = {
        str(row.get("model_id")) for row in raw_rows if row.get("model_id")
    }
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_raw_model, model, report_dir=report_dir, args=args): model
            for model in models
            if args.rerun or model not in completed_raw_models
        }
        for future in as_completed(futures):
            raw_rows.append(future.result())
            write_summary_csv(raw_rows, raw_summary_path)

    judge_rows = []
    if not args.skip_judging:
        if len(args.judge) < 2:
            raise SystemExit("judging requires at least two --judge specs")
        judged_summary_path = run_root / "judged_summary.csv"
        judge_rows = [] if args.rerun else read_summary_csv(judged_summary_path)
        judged_models = {
            str(row.get("model_id")) for row in judge_rows if row.get("model_id")
        }
        raw_success_models = {
            str(row.get("model_id"))
            for row in raw_rows
            if row.get("model_id") and not row.get("status")
        }
        judge_models = [
            model
            for model in models
            if model in raw_success_models
            and (args.rerun or model not in judged_models)
            if report_has_customer_messages(report_dir / f"{slugify(model)}.json")
        ]
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    run_judge_model,
                    model,
                    report_dir=report_dir,
                    judged_report_dir=judged_report_dir,
                    judging_dir=judging_dir,
                    args=args,
                ): model
                for model in judge_models
            }
            for future in as_completed(futures):
                row = future.result()
                if row:
                    judge_rows.append(row)
                    write_summary_csv(judge_rows, judged_summary_path)

    write_summary_csv(raw_rows, raw_summary_path)
    if judge_rows:
        write_summary_csv(judge_rows, run_root / "judged_summary.csv")
    print(f"Run artifacts: {run_root}")
    return 0


def run_raw_model(model: str, *, report_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    raw_path = report_dir / f"{slugify(model)}.json"
    if args.rerun or not raw_path.exists():
        command = [
            str(PYTHON),
            str(RUNNER),
            "score-provider",
            "--provider",
            "openrouter",
            "--model",
            model,
            "--name",
            f"openrouter_{slugify(model)}",
            "--track",
            "text_to_action",
            "--trials",
            str(args.trials),
            "--max-output-tokens",
            str(args.max_output_tokens),
            "--temperature",
            str(args.temperature),
            "--output",
            str(raw_path),
        ]
        command.append("--native-tools" if args.adapter_mode == "native-tools" else "--json-trace")
        try:
            run(command, cwd=ROOT, timeout=args.model_timeout_seconds)
        except subprocess.TimeoutExpired:
            return skipped_summary(
                model,
                raw_path,
                status="timeout",
                detail=f"raw model timeout after {args.model_timeout_seconds}s",
            )
        except subprocess.CalledProcessError as exc:
            return skipped_summary(
                model,
                raw_path,
                status="error",
                detail=f"raw model exited with code {exc.returncode}",
            )
    if not raw_path.exists():
        return skipped_summary(model, raw_path, status="missing", detail="raw report was not created")
    return report_summary(raw_path)


def run_judge_model(
    model: str,
    *,
    report_dir: Path,
    judged_report_dir: Path,
    judging_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    raw_path = report_dir / f"{slugify(model)}.json"
    slug = slugify(model)
    annotations_path = judging_dir / f"{slug}_annotations.jsonl"
    judge_report_path = judging_dir / f"{slug}_judge_report.json"
    judged_path = judged_report_dir / f"{slug}_judged.json"
    if args.rerun or not judged_path.exists():
        command = [
            str(PYTHON),
            str(RUNNER),
            "model-judge",
            str(raw_path),
        ]
        for judge in args.judge:
            command.extend(["--judge", judge])
        if args.adjudicator:
            command.extend(["--adjudicator", args.adjudicator])
        command.extend(
            [
                "--annotations-output",
                str(annotations_path),
                "--judge-report-output",
                str(judge_report_path),
                "--judged-report-output",
                str(judged_path),
            ]
        )
        try:
            run_capturing_stderr(command, cwd=ROOT, timeout=args.judge_timeout_seconds)
        except subprocess.TimeoutExpired:
            return skipped_summary(
                model,
                judged_path,
                status="judge_timeout",
                detail=f"judge timeout after {args.judge_timeout_seconds}s",
            )
        except subprocess.CalledProcessError as exc:
            reason = (exc.stderr or "").strip().splitlines()
            detail = f"judge exited with code {exc.returncode}"
            if reason:
                detail = f"{detail}: {reason[-1]}"
            return skipped_summary(
                model,
                judged_path,
                status="judge_error",
                detail=detail,
            )
    if judged_path.exists():
        return report_summary(judged_path, judge_report_path=judge_report_path)
    return None


def parse_openrouter_models(path: Path) -> list[str]:
    pattern = re.compile(r"^\|\s*\d+\s*\|\s*`([^`]+)`")
    models = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            models.append(match.group(1))
    return models


def run(command: list[str], *, cwd: Path, timeout: int | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, timeout=timeout)


def run_capturing_stderr(command: list[str], *, cwd: Path, timeout: int | None = None) -> None:
    """Run a command, re-raising failures with the child's stderr attached."""
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        timeout=timeout,
        stderr=subprocess.PIPE,
        text=True,
    )


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def report_has_customer_messages(path: Path) -> bool:
    if not path.exists():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    for result in report.get("results", []):
        for trial in result.get("trials", []):
            if trial.get("messages"):
                return True
    return False


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def report_summary(path: Path, *, judge_report_path: Path | None = None) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report.get("metric_scores", {})
    operational = report.get("operational_metrics", {})
    errors = 0
    trials = 0
    for result in report.get("results", []):
        for trial in result.get("trials", []):
            trials += 1
            if trial.get("error"):
                errors += 1
    row = {
        "model_id": report.get("model_metadata", {}).get("model_id"),
        "path": str(path),
        "overall_score": report.get("overall_score"),
        "task_success": metrics.get("task_success"),
        "tool_correctness": metrics.get("tool_correctness"),
        "safety": metrics.get("safety"),
        "pass_at_k": report.get("pass_at_k"),
        "pass_k": report.get("pass_k"),
        "mean_pass_rate": report.get("mean_pass_rate"),
        "conversation_experience_score": report.get("conversation_experience_score"),
        "median_latency_ms": operational.get("median_latency_ms"),
        "avg_tool_calls": operational.get("avg_tool_calls"),
        "num_trials": trials,
        "adapter_error_trials": errors,
    }
    if judge_report_path and judge_report_path.exists():
        judge = json.loads(judge_report_path.read_text(encoding="utf-8"))
        row["judge_agreement_alpha"] = judge.get("agreement", {}).get("overall")
        row["judge_annotations"] = judge.get("num_annotations")
        row["judge_raters"] = judge.get("num_raters")
    return row


def skipped_summary(model: str, path: Path, *, status: str, detail: str) -> dict[str, Any]:
    return {
        "model_id": model,
        "path": str(path),
        "status": status,
        "detail": detail,
    }


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            sort_score(row.get("task_success")),
            sort_score(row.get("pass_k")),
            sort_score(row.get("overall_score")),
        ),
        reverse=True,
    )
    fields = list(dict.fromkeys(field for row in sorted_rows for field in row))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted_rows)


def sort_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
