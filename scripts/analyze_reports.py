import os
import json
import glob

report_dir = "data/openvoicecs/reports/openrouter_batch_20260613"
json_files = glob.glob(os.path.join(report_dir, "*.json"))

rows = []
for file_path in json_files:
    if "batch_summary_max10_native.json" in file_path or "smoke" in file_path:
        continue
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    model_id = os.path.basename(file_path).replace("_max10_native", "")
    metrics = data.get("metric_scores", {}) or {}
    stability = data.get("stability_metrics", {}) or {}
    ops = data.get("operational_metrics", {}) or {}
    
    successes = data.get("confidence_intervals", {}).get("pass_at_k", {}).get("successes", 0)
    
    failures = data.get("failure_analysis", {}).get("categories", {}) or {}
    api_errors = failures.get("adapter_or_api_error", 0)
    tool_failures = failures.get("ignored_tool_failure", 0)
    
    row = {
        "model": model_id,
        "overall_score": data.get("overall_score", 0.0),
        "successes": successes,
        "task_success": metrics.get("task_success", 0.0),
        "factual_grounding": metrics.get("factual_grounding", 0.0),
        "sop_compliance": metrics.get("sop_compliance", 0.0),
        "privacy": metrics.get("privacy", 0.0),
        "auth_integrity": metrics.get("auth_integrity", 0.0),
        "tool_correctness": metrics.get("tool_correctness", 0.0),
        "safety": metrics.get("safety", 0.0),
        "api_errors": api_errors,
        "ignored_tool_failures": tool_failures,
        "avg_wasted_tool_calls": stability.get("avg_wasted_tool_calls", 0.0),
        "avg_latency_ms": ops.get("avg_latency_ms", 0.0)
    }
    rows.append(row)

rows.sort(key=lambda x: x["overall_score"], reverse=True)

# Print a nice ASCII table
headers = ["model", "overall", "pass/10", "task_succ", "factual", "sop", "privacy", "auth", "tool_corr", "safety", "api_err", "wasted_tc", "lat_ms"]
fmt = "{:<32} | {:>7} | {:>7} | {:>9} | {:>7} | {:>5} | {:>7} | {:>4} | {:>9} | {:>6} | {:>7} | {:>9} | {:>8}"

print(fmt.format(*headers))
print("-" * 140)
for r in rows:
    # formatting values
    print(fmt.format(
        r["model"][:32],
        f"{r['overall_score']:.2f}",
        f"{r['successes']}/10",
        f"{r['task_success']:.2f}",
        f"{r['factual_grounding']:.2f}",
        f"{r['sop_compliance']:.2f}",
        f"{r['privacy']:.2f}",
        f"{r['auth_integrity']:.2f}",
        f"{r['tool_correctness']:.2f}",
        f"{r['safety']:.2f}",
        f"{r['api_errors']}",
        f"{r['avg_wasted_tool_calls'] or 0.0:.1f}",
        f"{r['avg_latency_ms'] or 0.0:.0f}"
    ))
