import glob
import json
import os
from collections import Counter, defaultdict

report_dir = "data/openvoicecs/reports/openrouter_batch_20260613"
json_files = glob.glob(os.path.join(report_dir, "*.json"))

scenario_failures = defaultdict(list)
for file_path in json_files:
    if "batch_summary_max10_native.json" in file_path or "smoke" in file_path:
        continue
    with open(file_path) as f:
        data = json.load(f)

    model_id = os.path.basename(file_path).replace("_max10_native.json", "")
    scenarios = data.get("failure_analysis", {}).get("scenarios", {})
    for sc_id, sc_info in scenarios.items():
        cats = sc_info.get("categories", [])
        if cats:
            scenario_failures[sc_id].extend(cats)

for sc_id, cats in sorted(scenario_failures.items()):
    counts = Counter(cats)
    print(f"Scenario: {sc_id}")
    for cat, cnt in counts.most_common():
        print(f"  - {cat:<32}: {cnt} times")
