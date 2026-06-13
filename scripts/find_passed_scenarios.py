import os
import json
import glob

report_dir = "data/openvoicecs/reports/openrouter_batch_20260613"
json_files = glob.glob(os.path.join(report_dir, "*.json"))

model_passes = {}
for file_path in json_files:
    if "batch_summary_max10_native.json" in file_path or "smoke" in file_path:
        continue
    with open(file_path, "r") as f:
        data = json.load(f)
    
    model_id = os.path.basename(file_path).replace("_max10_native.json", "")
    passed_scenarios = []
    
    # Try to find trials that passed
    for result in data.get("results", []):
        if result.get("pass_at_k"):
            passed_scenarios.append(result["id"])
            
    # Or if structured differently, look at results key in the root:
    # Actually let's look at the "results" array or "trials" or "failure_analysis"
    # Let's inspect the keys of data:
    # Let's write a robust way to find passes:
    if "results" in data:
        for r in data["results"]:
            if r.get("pass_at_k") or r.get("passed"):
                passed_scenarios.append(r["id"])
    else:
        # Let's check scenarios key inside failure_analysis
        scenarios = data.get("failure_analysis", {}).get("scenarios", {})
        for name, info in scenarios.items():
            if not info.get("categories"): # empty categories means no failures
                passed_scenarios.append(name)
                
    model_passes[model_id] = passed_scenarios

for m, passes in sorted(model_passes.items(), key=lambda x: x[0]):
    print(f"{m:<40} passed: {passes}")
