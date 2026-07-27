import json

with open("data/openvoicecs/scenarios_v0.1.json") as f:
    data = json.load(f)

for s in data["scenarios"]:
    if s["id"] == "fintech-fraud-card-replacement-001":
        print(json.dumps(s, indent=2))
        break
