import json

with open("data/openvoicecs/scenarios_v0.1.json") as f:
    data = json.load(f)

for s in data["scenarios"]:
    if s["id"] == "healthcare-admin-schedule-refill-001":
        print(json.dumps(s, indent=2))
        break
