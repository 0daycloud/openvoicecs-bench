import json

with open("data/openvoicecs/scenarios_v0.1.json", "r") as f:
    data = json.load(f)

for s in data["scenarios"]:
    if s["id"] == "saas-account-access-001":
        print(json.dumps(s, indent=2))
        break
