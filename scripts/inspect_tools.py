import json

with open("data/openvoicecs/scenarios_v0.1.json", "r") as f:
    data = json.load(f)

sc_ids = [
    "fintech-fraud-card-replacement-001",
    "healthcare-admin-schedule-refill-001"
]

for s in data["scenarios"]:
    if s["id"] in sc_ids:
        print(f"\n==================================================")
        print(f"Scenario ID: {s['id']}")
        print(f"Tools provided to agent:")
        for t in s.get("tools", []):
            print(f"  - {t['name']}")
        print(f"Expected Tool Calls (Oracle):")
        oracle = s.get("oracle", {})
        for tc in oracle.get("expected_tool_calls", []):
            print(f"  - {tc['name']}: {tc.get('arguments')}")
