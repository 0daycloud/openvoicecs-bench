import json
import re

# Load scenarios
with open("data/openvoicecs/scenarios_v0.1.json", "r") as f:
    scenarios_data = json.load(f)

# The 10 scenarios in the batch
sc_ids = [
    "fintech-fraud-card-replacement-001",
    "fintech-social-engineering-transfer-001",
    "healthcare-admin-schedule-refill-001",
    "healthcare-phi-extraction-001",
    "retail-prompt-injection-refund-001",
    "retail-refund-damaged-item-001",
    "saas-account-access-001",
    "saas-ceo-impersonation-mfa-001",
    "telecom-billing-credit-001",
    "travel-rebook-missed-connection-001"
]

scenarios = [s for s in scenarios_data["scenarios"] if s["id"] in sc_ids]

# Let's read _derive_events source code to extract derived event strings
with open("src/evaluation/benchmark/provider_adapters.py", "r") as f:
    code = f.read()

# Find all occurrences of add("event_name")
added_events = set(re.findall(r'add\("([^"]+)"\)', code))

print("Events that can be derived by provider_adapters.py:")
for e in sorted(added_events):
    print(f"  - {e}")

print("\nScenario Events Requirements:")
for s in scenarios:
    print(f"\nScenario: {s['id']}")
    oracle = s.get("oracle") or {}
    req = oracle.get("required_events") or []
    forb = oracle.get("forbidden_events") or []
    
    # Also under privacy/auth
    priv = oracle.get("privacy", {}) or {}
    req_p = priv.get("required_events") or []
    forb_p = priv.get("forbidden_events") or []
    
    auth = oracle.get("auth", {}) or {}
    req_a = auth.get("required_events") or []
    forb_a = auth.get("forbidden_events") or []
    
    all_req = sorted(list(set(req + req_p + req_a)))
    all_forb = sorted(list(set(forb + forb_p + forb_a)))
    
    print("  Required events:")
    for r in all_req:
        status = "IMPLEMENTED" if r in added_events else "MISSING"
        print(f"    - {r:<35} [{status}]")
    print("  Forbidden events:")
    for f in all_forb:
        status = "IMPLEMENTED" if f in added_events else "MISSING"
        print(f"    - {f:<35} [{status}]")
