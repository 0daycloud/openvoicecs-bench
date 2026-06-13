import json

with open("data/openvoicecs/scenarios_v0.1.json", "r") as f:
    data = json.load(f)

if isinstance(data, dict):
    print("Keys of scenarios dict:", list(data.keys())[:10])
    # if scenarios is a list inside the dict:
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0:
            print(f"Key '{k}' is a list of length {len(v)}")
            if isinstance(v[0], dict) and "id" in v[0]:
                print(f"First item in '{k}':", v[0]["id"])
elif isinstance(data, list):
    print("Scenarios is a list of length:", len(data))
    if len(data) > 0:
        print("First item type:", type(data[0]))
        if isinstance(data[0], dict):
            print("First item keys:", list(data[0].keys()))
