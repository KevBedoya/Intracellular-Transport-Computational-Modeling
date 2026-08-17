"""Exercise the job API against a running server."""
import json
import sys

import requests

BASE = sys.argv[1].rstrip("/")
ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}  {detail}")


print("== discovery ==")
r = requests.get(f"{BASE}/")
check("GET / lists endpoints", r.status_code == 200 and "endpoints" in r.json())

r = requests.get(f"{BASE}/health")
check("GET /health ok", r.status_code == 200 and r.json()["status"] == "ok",
      r.text[:120])

r = requests.get(f"{BASE}/computations")
comps = r.json().get("computations", [])
check("GET /computations returns 18", len(comps) == 18, f"got {len(comps)}")
check("includes the new kernel",
      "Characteristic Time (mass vs v)" in comps)

print("== schema ==")
r = requests.get(f"{BASE}/computations/Characteristic Time (mass vs v)")
sc = r.json()
req_names = [f["name"] for f in sc.get("required", [])]
check("schema 200", r.status_code == 200)
check("schema lists v_LIST as required", "v_LIST" in req_names, str(req_names))
check("schema hints present",
      all(f.get("hint") for f in sc.get("required", [])))
r = requests.get(f"{BASE}/computations/Nope")
check("unknown computation -> 404", r.status_code == 404)

print("== submit validation ==")
r = requests.post(f"{BASE}/jobs", json={"computation": "Bogus", "params": {}})
check("unknown computation -> 400", r.status_code == 400, r.text[:100])
r = requests.post(f"{BASE}/jobs", json={"params": {}})
check("missing computation -> 400", r.status_code == 400)
r = requests.post(f"{BASE}/jobs", data="not json",
                  headers={"Content-Type": "application/json"})
check("non-JSON body -> 400", r.status_code == 400)
r = requests.post(f"{BASE}/jobs",
                  json={"computation": "Mass Analysis", "params": "oops"})
check("params not an object -> 400", r.status_code == 400, r.text[:100])

print("== submit + retrieve ==")
payload = {
    "computation": "Characteristic Time (mass vs v)",
    "params": {"rg_param": 32, "ry_param": 32, "v_LIST": [10000],
               "w_param": 100, "T_param": 1, "N_LIST": [0, 8, 16, 24],
               "show_plt": False},
    "submitted_by": "api_test",
}
r = requests.post(f"{BASE}/jobs", json=payload)
check("POST /jobs -> 201", r.status_code == 201, r.text[:120])
job_id = r.json()["job_id"]
print(f"        job_id = {job_id}")

r = requests.get(f"{BASE}/jobs/{job_id}")
check("GET /jobs/<full id> -> 200", r.status_code == 200)
check("params round-tripped with real types",
      r.json()["params"]["v_LIST"] == [10000]
      and r.json()["params"]["rg_param"] == 32,
      json.dumps(r.json().get("params")))
check("submitted_by recorded", r.json()["submitted_by"] == "api_test")

r = requests.get(f"{BASE}/jobs/{job_id[:8]}")
check("GET /jobs/<prefix> -> 200", r.status_code == 200)
r = requests.get(f"{BASE}/jobs/ffffffffffff")
check("unknown job -> 404", r.status_code == 404)

r = requests.get(f"{BASE}/jobs?status=queued")
check("GET /jobs?status=queued includes it",
      any(j["id"] == job_id for j in r.json()["jobs"]))
r = requests.get(f"{BASE}/jobs?status=nonsense")
check("bad status -> 400", r.status_code == 400)

print("== outputs (not yet run) ==")
r = requests.get(f"{BASE}/jobs/{job_id}/outputs")
check("outputs 200 with empty list",
      r.status_code == 200 and r.json()["files"] == [], r.text[:120])

print("== path traversal guard ==")
for evil in ["../../../../../../Windows/win.ini",
             "..%2f..%2f..%2fWindows%2fwin.ini"]:
    r = requests.get(f"{BASE}/jobs/{job_id}/outputs/{evil}")
    check(f"rejects {evil[:28]!r}", r.status_code in (403, 404),
          f"status {r.status_code}")

print("== cancel ==")
r = requests.delete(f"{BASE}/jobs/{job_id}")
check("DELETE queued job -> 200", r.status_code == 200, r.text[:120])
r = requests.delete(f"{BASE}/jobs/{job_id}")
check("cancelling twice -> 409", r.status_code == 409, r.text[:120])

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
