import json, time, urllib.request
jobs = json.load(open("jobs_96_modal.json"))
ids = list(jobs["mass_analysis"].values()) + [jobs["char_time"]]
while True:
    st = {i: json.load(urllib.request.urlopen(f"http://100.83.174.69:8000/jobs/{i}")) for i in ids}
    if all(j["status"] in ("succeeded", "failed", "cancelled") for j in st.values()):
        break
    time.sleep(30)
for i, j in st.items():
    print(i[:8], j["computation"], j["status"], j["started_at"], j["finished_at"], (j["error"] or "")[:300])
