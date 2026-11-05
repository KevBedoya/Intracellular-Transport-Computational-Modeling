"""Queue the modal 96x96 study on the API: 16 Mass Analysis jobs (+ PSD), then one (a, b) char-time grid.

Fixed parameters from the 2026-10-07 study (jobs 0b8d4022 / 54cdf1f3): 96x96, v = 1,
N_LIST = [0, 24, 48, 72], tau = 0.01; T = 4 for every pair.

Mass Analysis stores one sample every 861 * 5 = 4305 steps -- exactly the H ~ 1.0e-4
spacing the 10-07 central differences used (s = 861 samples of 5 steps) -- rather than
every 5 steps, which would be 34 M rows per series per pair.
"""
import json, os, urllib.request

API = "http://100.83.174.69:8000"
AB = [0.1, 1.0, 10.0, 100.0]
FIXED = dict(rg_param=96, ry_param=96, v_param=1.0, T_param=4.0, N_LIST=[0, 24, 48, 72])


def submit(computation, params):
    body = json.dumps({"computation": computation, "params": params,
                       "submitted_by": "claude: modal 96x96 study"}).encode()
    req = urllib.request.Request(API + "/jobs", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)["job_id"]


jobs = {"mass_analysis": {}, "char_time": None}
FIRST = "4dc5d4095cf047caa9537d9e2b2ec2bf"   # (0.1, 0.1), queued before the response-key fix
for a in AB:
    for b in AB:
        if (a, b) == (0.1, 0.1):
            jobs["mass_analysis"]["0.1,0.1"] = FIRST
            continue
        jobs["mass_analysis"][f"{a:g},{b:g}"] = submit("Mass Analysis", dict(
            FIXED, a_param=a, b_param=b, engine="modal", compute_psd=True,
            MA_collection_factor=4305, MA_collection_factor_limit=5000, save_png=True))
jobs["char_time"] = submit("Characteristic Time (a,b grid)", dict(
    FIXED, a_list=AB, b_list=AB, engine="modal", tau=0.01, char_time_mult=1.0,
    compute_psd=True, MA_collection_factor=5, show_plt=False))

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs_96_modal.json")
json.dump(jobs, open(path, "w"), indent=1)
print(json.dumps(jobs, indent=1))
