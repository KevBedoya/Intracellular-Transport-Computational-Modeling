import json
import os
from . import hist_ent


def save_queue_to_json(job_list, filepath):
    try:
        data = [job.to_dict() for job in job_list]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[Error] Failed to save job queue: {e}")


def load_queue_from_json(filepath):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return [hist_ent.ComputationRecord.from_dict(entry) for entry in data]
    except Exception as e:
        print(f"[Error] Failed to load job queue: {e}")
        return
