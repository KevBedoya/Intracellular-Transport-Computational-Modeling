import time
import threading
from datetime import datetime
from . import queue_persistence as Qpers
from . import hist_ent
from . import contr


class JobQueue:
    def __init__(self, persistence_path="saved_queue_jobs.json"):
        self.jobs = []
        self.observation_list = []
        self.lock = threading.Lock()
        self.current_job = None
        self.queue_thread = None
        self.running = False
        self.persistence_path = persistence_path

        self.restore_queue()

    def enqueue(self, record: hist_ent.ComputationRecord):
        with self.lock:
            self.jobs.append(record)
            # will need to implement the time_for_execution into the ComputationRecord
            self.jobs.sort(key=lambda job: job.time_for_execution or 0)
            self.save()

    def save(self):
        Qpers.save_queue_to_json(self.jobs, self.persistence_path)

    def restore_queue(self):
        restored = Qpers.load_queue_from_json(self.persistence_path)
        if restored:
            self.jobs = restored

    def start_queue(self):
        if not self.running:
            self.running = True
            self.queue_thread = threading.Thread(target=self._run_loop)
            self.queue_thread.start()

    def stop_queue(self):
        self.running = False
        if self.queue_thread:
            self.queue_thread.join()

    def _run_loop(self):
        while self.running:
            with self.lock:
                if not self.jobs:
                    break

                now = time.time()
                self.jobs.sort(key=lambda job: (job.time_for_execution or 0))
                job = self.jobs[0]

                if job.time_for_execution and job.time_for_execution > now:
                    time.sleep(min(5, job.time_for_execution - now))
                    continue

                self.current_job = job
                job.status = "running"
                self.save()

            try:
                result = contr.run_selected_computation(job.comp_type, job.params)
                job.mfpt = result.get("MFPT")
                job.duration = result.get("duration")
                job.csv_files = result.get("csv_files") or []
                job.png_files = result.get("png_files") or []
                job.status = "completed"
                job.error_msg = None
            except Exception as e:
                job.status = "failed"
                job.error_msg = str(e)
            finally:
                with self.lock:
                    completed_job = self.jobs.pop(0)
                    self.observation_list.append(completed_job)
                    self.save()
                    self.current_job = None

    def cancel_current_job(self):
        with self.lock:
            self.running = False
            if self.current_job:
                self.current_job.status = "cancelled"
                if self.current_job in self.jobs:
                    self.jobs.remove(self.current_job)
                self.save()
                self.current_job = None

    def clear_queue(self):
        with self.lock:
            self.jobs.clear()
            self.save()

    def get_jobs(self):
        with self.lock:
            return list(self.jobs)

    def get_completed_jobs(self):
        with self.lock:
            return list(self.observation_list)

    def remove(self, job_to_remove):
        self.jobs = [job for job in self.jobs if job.job_id != job_to_remove.job_id]
        self.save()

    def getComputationType(self, target_job):

        result_job = None
        for job in self.jobs:
            if job.job_id == target_job.job_id:
                result_job = target_job
        return result_job.comp_type

    @staticmethod
    def getID(job):
        # job_ID = [job for job in self.jobs if job.job_id == job_input.job_id]
        return job.job_id


global_queue = JobQueue()
