"""
queue_ops.py
============

:class:`QueueMixin` -- everything related to the job queue: enqueueing,
reordering, editing, removing, and running queued computations.

Two execution paths exist:

* :meth:`run_job_queue_mp` (the wired-up one) drains the queue **one job at a
  time** using the same out-of-process worker as a single run, polling each
  child via a :class:`QTimer` so the UI stays responsive
  (:meth:`run_next_job` -> :meth:`check_job_result` ->
  :meth:`handle_completed_job` / :meth:`handle_failed_job`).
* :meth:`run_job_queue` is the older **synchronous** variant kept as a fallback;
  it is not connected to any button.

Visibility helpers (:meth:`toggle_move_buttons_visibility`,
:meth:`update_queue_controls_visibility`,
:meth:`update_queue_execute_button_visibility`) keep the contextual queue
buttons in sync with the queue's contents and the current selection.

The global queue itself (``job_queue.global_queue``) and the numerical backend
are untouched -- this mixin only manages the GUI representation of the queue.
"""

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMessageBox, QListWidgetItem, QDialog, QFormLayout, QLabel, QLineEdit,
    QDialogButtonBox,
)

import sys

from project_src_package_2025.job_queuing_system import job_queue

from .. import aux_gui_funcs
from .. import computation_history_entry
from .. import history_cache
from .. import controller


class QueueMixin:
    """Job-queue management and batch execution."""

    # ----------------------------- enqueue ----------------------------- #
    def enqueue_job(self):
        """Snapshot the current inputs into a pending job and queue it."""
        inputs = {param: field.text() for param, field in self.param_inputs.items()}
        comp_type = self.comp_select.currentText()

        job = computation_history_entry.ComputationRecord(
            comp_type=comp_type,
            params=inputs,
            status="pending",
            time_for_execution=0
        )

        job_queue.global_queue.enqueue(job)
        self.output_display.append(f"Job enqueued: {job.display_name()}     [{self.produce_timestamp()}]")

        item = QListWidgetItem(job.display_name())
        item.setData(Qt.UserRole, job)
        self.queue_list_widget.addItem(item)
        self.update_queue_controls_visibility()
        self.update_queue_execute_button_visibility()

    # ------------------------- batch run (MP) -------------------------- #
    def run_job_queue_mp(self):
        """Begin draining the queue using the out-of-process worker."""
        self.pending_jobs = job_queue.global_queue.get_jobs()
        if not self.pending_jobs:
            QMessageBox.information(self, "Job Queue", "No jobs in the queue")
            return
        self.output_display.append(
            f"Starting {len(self.pending_jobs)} queued job(s) ...        [{self.produce_timestamp()}]")
        self.queue_list_widget.clear()
        self.run_next_job()

    def run_next_job(self):
        """Pop and launch the next pending job, or finish if none remain."""
        import json

        if not self.pending_jobs:
            self.output_display.append(f"Job queue finished.     [{self.produce_timestamp()}]")
            job_queue.global_queue.clear_queue()
            self.update_history_dropdown_visibility()
            return

        self.spinner_label.setVisible(True)
        self.spinner_movie.start()

        self.current_job = self.pending_jobs.pop(0)
        job = self.current_job

        self.output_display.append(f"Running: {job.display_name()}      [{self.produce_timestamp()}]")
        self.set_launch_color("running")
        self.output_display.repaint()

        python_exec = sys.executable

        computation_name = job.comp_type
        inputs = job.params

        args = [
            python_exec,
            "-m",
            "multiprocessing_tools.subprocess_launcher",
            computation_name,
            json.dumps(inputs)
        ]

        from project_src_package_2025.multiprocessing_tools import subprocess_launcher
        self.process = subprocess_launcher.launch_subprocess(args)

        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self.check_job_result)
        self.poll_timer.start(100)

    def check_job_result(self):
        """Poll the current job's worker; dispatch its result when it exits."""
        import os
        import json

        if self.process.poll() is None:
            return
        self.poll_timer.stop()

        from system_configuration import file_paths as fp
        output_dir = fp.json_output
        result_file = os.path.join(output_dir, f"result.json")

        try:
            if not os.path.join(result_file) or os.path.getsize(result_file) == 0:
                raise FileNotFoundError(f"Result: {result_file} file missing or empty")

            with open(result_file, "r") as f:
                response = json.load(f)
            os.remove(result_file)

            if response["status"] == "ok":
                self.handle_completed_job(response["result"])
            else:
                self.handle_failed_job(response["message"])
        except Exception as e:
            self.output_display.append(
                f"{e} [{self.produce_timestamp()}]"
            )

    def handle_completed_job(self, result):
        """Record a finished job, refresh the result widgets, run the next one."""
        job = self.current_job
        job.status = "completed"
        job.error_msg = None

        job.mfpt = result.get("MFPT")
        job.duration = result.get("duration")

        output_dirs = result.get("output_dirs")
        if output_dirs:
            job.csv_files, job.png_files = aux_gui_funcs.extract_csv_and_png_paths(output_dirs)

        if job.mfpt:
            self.output_display.append(f"Computation returned MFPT = {job.mfpt:.6f}     [{self.produce_timestamp()}]")

        if job.duration:
            self.output_display.append(
                f"Dimensionless time: {job.duration}:.6f     [{self.produce_timestamp()}]")

        if job.csv_files or job.png_files:
            self.output_files_widget.update_display(job.csv_files, job.png_files)
            self.output_files_widget.show()
            self.png_preview_widget.update_png_list(job.png_files)
            self.png_preview_widget.show()
        else:
            self.output_files_widget.hide()
            self.png_preview_widget.hide()

        self.output_display.append(
            f"Job: {job_queue.global_queue.getComputationType(job)} executed successfully.       [{self.produce_timestamp()}]")

        # Record and clean up
        self.set_launch_color("success")
        history_cache.cache.add_entry(job)
        self.history_dropdown.addItem(job.display_name())
        self.remove_visual_job(job)
        self.spinner_label.setVisible(False)
        self.spinner_movie.stop()
        self.run_next_job()

    def handle_failed_job(self, error_msg):
        """Record a failed job, surface the error, then run the next one."""
        job = self.current_job
        job.status = "failed"
        job.error_msg = error_msg

        self.set_launch_color("error")
        self.output_display.append(f"Job failed: {error_msg}        [{self.produce_timestamp()}]")
        QMessageBox.critical(self, "Job Failed", f"Job {job.display_name()} failed:\n{error_msg}")

        history_cache.cache.add_entry(job)
        self.history_dropdown.addItem(job.display_name())
        self.remove_visual_job(job)

        self.spinner_label.setVisible(False)
        self.spinner_movie.stop()
        self.run_next_job()

    def run_job_queue(self):
        """Deprecated synchronous queue runner (kept as a non-wired fallback).

        Superseded by :meth:`run_job_queue_mp`, which runs each job out of
        process. Retained for reference / emergencies; not connected to any
        widget. (The stale ``self.mfpt_label`` write from the original code was
        removed -- that label no longer exists -- so this method no longer
        raises if ever invoked.)
        """
        jobs = job_queue.global_queue.get_jobs()
        if not jobs:
            QMessageBox.information(self, "Job Queue", "No jobs in the queue.")
            return

        self.output_display.append(f"Starting {len(jobs)} queued job(s)...      [{self.produce_timestamp()}]")
        self.spinner_label.setVisible(True)
        self.spinner_movie.start()

        while jobs:
            job = jobs.pop(0)

            self.output_display.append(f"Running: {job.display_name()}")

            try:
                result = controller.run_selected_computation(job.comp_type, job.params)

                job.mfpt = result.get("MFPT")
                job.duration = result.get("duration")
                job.status = "completed"
                job.error_msg = None

                output_dirs = result.get("output_dirs")
                if output_dirs:
                    job.csv_files, job.png_files = aux_gui_funcs.extract_csv_and_png_paths(output_dirs)

                # Update display like in run_computation()
                if job.mfpt:
                    self.output_display.append(
                        f"Computation returned MFPT = {job.mfpt:.6f}         [{self.produce_timestamp()}]")
                if job.duration:
                    self.output_display.append(
                        f"Dimensionless time: {job.duration}:.6f     [{self.produce_timestamp()}]")

                if job.csv_files or job.png_files:
                    self.output_files_widget.update_display(job.csv_files, job.png_files)
                    self.png_preview_widget.update_png_list(job.png_files)
                    self.output_files_widget.show()
                    self.png_preview_widget.show()
                else:
                    self.output_files_widget.hide()
                    self.png_preview_widget.hide()

            except Exception as e:
                job.status = "failed"
                job.error_msg = str(e)
                QMessageBox.critical(self, "Error", f"Job {job.display_name()} failed:\n{str(e)}")
                self.output_display.append(f"Job failed: {str(e)}       [{self.produce_timestamp()}]")

            # Save and archive job
            history_cache.cache.add_entry(job)
            self.history_dropdown.addItem(job.display_name())
            self.remove_visual_job(job)
            self.spinner_label.setVisible(False)
            self.spinner_movie.stop()

        # Clear queue after finishing
        job_queue.global_queue.clear_queue()
        self.output_display.append(f"Job queue finished.     [{self.produce_timestamp()}]")
        self.queue_list_widget.clear()
        self.update_history_dropdown_visibility()

    # ----------------------- queue list operations --------------------- #
    def remove_visual_job(self, job):
        """Remove the list-widget row backed by ``job`` (if present)."""
        for i in range(self.queue_list_widget.count()):
            item = self.queue_list_widget.item(i)
            if item.data(Qt.UserRole) == job:
                self.queue_list_widget.takeItem(i)
                break

    def edit_selected_job(self):
        """Open a modal dialog to edit the selected job's parameters."""
        item = self.queue_list_widget.currentItem()
        if not item:
            return

        job = item.data(Qt.UserRole)

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Job Parameters")
        layout = QFormLayout(dialog)

        fields = {}
        for key, val in job.params.items():
            line = QLineEdit(str(val))
            layout.addRow(QLabel(key), line)
            fields[key] = line

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        def on_accept():
            for key in job.params:
                job.params[key] = fields[key].text()
            item.setText(job.display_name())  # Update label if needed
            dialog.accept()

        button_box.accepted.connect(on_accept)
        button_box.rejected.connect(dialog.reject)

        dialog.exec_()

    def move_selected_job_up(self):
        """Move the selected job one position earlier in the queue."""
        row = self.queue_list_widget.currentRow()
        if row <= 0:
            return

        item = self.queue_list_widget.takeItem(row)
        self.queue_list_widget.insertItem(row - 1, item)
        self.queue_list_widget.setCurrentRow(row - 1)

        job_queue.global_queue.jobs[row], job_queue.global_queue.jobs[row - 1] = \
            job_queue.global_queue.jobs[row - 1], job_queue.global_queue.jobs[row]

    def move_selected_job_down(self):
        """Move the selected job one position later in the queue."""
        row = self.queue_list_widget.currentRow()
        if row < 0 or row >= self.queue_list_widget.count() - 1:
            return

        item = self.queue_list_widget.takeItem(row)
        self.queue_list_widget.insertItem(row + 1, item)
        self.queue_list_widget.setCurrentRow(row + 1)

        job_queue.global_queue.jobs[row], job_queue.global_queue.jobs[row + 1] = \
            job_queue.global_queue.jobs[row + 1], job_queue.global_queue.jobs[row]

    def remove_selected_job(self):
        """Remove the selected job from both the queue and the list widget."""
        item = self.queue_list_widget.currentItem()
        if not item:
            return

        job = item.data(Qt.UserRole)
        self.output_display.append(
            f"Removed job: '{job_queue.global_queue.getComputationType(job)}' from queue.      [{self.produce_timestamp()}]")
        # Remove from GUI
        job_queue.global_queue.remove(job)
        self.queue_list_widget.takeItem(self.queue_list_widget.currentRow())
        self.update_queue_controls_visibility()
        self.update_queue_execute_button_visibility()

    def clear_entire_queue(self):
        """Empty the queue and reset the queue UI."""
        # Clear job queue
        job_queue.global_queue.clear_queue()

        # Clear GUI
        self.queue_list_widget.clear()
        self.update_queue_controls_visibility()
        self.update_queue_execute_button_visibility()
        self.output_display.clear()

    # ------------------------- visibility helpers ---------------------- #
    def toggle_move_buttons_visibility(self):
        """Show the move/remove/edit buttons only when a row is selected."""
        selected = len(self.queue_list_widget.selectedItems()) > 0

        self.move_up_button.setVisible(selected)
        self.move_down_button.setVisible(selected)
        self.remove_job_button.setVisible(selected)
        self.edit_job_button.setVisible(selected)

    def update_queue_controls_visibility(self):
        """Show the Clear-Queue button only when the queue is non-empty."""
        has_jobs = self.queue_list_widget.count() > 0
        self.clear_queue_button.setVisible(has_jobs)

    def update_queue_execute_button_visibility(self):
        """Show the Execute-Queue button only when the queue is non-empty."""
        has_jobs = self.queue_list_widget.count() > 0
        self.run_queue_button.setVisible(has_jobs)

    def get_job_queue(self):
        """Repopulate the list widget from the persisted global queue."""
        for job in job_queue.global_queue.jobs:
            item = QListWidgetItem(job.display_name())
            item.setData(Qt.UserRole, job)
            self.queue_list_widget.addItem(item)
