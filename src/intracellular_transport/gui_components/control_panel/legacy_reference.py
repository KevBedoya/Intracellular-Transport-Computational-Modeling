"""
legacy_reference.py
===================

Historical / not-yet-active code preserved verbatim from the original
``views.py``. **Nothing in this module is imported or executed** -- it exists
only so the design notes and prototypes that were carried along as comments in
the monolith are not lost in the refactor.

Two groups are preserved below as inert string blocks:

1. *To be implemented in a future version* (dated 7/31/25): per-frame / FPS
   sliders for the animation controls, a multiprocessing animation launcher
   (``handle_launch_animation_mp``), and a JIT-compilation warm-up flow.
2. *To be deprecated* (dated 7/31/25): the original synchronous
   ``run_computation`` (pre-multiprocessing). The live synchronous queue runner
   it inspired still exists as ``QueueMixin.run_job_queue`` (kept but not wired).

If/when any of this is revived, move it into the appropriate mixin and delete it
here.
"""

FUTURE_WORK = r'''
# ===================== To be implemented in a future version [7/31/25] @Kevin =====================

# Steps per frame (SPF) slider
# self.steps_slider = QSlider(Qt.Horizontal)
# self.steps_slider.setMinimum(1)
# self.steps_slider.setMaximum(150)
# self.steps_slider.setValue(10)
# self.steps_slider.setTickInterval(10)
# self.steps_slider.setTickPosition(QSlider.TicksBelow)
#
# self.steps_label = QLabel("Steps per Frame: 10")
# self.steps_slider.valueChanged.connect(
#     lambda val: self.steps_label.setText(f"Steps per Frame: {val}")
# )

# Frames per second (FPS) slider
# self.interval_slider = QSlider(Qt.Horizontal)
# self.interval_slider.setMinimum(10)
# self.interval_slider.setMaximum(60)
# self.interval_slider.setValue(50)
# self.interval_slider.setTickInterval(5)
# self.interval_slider.setTickPosition(QSlider.TicksBelow)
#
# self.interval_label = QLabel("Frames per second: 50")
# self.interval_slider.valueChanged.connect(
#     lambda val: self.interval_label.setText(f"Frames per Second: {val}")
# )

# self.anim_layout.addWidget(self.steps_label)
# self.anim_layout.addWidget(self.steps_slider)
# self.anim_layout.addSpacing(10)
# self.anim_layout.addWidget(self.interval_label)
# anim_layout.addWidget(self.fps_slider)
# self.anim_layout.addWidget(self.interval_slider)
# self.anim_layout.addSpacing(10)

# def handle_launch_animation_mp(self):
#     try:
#         rg = int(self.param_inputs[ "rg_param" ].text())
#         ry = int(self.param_inputs[ "ry_param" ].text())
#         w = float(self.param_inputs[ "w_param" ].text())
#         v = float(self.param_inputs[ "v_param" ].text())
#         N_raw = self.param_inputs[ "N_LIST" ].text()
#         N = list(map(int, re.findall(r'\d+', N_raw))) if N_raw else [ ]
#         T = float(self.param_inputs[ "T_param" ].text())
#         d_tube = float(self.param_inputs[ "d_tube" ].text())
#         K_param = 1000
#
#         # Step 2: Start background process
#         self.ani_queue = Queue()
#         self.ani_process = Process(
#             target=evo.compute_batches_in_background,
#             args=(rg, ry, w, v, N, K_param, T, d_tube, self.ani_queue)
#         )
#         self.ani_process.start()
#
#         # Before calling animate_diffusion_mp
#         start_time = time.time()
#         while self.ani_queue.empty() and time.time() - start_time < 1.0:
#             time.sleep(0.01)  # 10 ms
#
#         # steps_per_frame = self.steps_slider.value()
#         # interval_ms = int(1e3 / self.interval_slider.value())
#
#         steps_per_frame = 50
#         interval_ms = 10
#
#         # Step 3: Launch canvas connected to queue
#         canvas = evo.animate_diffusion_mp(
#             rg, ry, w, v, N, K_param, T, d_tube,
#             steps_per_frame=steps_per_frame,
#             interval_ms=interval_ms,
#             result_queue=self.ani_queue
#         )
#
#         # Step 4: Display canvas
#         if hasattr(self, 'current_canvas') and self.current_canvas:
#             self.visualization_area.removeWidget(self.current_canvas)
#             self.current_canvas.setParent(None)
#
#         self.current_canvas = canvas
#         self.visualization_area.addWidget(canvas)
#         self.visualization_area.setCurrentWidget(canvas)
#
#     except Exception as e:
#         QMessageBox.critical(self, "Error", f"Failed to launch animation: \n{str(e)}")

# def initiate_JIT_compilation(self):
#     self.launch_button.setEnabled(False)
#     self.enqueue_button.setEnabled(False)
#     self.run_queue_button.setEnabled(False)
#     self.launch_animation_button.setEnabled(False)
#
#     self.output_display.append("Initiating JIT numba compilation...")
#     self.output_display.append("Computational options will be temporarily disabled.")
#
#     self.jit_compile_process = controller.initiate_compilation()
#
#     self.jit_timer = QTimer()
#     self.jit_timer.timeout.connect(self.check_jit_compilation_done)
#     self.jit_timer.start(200)
#
# def check_jit_compilation_done(self):
#     if self.jit_compile_process is not None and not self.jit_compile_process.is_alive():
#         self.jit_timer.stop()
#         self.jit_compile_process = None
#
#         self.output_display.append("JIT numba compilation successfully executed.")
#         self.output_display.append("Computational options are now enabled for usage.")
#
#         self.launch_button.setEnabled(True)
#         self.enqueue_button.setEnabled(True)
#         self.run_queue_button.setEnabled(True)
#         self.launch_animation_button.setEnabled(True)
'''

DEPRECATED = r'''
# ===================== To be deprecated [7/31/25] @Kevin =====================
# Original computation launcher implemented WITHOUT multiprocessing.
# Superseded by ComputationMixin.run_computation_mp.

# def run_computation(self):
#     self.set_launch_color("running")
#     self.output_display.clear()
#     self.duration_label.hide()
#
#     try:
#         inputs = {param: field.text() for param, field in self.param_inputs.items()}
#         result = controller.run_selected_computation(self.comp_select.currentText(), inputs)
#         print(f"Result: {result}")  # For now, log result in terminal
#
#         if isinstance(result, dict):
#             if "MFPT" in result:
#                 self.output_display.append(f"Computation returned MFPT = {result['MFPT']:.6f}\n")
#             if "duration" in result:
#                 self.duration_label.setText(f"Duration: {result['duration']:.6f} seconds")
#                 self.duration_label.show()
#         self.set_launch_color("success")
#
#         csv_paths = []
#         png_paths = []
#
#         if "output_dirs" in result:
#             csv_paths, png_paths = aux_gui_funcs.extract_csv_and_png_paths(result["output_dirs"])
#             if csv_paths or png_paths:
#                 self.output_files_widget.update_display(csv_paths, png_paths)
#                 self.output_files_widget.show()
#                 self.png_preview_widget.update_png_list(png_paths)
#                 self.png_preview_widget.show()
#             else:
#                 self.output_files_widget.hide()
#                 self.png_preview_widget.hide()
#         else:
#             self.output_files_widget.hide()
#
#         status = "completed"
#         error_msg = None
#
#         try:
#             result = controller.run_selected_computation(self.comp_select.currentText(), inputs)
#         except Exception as e:
#             result = {}
#             status = "failed"
#             error_msg = str(e)
#             QMessageBox.critical(self, "Error", str(e))
#             self.set_launch_color("error")
#
#         record = computation_history_entry.ComputationRecord(
#             comp_type=self.comp_select.currentText(),
#             params=inputs,
#             mfpt=result.get("MFPT"),
#             duration=result.get("duration"),
#             csv_files=csv_paths,
#             png_files=png_paths,
#             status=status,
#             error_msg=error_msg
#         )
#
#         history_cache.cache.add_entry(record)
#         self.history_dropdown.addItem(record.display_name())
#         self.update_history_dropdown_visibility()
#
#     except Exception as e:
#         QMessageBox.critical(self, "Error", str(e))
#         self.set_launch_color("error")
'''
