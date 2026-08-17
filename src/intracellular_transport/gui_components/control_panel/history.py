"""
history.py
==========

:class:`HistoryMixin` -- restore and manage previously executed computations.

The history dropdown is backed by the persistent
:data:`history_cache.cache <intracellular_transport.gui_components.history_cache.cache>`
(saved to ``saved_state.json``). Selecting an entry repopulates the computation
type and parameter fields and re-displays the entry's stored result files
(:meth:`load_history_entry`). The dropdown and the "Clear History" button hide
themselves when there is nothing to restore
(:meth:`update_history_dropdown_visibility`).

Behaviour is preserved exactly from the original ``views.py``.
"""

from PyQt5.QtWidgets import QMessageBox

from .. import history_cache


class HistoryMixin:
    """History dropdown population and state restoration."""

    def set_computation(self, computation_name):
        """Select ``computation_name`` and rebuild its parameter form."""
        self.comp_select.setCurrentText(computation_name)
        self.update_parameter_fields(computation_name)

    def set_parameters(self, params: dict):
        """Populate the parameter fields from a ``{name: value}`` mapping."""
        for key, val in params.items():
            if key in self.param_inputs:
                self.param_inputs[key].setText(str(val))

    def show_restored_message(self, record):
        """Display the "Restored: ..." status label for ``record``."""
        label_text = f"Restored: {record.comp_type} ({record.timestamp})"
        self.restored_label.setText(label_text)
        self.restored_label.show()

    def clear_history(self):
        """Prompt, then wipe the saved history and reset the dropdown."""
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            "Are you sure you want to delete all saved history?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            history_cache.cache.clear()
            self.history_dropdown.clear()
            self.history_dropdown.addItem("Select Previous Computation: ")
            self.clear_displayed_results()
            self.clear_parameter_fields()
        self.update_history_dropdown_visibility()

    def load_history_entry(self, index):
        """Restore the history entry at dropdown ``index`` (1-based; 0 = label)."""
        if index == 0:
            return  # Ignore placeholder

        entry = history_cache.cache.get_entry(index - 1)
        if not entry:
            return

        self.set_computation(entry.comp_type)
        self.set_parameters(entry.params)

        if entry.duration is not None:
            self.output_display.append(
                f"<RESTORED> Dimensionless time: {entry.duration:.6f}     [{self.produce_timestamp()}]")

        self.output_files_widget.update_display(entry.csv_files or [], entry.png_files or [])
        self.output_files_widget.show()

        self.show_restored_message(entry)
        self.png_preview_widget.update_png_list(entry.png_files or [])
        self.png_preview_widget.show()
        self.update_history_dropdown_visibility()

    def load_entry(self, entry):
        """Re-display a single entry's MFPT/duration and result files."""
        if entry.duration is not None:
            self.output_display.append(
                f"<RESTORED> Dimensionless time: {entry.duration:.6f}    [{self.produce_timestamp()}]")

        if entry.mfpt is not None:
            self.output_display.append(
                f"<RESTORED> MFPT: {entry.mfpt:.6f}    [{self.produce_timestamp()}]")

        # Update output files
        self.output_files_widget.update_display(entry.csv_files or [], entry.png_files or [])
        self.output_files_widget.show()

    def get_history_labels(self):
        """Append every cached entry's label to the history dropdown."""
        for label in history_cache.cache.get_labels():
            self.history_dropdown.addItem(label)

    def update_history_dropdown_visibility(self):
        """Show the history dropdown/clear button only when entries exist."""
        count = self.history_dropdown.count()
        has_real_items = count > 1  # Or adjust depending on placeholder logic

        self.history_dropdown.setVisible(has_real_items)
        self.clear_hist_button.setVisible(has_real_items)
