"""
Drift alerts panel — bottom panel showing real-time warnings.

Verified PyQt6 APIs:
- QTableWidget, QTableWidgetItem: from PyQt6.QtWidgets
- QTableWidget.setRowCount(n): set number of rows
- QTableWidget.setColumnCount(n): set number of columns
- QTableWidget.setHorizontalHeaderLabels([...]): set column headers
- QTableWidget.insertRow(position): insert a row
- QTableWidget.setItem(row, col, QTableWidgetItem(text)): set cell
- QTableWidget.scrollToBottom(): auto-scroll
- QHeaderView.SectionResizeMode.Stretch: scoped enum for header sizing
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt

from ..theme import COLORS


class DriftAlertsPanel(QWidget):
    """
    Bottom panel: Real-time drift alerts and warnings.

    Shows a scrolling table of alerts emitted by the early
    warning system, color-coded by severity.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._alert_count = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header with controls
        header = QHBoxLayout()

        title = QLabel("⚡ Drift Alerts")
        title.setStyleSheet(
            f"font-weight: bold; color: {COLORS['danger']}; font-size: 14px;"
        )
        header.addWidget(title)

        header.addStretch()

        self.count_label = QLabel("0 alerts")
        self.count_label.setStyleSheet(f"color: {COLORS['fg_muted']};")
        header.addWidget(self.count_label)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        header.addWidget(clear_btn)

        layout.addLayout(header)

        # Alerts table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Time", "Severity", "Layer", "Head", "Type", "Message",
        ])

        # Column sizing
        h_header = self.table.horizontalHeader()
        h_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.table.verticalHeader().setVisible(False)

        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                gridline-color: {COLORS['border']};
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_surface']};
                border: 1px solid {COLORS['border']};
                padding: 4px;
                font-weight: bold;
            }}
        """)

        layout.addWidget(self.table)

    def add_alert(self, warning) -> None:
        """
        Add a drift warning to the alerts table.

        Args:
            warning: A DriftWarning object with timestamp, layer, head,
                     predicted_vuln_type, confidence, severity, message.
        """
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Time
        ts = time.strftime(
            "%H:%M:%S",
            time.localtime(warning.timestamp),
        )
        self.table.setItem(row, 0, QTableWidgetItem(ts))

        # Severity (color-coded)
        severity_item = QTableWidgetItem(warning.severity)
        severity_colors = {
            "CRITICAL": COLORS["danger"],
            "HIGH": "#ff7043",
            "MEDIUM": COLORS["warning"],
            "LOW": COLORS["fg_muted"],
        }
        color = severity_colors.get(warning.severity, COLORS["fg_muted"])
        severity_item.setForeground(QColor(color))
        self.table.setItem(row, 1, severity_item)

        # Layer and Head
        self.table.setItem(row, 2, QTableWidgetItem(str(warning.layer)))
        self.table.setItem(row, 3, QTableWidgetItem(str(warning.head)))

        # Predicted vuln type
        self.table.setItem(
            row, 4, QTableWidgetItem(warning.predicted_vuln_type)
        )

        # Message
        self.table.setItem(row, 5, QTableWidgetItem(warning.message))

        self._alert_count += 1
        self.count_label.setText(f"{self._alert_count} alerts")

        # Auto-scroll to latest
        self.table.scrollToBottom()

    def clear(self) -> None:
        """Clear all alerts."""
        self.table.setRowCount(0)
        self._alert_count = 0
        self.count_label.setText("0 alerts")
