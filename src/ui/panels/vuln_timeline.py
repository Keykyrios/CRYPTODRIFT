"""
Vulnerability timeline panel — real-time line chart.

Verified pyqtgraph APIs:
- pg.PlotWidget(): plot container
- PlotWidget.addLegend(): add legend (MUST call before plot())
- PlotWidget.plot(x, y, pen, name): returns PlotDataItem
- PlotDataItem.setData(x, y): update data efficiently
- pg.mkPen(color, width): create pen for line styling
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
)
from PyQt6.QtCore import Qt

import pyqtgraph as pg

from ..theme import COLORS


class VulnTimelinePanel(QWidget):
    """
    Right panel: Vulnerability count over iterations.

    Displays:
    - Total vulnerability count per iteration (line chart)
    - Weighted severity score per iteration
    - Per-type breakdown curves
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._iterations: list[int] = []
        self._total_vulns: list[int] = []
        self._scores: list[int] = []
        self._per_type: dict[str, list[int]] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        title = QLabel("Vulnerability Timeline")
        title.setStyleSheet(
            f"font-weight: bold; color: {COLORS['warning']}; font-size: 14px;"
        )
        layout.addWidget(title)

        # Count plot
        self.count_plot = pg.PlotWidget()
        self.count_plot.setLabel("left", "Count")
        self.count_plot.setLabel("bottom", "Iteration")
        self.count_plot.setTitle(
            "Vulnerability Count", color=COLORS["fg_secondary"], size="10pt"
        )
        self.count_plot.showGrid(x=True, y=True, alpha=0.2)
        self.count_plot.addLegend()

        # Create initial curves — hold references for setData updates
        self.total_curve = self.count_plot.plot(
            [], [],
            pen=pg.mkPen(color=COLORS["danger"], width=2),
            name="Total",
        )
        self.score_curve = self.count_plot.plot(
            [], [],
            pen=pg.mkPen(color=COLORS["warning"], width=2, style=Qt.PenStyle.DashLine),
            name="Weighted Score",
        )

        layout.addWidget(self.count_plot)

        # Summary stats
        self.summary_label = QLabel(
            "Vulns: 0 | Δ: 0 | Score: 0"
        )
        self.summary_label.setStyleSheet(f"color: {COLORS['fg_muted']};")
        layout.addWidget(self.summary_label)

    def add_point(
        self,
        iteration: int,
        vuln_count: int,
        vuln_types: dict,
    ) -> None:
        """
        Add a data point to the timeline.

        Args:
            iteration: Iteration number.
            vuln_count: Total vulnerability count.
            vuln_types: Dict of {vuln_type: count}.
        """
        self._iterations.append(iteration)
        self._total_vulns.append(vuln_count)

        # Compute weighted score (simple sum of severity weights)
        score = sum(vuln_types.values())
        self._scores.append(score)

        # Update main curves using setData (NOT repeated plot calls)
        x = np.array(self._iterations, dtype=np.float64)
        self.total_curve.setData(x, np.array(self._total_vulns, dtype=np.float64))
        self.score_curve.setData(x, np.array(self._scores, dtype=np.float64))

        # Update per-type curves
        for vtype, count in vuln_types.items():
            if vtype not in self._per_type:
                self._per_type[vtype] = [0] * (len(self._iterations) - 1)
                # Create a new curve for this type
                color = self._get_type_color(vtype)
                curve = self.count_plot.plot(
                    [], [],
                    pen=pg.mkPen(color=color, width=1),
                    name=vtype,
                )
                self._curves[vtype] = curve
            self._per_type[vtype].append(count)

            # Pad other types that didn't appear
            for other_type in self._per_type:
                if other_type != vtype and len(self._per_type[other_type]) < len(self._iterations):
                    self._per_type[other_type].append(
                        self._per_type[other_type][-1] if self._per_type[other_type] else 0
                    )

        # Update per-type curve data
        for vtype, counts in self._per_type.items():
            if vtype in self._curves:
                y = np.array(counts[-len(self._iterations):], dtype=np.float64)
                self._curves[vtype].setData(x[:len(y)], y)

        # Update summary
        delta = 0
        if len(self._total_vulns) >= 2:
            delta = self._total_vulns[-1] - self._total_vulns[-2]
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        color = COLORS['danger'] if delta > 0 else COLORS['success']
        self.summary_label.setText(
            f"Vulns: {vuln_count} | Δ: {delta_str} | Score: {score}"
        )
        self.summary_label.setStyleSheet(f"color: {color};")

    def clear(self) -> None:
        """Reset all timeline data."""
        self._iterations.clear()
        self._total_vulns.clear()
        self._scores.clear()
        self._per_type.clear()
        self.total_curve.setData([], [])
        self.score_curve.setData([], [])
        for curve in self._curves.values():
            self.count_plot.removeItem(curve)
        self._curves.clear()
        self.summary_label.setText("Vulns: 0 | Δ: 0 | Score: 0")

    @staticmethod
    def _get_type_color(vuln_type: str) -> str:
        """Get a consistent color for a vulnerability type."""
        type_colors = {
            "IV_REUSE": "#ef5350",
            "WEAK_KDF": "#ff7043",
            "TIMING_UNSAFE_COMPARE": "#ab47bc",
            "ECB_MODE": "#e53935",
            "HARDCODED_KEY": "#f44336",
            "PREDICTABLE_RNG": "#ff9800",
            "MAC_THEN_ENCRYPT": "#7e57c2",
            "WEAK_HASH": "#ec407a",
        }
        return type_colors.get(vuln_type, "#78909c")
