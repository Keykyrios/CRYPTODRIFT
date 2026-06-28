"""
CryptoDrift main window — 4-panel research instrument layout.

Verified PyQt6 APIs:
- QMainWindow: from PyQt6.QtWidgets
- QSplitter: from PyQt6.QtWidgets, orientation via Qt.Orientation.Horizontal/Vertical
- QStatusBar: from PyQt6.QtWidgets
- QMenuBar, QMenu, QAction: QAction is from PyQt6.QtGui (NOT QtWidgets in PyQt6!)
- QLabel, QWidget, QVBoxLayout, QHBoxLayout: from PyQt6.QtWidgets
- QTimer: from PyQt6.QtCore

CRITICAL PyQt6 difference from PyQt5:
- QAction moved from QtWidgets to QtGui
- Enums are scoped: Qt.Orientation.Horizontal, not Qt.Horizontal
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QStatusBar,
    QMenuBar,
    QMenu,
    QPushButton,
    QComboBox,
    QToolBar,
    QProgressBar,
)
from PyQt6.QtGui import QAction, QFont  # QAction is in QtGui in PyQt6!
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from .panels.code_editor import CodeEditorPanel
from .panels.attention_heatmap import AttentionHeatmapPanel
from .panels.vuln_timeline import VulnTimelinePanel
from .panels.drift_alerts import DriftAlertsPanel
from .data_loader import ExperimentDataLoader, DriftAlertView
from .theme import COLORS, get_font_ui

logger = logging.getLogger(__name__)


class CryptoDriftMainWindow(QMainWindow):
    """
    Main window with 4-panel layout:

    ┌──────────────┬─────────────────────┬──────────────────┐
    │              │                     │                  │
    │  Code Editor │  Attention Heatmap  │  Vuln Timeline   │
    │  (with diffs)│  (32×32 live grid)  │  (line chart)    │
    │              │                     │                  │
    ├──────────────┴─────────────────────┴──────────────────┤
    │  Drift Alerts + Session Controls                      │
    └───────────────────────────────────────────────────────┘
    """

    # Signals
    iteration_requested = pyqtSignal(str, str, int)  # code, strategy, iteration

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CryptoDrift \u2014 Mechanistic Crypto Degradation Analysis")
        self.setMinimumSize(1280, 720)
        self.resize(1600, 900)

        self._data_loader = ExperimentDataLoader()
        self._sessions = []
        self._current_session = None

        self._setup_menubar()
        self._setup_toolbar()
        self._setup_panels()
        self._setup_statusbar()
        self._load_all_sessions()

    def _setup_menubar(self) -> None:
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_action = QAction("Load Corpus...", self)
        load_action.setShortcut("Ctrl+O")
        file_menu.addAction(load_action)

        export_action = QAction("Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Experiment menu
        exp_menu = menubar.addMenu("&Experiment")

        run_action = QAction("Run Session...", self)
        run_action.setShortcut("F5")
        exp_menu.addAction(run_action)

        step_action = QAction("Step One Iteration", self)
        step_action.setShortcut("F6")
        exp_menu.addAction(step_action)

        exp_menu.addSeparator()

        batch_action = QAction("Run Full Experiment...", self)
        exp_menu.addAction(batch_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        toggle_alerts = QAction("Toggle Alerts Panel", self)
        toggle_alerts.setShortcut("Ctrl+B")
        view_menu.addAction(toggle_alerts)

    def _setup_toolbar(self) -> None:
        """Create the toolbar with session controls."""
        toolbar = QToolBar("Session Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Session selector
        session_label = QLabel("  Session: ")
        toolbar.addWidget(session_label)

        self.session_combo = QComboBox()
        self.session_combo.setMinimumWidth(280)
        self.session_combo.currentIndexChanged.connect(self._on_session_changed)
        toolbar.addWidget(self.session_combo)

        toolbar.addSeparator()

        # Iteration selector
        iter_label = QLabel("  Iteration: ")
        toolbar.addWidget(iter_label)

        self.iter_combo = QComboBox()
        self.iter_combo.setMinimumWidth(80)
        self.iter_combo.currentIndexChanged.connect(self._on_iteration_changed)
        toolbar.addWidget(self.iter_combo)

        toolbar.addSeparator()

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Idle")
        self.progress_bar.setValue(0)
        toolbar.addWidget(self.progress_bar)

    def _setup_panels(self) -> None:
        """Create the 4-panel layout using QSplitters."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(2)

        # Vertical splitter: top panels | bottom alerts
        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(self.v_splitter)

        # Horizontal splitter for top three panels
        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.v_splitter.addWidget(self.h_splitter)

        # Left panel: Code Editor
        self.code_panel = CodeEditorPanel()
        self.h_splitter.addWidget(self.code_panel)

        # Center panel: Attention Heatmap
        self.heatmap_panel = AttentionHeatmapPanel()
        self.h_splitter.addWidget(self.heatmap_panel)

        # Right panel: Vulnerability Timeline
        self.timeline_panel = VulnTimelinePanel()
        self.h_splitter.addWidget(self.timeline_panel)

        # Set initial splitter sizes (3:4:3 ratio)
        self.h_splitter.setSizes([300, 400, 300])

        # Bottom panel: Drift Alerts
        self.alerts_panel = DriftAlertsPanel()
        self.v_splitter.addWidget(self.alerts_panel)

        # Set vertical splitter sizes (75% top, 25% bottom)
        self.v_splitter.setSizes([700, 200])

    def _setup_statusbar(self) -> None:
        """Create the status bar."""
        statusbar = QStatusBar()
        self.setStatusBar(statusbar)

        # GPU memory indicator
        self.gpu_label = QLabel("GPU: --")
        self.gpu_label.setStyleSheet(f"color: {COLORS['fg_muted']};")
        statusbar.addPermanentWidget(self.gpu_label)

        # Current session info
        self.session_label = QLabel("No active session")
        statusbar.addWidget(self.session_label)

        # Iteration counter
        self.iter_label = QLabel("Iteration: 0/0")
        statusbar.addPermanentWidget(self.iter_label)

    # ── Data Loading ───────────────────────────────────────────────

    def _load_all_sessions(self) -> None:
        """Load all sessions from the database."""
        try:
            self._sessions = self._data_loader.get_sessions()
        except Exception as e:
            logger.error("Failed to load sessions: %s", e)
            self._sessions = []
            return

        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        for s in self._sessions:
            vuln_total = sum(it.vuln_count for it in s.iterations)
            label = f"{s.corpus_sample} x {s.strategy}"
            if vuln_total > 0:
                label += f"  [{vuln_total} vulns]"
            self.session_combo.addItem(label)
        self.session_combo.blockSignals(False)

        if self._sessions:
            # Auto-select the session with most vulns
            best = max(range(len(self._sessions)),
                       key=lambda i: sum(it.vuln_count for it in self._sessions[i].iterations))
            self.session_combo.setCurrentIndex(best)
            self._on_session_changed(best)

    def _on_session_changed(self, index: int) -> None:
        """Handle session selection change."""
        if index < 0 or index >= len(self._sessions):
            return

        self._current_session = self._sessions[index]
        session = self._current_session

        # Update iteration selector
        self.iter_combo.blockSignals(True)
        self.iter_combo.clear()
        for it in session.iterations:
            self.iter_combo.addItem(str(it.iteration_num))
        self.iter_combo.blockSignals(False)

        # Update session info
        self.session_label.setText(
            f"Session: {session.session_id[:8]}... | "
            f"{session.corpus_sample} x {session.strategy}"
        )

        # Load full timeline for this session
        self.timeline_panel.clear()
        for it in session.iterations:
            self.timeline_panel.add_point(
                it.iteration_num, it.vuln_count, it.vuln_types
            )

        # Show last iteration by default
        if session.iterations:
            self.iter_combo.setCurrentIndex(len(session.iterations) - 1)
            self._on_iteration_changed(len(session.iterations) - 1)

        self.progress_bar.setMaximum(session.total_iterations)
        self.progress_bar.setValue(len(session.iterations))
        self.progress_bar.setFormat(
            f"{len(session.iterations)}/{session.total_iterations} iterations"
        )

    def _on_iteration_changed(self, index: int) -> None:
        """Handle iteration selection change."""
        if (self._current_session is None
                or index < 0
                or index >= len(self._current_session.iterations)):
            return

        it = self._current_session.iterations[index]

        # Update code panel
        self.code_panel.set_code(it.code_after, it.iteration_num)

        # Update heatmap
        if it.entropy_matrix is not None and it.entropy_matrix.any():
            self.heatmap_panel.update_data(it.entropy_matrix)
        else:
            # Use mean_entropy as a uniform fill so it's not blank
            uniform = np.full((32, 32), it.mean_entropy, dtype=np.float32)
            self.heatmap_panel.update_data(uniform)

        # Update iteration label
        self.iter_label.setText(
            f"Iteration: {it.iteration_num} | "
            f"Entropy: {it.mean_entropy:.4f} | "
            f"Vulns: {it.vuln_count}"
        )

        # Add alerts for any vulns at this iteration
        self.alerts_panel.clear()
        import time as _time
        for v in it.vulns:
            alert = DriftAlertView(
                timestamp=_time.time(),
                severity=v.get("severity", "HIGH"),
                layer=-1,
                head=-1,
                predicted_vuln_type=v.get("vuln_type", "UNKNOWN"),
                message=v.get("description", "")[:120],
            )
            self.alerts_panel.add_alert(alert)

    # ── Public Update Methods ─────────────────────────────────────

    def update_code(
        self, code: str, iteration: int, diff_lines: Optional[list] = None
    ) -> None:
        """Update the code editor panel."""
        self.code_panel.set_code(code, iteration)

    def update_heatmap(self, entropy_matrix) -> None:
        """Update the attention heatmap."""
        self.heatmap_panel.update_data(entropy_matrix)

    def update_timeline(
        self, iteration: int, vuln_count: int, vuln_types: dict
    ) -> None:
        """Add a data point to the vulnerability timeline."""
        self.timeline_panel.add_point(iteration, vuln_count, vuln_types)

    def add_alert(self, warning) -> None:
        """Add a drift alert to the alerts panel."""
        self.alerts_panel.add_alert(warning)

    def set_progress(self, current: int, total: int) -> None:
        """Update the progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"Iteration {current}/{total}")

    def update_gpu_status(self, info: dict) -> None:
        """Update GPU memory display."""
        if info.get("available"):
            used = info.get("allocated_mb", 0)
            total = info.get("total_mb", 0)
            self.gpu_label.setText(
                f"GPU: {used:.0f}/{total:.0f} MB"
            )
        else:
            self.gpu_label.setText("GPU: N/A")

    def set_session_info(self, session_id: str, strategy: str) -> None:
        """Update session info in status bar."""
        self.session_label.setText(
            f"Session: {session_id[:8]}... | Strategy: {strategy}"
        )

    def set_iteration_info(self, current: int, total: int) -> None:
        """Update iteration counter."""
        self.iter_label.setText(f"Iteration: {current}/{total}")
