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
        self.setWindowTitle("CryptoDrift — Mechanistic Crypto Degradation Analysis")
        self.setMinimumSize(1280, 720)
        self.resize(1600, 900)

        self._setup_menubar()
        self._setup_toolbar()
        self._setup_panels()
        self._setup_statusbar()

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

        # Strategy selector
        strategy_label = QLabel("  Strategy: ")
        toolbar.addWidget(strategy_label)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "EF — Efficiency",
            "FF — Feature",
            "SF — Security",
            "AI — Ambiguous",
        ])
        self.strategy_combo.setMinimumWidth(160)
        toolbar.addWidget(self.strategy_combo)

        toolbar.addSeparator()

        # Iteration controls
        self.btn_step = QPushButton("▶ Step")
        self.btn_step.setToolTip("Run one iteration (F6)")
        toolbar.addWidget(self.btn_step)

        self.btn_run_all = QPushButton("▶▶ Run All")
        self.btn_run_all.setToolTip("Run all iterations (F5)")
        toolbar.addWidget(self.btn_run_all)

        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setToolTip("Stop current run")
        self.btn_stop.setEnabled(False)
        toolbar.addWidget(self.btn_stop)

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
