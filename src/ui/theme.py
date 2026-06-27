"""
Dark theme system for CryptoDrift PyQt6 client.

Verified PyQt6 imports:
- QPalette, QColor, QFont: from PyQt6.QtGui
- Qt: from PyQt6.QtCore
- QApplication: from PyQt6.QtWidgets

PyQt6 uses scoped enums (e.g., Qt.GlobalColor.white, not Qt.white).
"""

from __future__ import annotations

from PyQt6.QtGui import QPalette, QColor, QFont
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication


# ── Color Palette ──────────────────────────────────────────────────
# Deep navy dark theme with crypto-research aesthetic

COLORS = {
    # Backgrounds
    "bg_primary": "#0a0e1a",
    "bg_secondary": "#111827",
    "bg_surface": "#1a2235",
    "bg_elevated": "#1f2b3d",
    "bg_hover": "#253347",

    # Foregrounds
    "fg_primary": "#e2e8f0",
    "fg_secondary": "#94a3b8",
    "fg_muted": "#64748b",
    "fg_accent": "#f8fafc",

    # Accent colors
    "accent_blue": "#4fc3f7",
    "accent_cyan": "#22d3ee",
    "accent_purple": "#a78bfa",
    "accent_indigo": "#818cf8",

    # Semantic colors
    "success": "#66bb6a",
    "warning": "#ffb74d",
    "danger": "#ef5350",
    "info": "#4fc3f7",

    # Heatmap colors (for attention visualization)
    "heat_cold": "#1a237e",    # Deep blue — high entropy (safe)
    "heat_cool": "#1565c0",    # Blue
    "heat_neutral": "#ffffff",  # White — neutral
    "heat_warm": "#ff8f00",    # Amber
    "heat_hot": "#d50000",     # Red — low entropy (danger)

    # Borders
    "border": "#2d3a4f",
    "border_light": "#3d4f6a",

    # Code editor
    "editor_bg": "#0d1117",
    "editor_line": "#161b22",
    "editor_selection": "#264f78",

    # Diff colors
    "diff_added_bg": "#1a3a2a",
    "diff_added_fg": "#66bb6a",
    "diff_removed_bg": "#3a1a1a",
    "diff_removed_fg": "#ef5350",
}


def apply_dark_theme(app: QApplication) -> None:
    """
    Apply the CryptoDrift dark theme to the application.

    Uses QPalette for system-level theming + QSS for fine control.
    """
    palette = QPalette()

    # Window backgrounds
    palette.setColor(
        QPalette.ColorRole.Window,
        QColor(COLORS["bg_primary"]),
    )
    palette.setColor(
        QPalette.ColorRole.WindowText,
        QColor(COLORS["fg_primary"]),
    )

    # Base (input fields, text edits)
    palette.setColor(
        QPalette.ColorRole.Base,
        QColor(COLORS["bg_secondary"]),
    )
    palette.setColor(
        QPalette.ColorRole.AlternateBase,
        QColor(COLORS["bg_surface"]),
    )

    # Text
    palette.setColor(
        QPalette.ColorRole.Text,
        QColor(COLORS["fg_primary"]),
    )
    palette.setColor(
        QPalette.ColorRole.PlaceholderText,
        QColor(COLORS["fg_muted"]),
    )

    # Buttons
    palette.setColor(
        QPalette.ColorRole.Button,
        QColor(COLORS["bg_surface"]),
    )
    palette.setColor(
        QPalette.ColorRole.ButtonText,
        QColor(COLORS["fg_primary"]),
    )

    # Highlights
    palette.setColor(
        QPalette.ColorRole.Highlight,
        QColor(COLORS["accent_blue"]),
    )
    palette.setColor(
        QPalette.ColorRole.HighlightedText,
        QColor(COLORS["bg_primary"]),
    )

    # Tooltips
    palette.setColor(
        QPalette.ColorRole.ToolTipBase,
        QColor(COLORS["bg_elevated"]),
    )
    palette.setColor(
        QPalette.ColorRole.ToolTipText,
        QColor(COLORS["fg_primary"]),
    )

    # Links
    palette.setColor(
        QPalette.ColorRole.Link,
        QColor(COLORS["accent_blue"]),
    )

    app.setPalette(palette)

    # Apply stylesheet for fine control
    app.setStyleSheet(get_stylesheet())


def get_stylesheet() -> str:
    """Generate the QSS stylesheet."""
    c = COLORS
    return f"""
        /* Global */
        QWidget {{
            font-family: 'Segoe UI', 'Inter', sans-serif;
            font-size: 13px;
            color: {c['fg_primary']};
        }}

        /* Main Window */
        QMainWindow {{
            background-color: {c['bg_primary']};
        }}

        /* Splitters */
        QSplitter::handle {{
            background-color: {c['border']};
            width: 2px;
            height: 2px;
        }}
        QSplitter::handle:hover {{
            background-color: {c['accent_blue']};
        }}

        /* Scroll bars */
        QScrollBar:vertical {{
            background: {c['bg_secondary']};
            width: 10px;
            border: none;
        }}
        QScrollBar::handle:vertical {{
            background: {c['bg_hover']};
            min-height: 30px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {c['fg_muted']};
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar:horizontal {{
            background: {c['bg_secondary']};
            height: 10px;
            border: none;
        }}
        QScrollBar::handle:horizontal {{
            background: {c['bg_hover']};
            min-width: 30px;
            border-radius: 5px;
        }}

        /* Labels */
        QLabel {{
            color: {c['fg_primary']};
        }}

        /* Group boxes */
        QGroupBox {{
            border: 1px solid {c['border']};
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 14px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: {c['accent_blue']};
        }}

        /* Push buttons */
        QPushButton {{
            background-color: {c['bg_surface']};
            border: 1px solid {c['border']};
            border-radius: 6px;
            padding: 6px 16px;
            color: {c['fg_primary']};
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {c['bg_hover']};
            border-color: {c['accent_blue']};
        }}
        QPushButton:pressed {{
            background-color: {c['bg_elevated']};
        }}
        QPushButton:disabled {{
            color: {c['fg_muted']};
            border-color: {c['bg_surface']};
        }}

        /* Combo boxes */
        QComboBox {{
            background-color: {c['bg_surface']};
            border: 1px solid {c['border']};
            border-radius: 6px;
            padding: 4px 8px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {c['bg_elevated']};
            border: 1px solid {c['border']};
            selection-background-color: {c['accent_blue']};
        }}

        /* Text edits */
        QPlainTextEdit, QTextEdit {{
            background-color: {c['editor_bg']};
            border: 1px solid {c['border']};
            border-radius: 4px;
            selection-background-color: {c['editor_selection']};
        }}

        /* Status bar */
        QStatusBar {{
            background-color: {c['bg_secondary']};
            border-top: 1px solid {c['border']};
            color: {c['fg_secondary']};
        }}

        /* Menu bar */
        QMenuBar {{
            background-color: {c['bg_secondary']};
            border-bottom: 1px solid {c['border']};
        }}
        QMenuBar::item:selected {{
            background-color: {c['bg_hover']};
        }}
        QMenu {{
            background-color: {c['bg_elevated']};
            border: 1px solid {c['border']};
        }}
        QMenu::item:selected {{
            background-color: {c['accent_blue']};
            color: {c['bg_primary']};
        }}

        /* Tab widget */
        QTabBar::tab {{
            background-color: {c['bg_secondary']};
            border: 1px solid {c['border']};
            padding: 6px 14px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{
            background-color: {c['bg_surface']};
            border-bottom-color: {c['bg_surface']};
            color: {c['accent_blue']};
        }}

        /* Progress bar */
        QProgressBar {{
            background-color: {c['bg_surface']};
            border: 1px solid {c['border']};
            border-radius: 4px;
            text-align: center;
            color: {c['fg_primary']};
        }}
        QProgressBar::chunk {{
            background-color: {c['accent_blue']};
            border-radius: 3px;
        }}
    """


def get_font_mono() -> QFont:
    """Get the monospace font for code display."""
    font = QFont("Cascadia Code", 11)
    if not font.exactMatch():
        font = QFont("Consolas", 11)
        if not font.exactMatch():
            font = QFont("Courier New", 11)
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def get_font_ui() -> QFont:
    """Get the UI font."""
    font = QFont("Segoe UI", 10)
    if not font.exactMatch():
        font = QFont("Inter", 10)
    return font
