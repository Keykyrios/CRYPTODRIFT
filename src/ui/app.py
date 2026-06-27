"""
CryptoDrift application entry point.

Verified:
- QApplication(sys.argv): standard PyQt6 entry
- app.exec() returns int (NOT app.exec_() — exec_() is PyQt5, exec() is PyQt6)
- sys.exit(app.exec()): standard pattern
"""

from __future__ import annotations

import sys
import logging

from PyQt6.QtWidgets import QApplication

from .theme import apply_dark_theme
from .main_window import CryptoDriftMainWindow

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Launch the CryptoDrift desktop application."""
    logger.info("Starting CryptoDrift...")

    app = QApplication(sys.argv)
    app.setApplicationName("CryptoDrift")
    app.setOrganizationName("CryptoDrift Research")

    # Apply dark theme
    apply_dark_theme(app)

    # Create and show main window
    window = CryptoDriftMainWindow()
    window.show()

    logger.info("CryptoDrift UI ready")

    # app.exec() in PyQt6 (NOT app.exec_() which is PyQt5)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
