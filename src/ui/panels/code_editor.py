"""
Code editor panel with Python syntax highlighting.

Verified PyQt6 APIs:
- QSyntaxHighlighter: from PyQt6.QtGui (NOT QtWidgets)
- QTextCharFormat: from PyQt6.QtGui
- QFont.Weight.Bold: scoped enum (NOT QFont.Bold)
- QRegularExpression: from PyQt6.QtCore (NOT re module for Qt)
- QPlainTextEdit: from PyQt6.QtWidgets
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QLabel,
    QComboBox,
)
from PyQt6.QtGui import (
    QSyntaxHighlighter,
    QTextCharFormat,
    QFont,
    QColor,
)
from PyQt6.QtCore import QRegularExpression

from ..theme import COLORS, get_font_mono


class PythonHighlighter(QSyntaxHighlighter):
    """Basic Python syntax highlighter for the code editor."""

    def __init__(self, document):
        super().__init__(document)
        self._rules: list[tuple[QRegularExpression, QTextCharFormat]] = []
        self._setup_rules()

    def _setup_rules(self) -> None:
        # Keywords
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("#c792ea"))
        kw_fmt.setFontWeight(QFont.Weight.Bold)
        keywords = [
            "and", "as", "assert", "async", "await", "break", "class",
            "continue", "def", "del", "elif", "else", "except", "finally",
            "for", "from", "global", "if", "import", "in", "is", "lambda",
            "nonlocal", "not", "or", "pass", "raise", "return", "try",
            "while", "with", "yield", "True", "False", "None",
        ]
        for kw in keywords:
            pattern = QRegularExpression(rf"\b{kw}\b")
            self._rules.append((pattern, kw_fmt))

        # Strings (single and double quoted)
        str_fmt = QTextCharFormat()
        str_fmt.setForeground(QColor("#c3e88d"))
        self._rules.append((
            QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'),
            str_fmt,
        ))
        self._rules.append((
            QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"),
            str_fmt,
        ))

        # Numbers
        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor("#f78c6c"))
        self._rules.append((
            QRegularExpression(r"\b\d+\.?\d*\b"),
            num_fmt,
        ))

        # Comments
        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor("#546e7a"))
        self._rules.append((
            QRegularExpression(r"#[^\n]*"),
            comment_fmt,
        ))

        # Function/method definitions
        func_fmt = QTextCharFormat()
        func_fmt.setForeground(QColor("#82aaff"))
        self._rules.append((
            QRegularExpression(r"\bdef\s+(\w+)"),
            func_fmt,
        ))

        # Decorators
        deco_fmt = QTextCharFormat()
        deco_fmt.setForeground(QColor("#ffcb6b"))
        self._rules.append((
            QRegularExpression(r"@\w+"),
            deco_fmt,
        ))

        # Crypto-relevant keywords (highlighted specially)
        crypto_fmt = QTextCharFormat()
        crypto_fmt.setForeground(QColor(COLORS["accent_cyan"]))
        crypto_fmt.setFontWeight(QFont.Weight.Bold)
        crypto_kw = [
            "AES", "RSA", "HMAC", "SHA", "PBKDF2", "GCM", "CBC",
            "ECB", "encrypt", "decrypt", "urandom", "token_bytes",
            "compare_digest", "scrypt", "Fernet", "nonce", "iv",
        ]
        for kw in crypto_kw:
            self._rules.append((
                QRegularExpression(rf"\b{kw}\b"),
                crypto_fmt,
            ))

    def highlightBlock(self, text: str) -> None:
        """Apply highlighting rules to a block of text."""
        for pattern, fmt in self._rules:
            iterator = pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(
                    match.capturedStart(),
                    match.capturedLength(),
                    fmt,
                )


class CodeEditorPanel(QWidget):
    """
    Left panel: Code editor with syntax highlighting.

    Shows the current iteration's code with Python highlighting
    and crypto-keyword emphasis.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_iteration = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        title = QLabel("Code Editor")
        title.setStyleSheet(
            f"font-weight: bold; color: {COLORS['accent_blue']}; font-size: 14px;"
        )
        header.addWidget(title)

        self.iter_selector = QComboBox()
        self.iter_selector.setMinimumWidth(120)
        self.iter_selector.addItem("Iteration 0 (Original)")
        header.addWidget(self.iter_selector)

        layout.addLayout(header)

        # Code editor
        self.editor = QPlainTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setFont(get_font_mono())
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {COLORS['editor_bg']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
            }}
        """)

        # Attach syntax highlighter
        self._highlighter = PythonHighlighter(self.editor.document())

        layout.addWidget(self.editor)

        # Vuln annotation count
        self.vuln_label = QLabel("Vulnerabilities: 0")
        self.vuln_label.setStyleSheet(f"color: {COLORS['fg_muted']};")
        layout.addWidget(self.vuln_label)

    def set_code(self, code: str, iteration: int) -> None:
        """Set the code content for a given iteration."""
        self._current_iteration = iteration
        self.editor.setPlainText(code)

        # Add iteration to selector if new
        label = f"Iteration {iteration}"
        found = False
        for i in range(self.iter_selector.count()):
            if self.iter_selector.itemText(i).startswith(label):
                found = True
                break
        if not found:
            self.iter_selector.addItem(label)
        self.iter_selector.setCurrentIndex(self.iter_selector.count() - 1)

    def set_vuln_count(self, count: int) -> None:
        """Update the vulnerability annotation count."""
        color = COLORS['danger'] if count > 0 else COLORS['success']
        self.vuln_label.setText(f"Vulnerabilities: {count}")
        self.vuln_label.setStyleSheet(f"color: {color};")
