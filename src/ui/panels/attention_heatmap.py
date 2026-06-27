"""
Attention heatmap panel — 32×32 entropy grid visualization.

Verified pyqtgraph APIs:
- pg.PlotWidget(): Creates a plot widget (inherits QWidget)
- pg.ImageItem(image=data): Creates an image item for heatmaps
- ImageItem.setImage(data): Update the displayed image data
- ImageItem.setLookupTable(lut): Set colormap as numpy array (256, 3) uint8
- ImageItem.setLevels([min, max]): Set data range for colormap
- pg.ColorMap(pos, color): Create colormap from positions + RGBA colors
- ColorMap.getLookupTable(start, stop, nPts): Generate LUT numpy array
- PlotWidget.addItem(item): Add graphics item to the plot
- PlotWidget.setLabel(axis, text): Set axis label
- PlotWidget.setAspectLocked(True): Lock aspect ratio
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
)
from PyQt6.QtCore import Qt

import pyqtgraph as pg

from ..theme import COLORS


def _make_entropy_colormap() -> np.ndarray:
    """
    Create a blue→white→red colormap for entropy display.

    Blue (high entropy = safe) → White (neutral) → Red (low entropy = danger)

    Returns 256×3 uint8 numpy array.
    """
    positions = np.array([0.0, 0.5, 1.0])
    colors = np.array([
        [213, 0, 0, 255],      # Red (low entropy — danger)
        [255, 255, 255, 255],  # White (mid)
        [26, 35, 126, 255],    # Deep blue (high entropy — safe)
    ], dtype=np.ubyte)

    cmap = pg.ColorMap(positions, colors)
    return cmap.getLookupTable(start=0.0, stop=1.0, nPts=256)


class AttentionHeatmapPanel(QWidget):
    """
    Center panel: Attention entropy heatmap.

    Displays a 32×32 grid (layers × heads) where color indicates
    normalized entropy of each attention head:
    - Blue = high entropy (dispersed attention, safe)
    - White = neutral
    - Red = low entropy (concentrated attention, potential danger)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = np.zeros((32, 32), dtype=np.float32)
        self._lut = _make_entropy_colormap()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        title = QLabel("Attention Entropy Heatmap")
        title.setStyleSheet(
            f"font-weight: bold; color: {COLORS['accent_purple']}; font-size: 14px;"
        )
        header.addWidget(title)

        self.layer_selector = QComboBox()
        self.layer_selector.addItem("All Layers (32×32)")
        self.layer_selector.addItem("Crypto Tokens Only")
        header.addWidget(self.layer_selector)
        layout.addLayout(header)

        # Heatmap plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel("left", "Layer")
        self.plot_widget.setLabel("bottom", "Head")

        # Create ImageItem
        self.img_item = pg.ImageItem()
        self.img_item.setLookupTable(self._lut)
        self.img_item.setLevels([0.0, 1.0])  # Normalized entropy range
        self.plot_widget.addItem(self.img_item)

        # Set initial data
        self.img_item.setImage(self._data)

        layout.addWidget(self.plot_widget)

        # Legend labels
        legend = QHBoxLayout()
        low_label = QLabel("🔴 Low Entropy (Danger)")
        low_label.setStyleSheet(f"color: {COLORS['danger']};")
        legend.addWidget(low_label)

        legend.addStretch()

        mid_label = QLabel("⚪ Neutral")
        mid_label.setStyleSheet(f"color: {COLORS['fg_secondary']};")
        legend.addWidget(mid_label)

        legend.addStretch()

        high_label = QLabel("🔵 High Entropy (Safe)")
        high_label.setStyleSheet(f"color: {COLORS['accent_blue']};")
        legend.addWidget(high_label)

        layout.addLayout(legend)

        # Stats line
        self.stats_label = QLabel("Mean entropy: -- | Min: -- (L--, H--)")
        self.stats_label.setStyleSheet(f"color: {COLORS['fg_muted']};")
        layout.addWidget(self.stats_label)

    def update_data(self, entropy_matrix: np.ndarray) -> None:
        """
        Update the heatmap with new entropy data.

        Args:
            entropy_matrix: Shape (32, 32) normalized entropy values.
        """
        if entropy_matrix.shape != (32, 32):
            # Pad or crop to 32×32
            padded = np.zeros((32, 32), dtype=np.float32)
            h = min(entropy_matrix.shape[0], 32)
            w = min(entropy_matrix.shape[1], 32)
            padded[:h, :w] = entropy_matrix[:h, :w]
            entropy_matrix = padded

        self._data = entropy_matrix.astype(np.float32)
        self.img_item.setImage(self._data)

        # Update stats
        mean_e = float(np.mean(self._data))
        min_idx = np.unravel_index(np.argmin(self._data), self._data.shape)
        min_val = float(self._data[min_idx])
        self.stats_label.setText(
            f"Mean entropy: {mean_e:.3f} | "
            f"Min: {min_val:.3f} (L{min_idx[0]}, H{min_idx[1]})"
        )

    def clear(self) -> None:
        """Reset the heatmap to zeros."""
        self._data = np.zeros((32, 32), dtype=np.float32)
        self.img_item.setImage(self._data)
        self.stats_label.setText("Mean entropy: -- | Min: -- (L--, H--)")
