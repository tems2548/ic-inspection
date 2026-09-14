"""Photo viewer with the detection boxes drawn on top.

When a read looks wrong the fault is usually the cut, not the model, so the
first thing the window has to show is *where* the reader looked: the package
box, each marking segment, and the segments it decided were a logo.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from icinspect.gui import heatmap

Box = tuple[int, int, int, int]

CHIP_COLOR = "#2f81f7"      # package outline
READ_COLOR = "#2ea043"      # a segment that was read
IGNORED_COLOR = "#8b949e"   # a logo or mark that was skipped
SELECTED_COLOR = "#f78166"


@dataclass
class Overlay:
    """One rectangle to draw over the photo, in source-image pixels."""

    box: Box
    color: str
    label: str = ""
    dashed: bool = False


class ImageView(ttk.Frame):
    """Canvas that fits an image to the window, with wheel zoom and drag pan.

    Zoom is stored as a scale about the canvas centre. `fit` mode recomputes it
    on every resize, and any manual zoom leaves fit mode until Reset.
    """

    MIN_SCALE = 0.02
    MAX_SCALE = 20.0

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, background="#1c1f26", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._image: Image.Image | None = None
        self._path: Path | None = None
        # What `_image` actually holds: "file:<path>" or "composite:<key>". The
        # panel grid and the photo it was built from share a path, so the path
        # alone cannot say whether the canvas is already showing the right thing.
        self._key: str | None = None
        self._overlays: list[Overlay] = []
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._offset = (0.0, 0.0)     # top-left of the image in canvas pixels
        self._fit = True
        self._drag: tuple[int, int] | None = None
        self._selected: int | None = None
        self._empty_hint = "No image selected"
        # PatchCore's per-pixel scores for this photo, drawn over it on demand.
        self._heat: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._limit: tuple[float, float, float, float] | None = None
        self.heat_visible = False
        self.heat_opacity = 0.75
        self.heat_floor = heatmap.FLOOR
        self.heat_style = "threshold"     # or "jet", anomalib's own look
        self.mask_visible = True

        self.canvas.bind("<Configure>", lambda _e: self._render())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))
        self.canvas.bind("<Double-Button-1>", lambda _e: self.reset_view())
        self.canvas.bind("<MouseWheel>", self._on_wheel)              # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.1))   # X11
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / 1.1))

    # -- content -----------------------------------------------------------

    def show(self, path: Path | None, overlays: list[Overlay] | None = None, keep_view: bool = False) -> None:
        """Display `path` (None clears the view) with its boxes."""
        key = None if path is None else f"file:{path}"
        same = key is not None and self._key == key
        self._overlays = list(overlays or [])
        self._selected = None
        if path is None:
            self._image = self._path = self._key = None
            self._render()
            return
        if not same:
            try:
                img = Image.open(path)
                img.load()
            except Exception as exc:  # a truncated capture should not kill the window
                self._image = self._path = self._key = None
                self._empty_hint = f"Cannot open {Path(path).name}:\n{exc}"
                self._render()
                return
            self._image = img.convert("RGB") if img.mode not in ("RGB", "L") else img
            self._path = Path(path)
            self._key = key
            if not keep_view:
                self._fit = True
        self._render()

    def set_heat(
        self,
        scores: np.ndarray | None,
        mask: np.ndarray | None = None,
        limit: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Attach (or clear) the anomaly map that belongs to the shown photo.

        `limit` is the region that was scored: outside it the photo is left
        untouched, because it took no part in the verdict.
        """
        self._heat = scores
        self._mask = mask
        self._limit = limit
        self._render()

    def has_heat(self) -> bool:
        return self._heat is not None

    def set_heat_view(
        self,
        visible: bool | None = None,
        opacity: float | None = None,
        floor: float | None = None,
        mask: bool | None = None,
        style: str | None = None,
    ) -> None:
        if visible is not None:
            self.heat_visible = visible
        if opacity is not None:
            self.heat_opacity = opacity
        if floor is not None:
            self.heat_floor = floor
        if mask is not None:
            self.mask_visible = mask
        if style is not None:
            self.heat_style = style
        self._render()

    def show_composite(self, image: Image.Image, key: str) -> None:
        """Show an image built in memory -- the panel grid -- instead of a file.

        `key` stands in for the path so panning and zooming survive a redraw of
        the same photo, and only a genuinely different one snaps back to fit.
        """
        self._overlays = []
        self._selected = None
        composite_key = f"composite:{key}"
        same = self._key == composite_key
        self._image = image.convert("RGB")
        self._path = Path(key)          # kept so "Open folder" still finds the capture
        self._key = composite_key
        if not same:
            self._fit = True
        self._render()

    def set_selected(self, index: int | None) -> None:
        """Highlight one overlay -- the segment picked in the reads list."""
        self._selected = index
        self._render()

    def reset_view(self) -> None:
        self._fit = True
        self._render()

    def zoom_to(self, box: Box | None, margin: float = 0.12) -> None:
        """Fill the canvas with one box -- the package, or a single segment.

        A 4K capture fitted to the pane leaves the marking a few pixels tall,
        which is no use for judging a bad read."""
        if self._image is None or box is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        x1, y1, x2, y2 = box
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        scale = min(cw / (bw * (1 + margin)), ch / (bh * (1 + margin)))
        self._scale = max(self.MIN_SCALE, min(self.MAX_SCALE, scale))
        self._offset = (cw / 2 - (x1 + bw / 2) * self._scale,
                        ch / 2 - (y1 + bh / 2) * self._scale)
        self._fit = False
        self._render()

    def zoom_in(self) -> None:
        self._zoom_at(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, 1.25)

    def zoom_out(self) -> None:
        self._zoom_at(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, 0.8)

    @property
    def path(self) -> Path | None:
        return self._path

    # -- interaction -------------------------------------------------------

    def _on_press(self, event: tk.Event) -> None:
        self._drag = (event.x, event.y)

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag is None or self._image is None:
            return
        dx, dy = event.x - self._drag[0], event.y - self._drag[1]
        self._drag = (event.x, event.y)
        self._fit = False
        self._offset = (self._offset[0] + dx, self._offset[1] + dy)
        self._render()

    def _on_wheel(self, event: tk.Event) -> None:
        self._zoom_at(event.x, event.y, 1.1 if event.delta > 0 else 1 / 1.1)

    def _zoom_at(self, cx: int, cy: int, factor: float) -> None:
        """Zoom keeping the image point under the cursor where it is."""
        if self._image is None:
            return
        new_scale = max(self.MIN_SCALE, min(self.MAX_SCALE, self._scale * factor))
        if new_scale == self._scale:
            return
        ratio = new_scale / self._scale
        ox, oy = self._offset
        self._offset = (cx - (cx - ox) * ratio, cy - (cy - oy) * ratio)
        self._scale = new_scale
        self._fit = False
        self._render()

    # -- drawing -----------------------------------------------------------

    def _fit_view(self, cw: int, ch: int) -> None:
        assert self._image is not None
        iw, ih = self._image.size
        self._scale = min(cw / iw, ch / ih) if iw and ih else 1.0
        self._offset = ((cw - iw * self._scale) / 2, (ch - ih * self._scale) / 2)

    def _render(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        if self._image is None:
            canvas.create_text(
                cw // 2, ch // 2, text=self._empty_hint, fill="#8b949e",
                font=("Segoe UI", 10), justify="center", width=max(cw - 40, 80),
            )
            return

        if self._fit:
            self._fit_view(cw, ch)

        iw, ih = self._image.size
        dw, dh = max(int(iw * self._scale), 1), max(int(ih * self._scale), 1)
        ox, oy = self._offset

        # Resample only the part that is actually on screen: a 4K photo at 8x
        # would otherwise build a 32K-pixel-wide bitmap to show a corner of it.
        vx1 = max(0, int(-ox / self._scale))
        vy1 = max(0, int(-oy / self._scale))
        vx2 = min(iw, int((cw - ox) / self._scale) + 1)
        vy2 = min(ih, int((ch - oy) / self._scale) + 1)
        if vx2 <= vx1 or vy2 <= vy1:
            self._draw_boxes(ox, oy)
            return

        crop = self._image.crop((vx1, vy1, vx2, vy2))
        target = (max(int((vx2 - vx1) * self._scale), 1), max(int((vy2 - vy1) * self._scale), 1))
        resample = Image.LANCZOS if self._scale < 1 else Image.NEAREST
        shown = crop.resize(target, resample)
        if self.heat_visible and self._heat is not None:
            # The map is tinted for the visible slice only, at screen size, so
            # zooming into a 4K photo never builds a full-resolution overlay.
            tint = heatmap.region_overlay(
                self._heat, self._mask,
                (vx1 / iw, vy1 / ih, vx2 / iw, vy2 / ih), target,
                style=self.heat_style, opacity=self.heat_opacity,
                show_mask=self.mask_visible, floor=self.heat_floor, limit=self._limit,
            )
            shown = Image.alpha_composite(shown.convert("RGBA"), tint)
        self._photo = ImageTk.PhotoImage(shown)
        canvas.create_image(ox + vx1 * self._scale, oy + vy1 * self._scale,
                            image=self._photo, anchor="nw")
        self._draw_boxes(ox, oy)

        canvas.create_text(
            8, ch - 8, anchor="sw", fill="#8b949e", font=("Segoe UI", 8),
            text=f"{iw}x{ih}  {self._scale * 100:.0f}%   wheel = zoom, drag = pan, double-click = fit",
        )
        _ = (dw, dh)

    def _draw_boxes(self, ox: float, oy: float) -> None:
        # Draw the selected box last so its label is never hidden under another.
        order = sorted(range(len(self._overlays)), key=lambda i: i == self._selected)
        for index in order:
            ov = self._overlays[index]
            x1, y1, x2, y2 = ov.box
            selected = index == self._selected
            color = SELECTED_COLOR if selected else ov.color
            left, top = ox + x1 * self._scale, oy + y1 * self._scale
            right, bottom = ox + x2 * self._scale, oy + y2 * self._scale
            self.canvas.create_rectangle(
                left, top, right, bottom,
                outline=color, width=3 if selected else 2,
                dash=(5, 3) if ov.dashed else None,
            )
            # Labels are only readable when the box has room for them; at
            # fit-the-whole-photo zoom they would pile up on top of each other.
            if ov.label and (selected or right - left > 70):
                text = self.canvas.create_text(
                    left + 3, top - 3, anchor="sw", text=ov.label, fill=color,
                    font=("Segoe UI", 9, "bold" if selected else "normal"),
                )
                box = self.canvas.bbox(text)
                if box:
                    shade = self.canvas.create_rectangle(
                        box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1,
                        fill="#0d1117", outline="",
                    )
                    self.canvas.tag_lower(shade, text)


class HeatLegend(ttk.Frame):
    """The heat scale, labelled -- a colour on the photo read back to a number.

    A heat ramp without this is decoration: the reader can see one patch is
    angrier than another but not whether either crossed the threshold.
    """

    WIDTH = 168
    HEIGHT = 10

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, width=self.WIDTH, height=self.HEIGHT + 13,
                                highlightthickness=0, background="#f0f0f0")
        self.canvas.pack()
        self._strip: ImageTk.PhotoImage | None = None
        self.draw(heatmap.FLOOR)

    def draw(self, floor: float, style: str = "threshold") -> None:
        canvas = self.canvas
        canvas.delete("all")
        # JET paints the whole range, so its legend starts at 0, not at a floor.
        strip = (heatmap.jet_strip(self.WIDTH, self.HEIGHT) if style == "jet"
                 else heatmap.scale_strip(self.WIDTH, self.HEIGHT, floor))
        # The strip is part-transparent by design, so flatten it onto the
        # widget's own background rather than letting Tk guess.
        plate = Image.new("RGBA", strip.size, (240, 240, 240, 255))
        self._strip = ImageTk.PhotoImage(Image.alpha_composite(plate, strip))
        canvas.create_image(0, 0, image=self._strip, anchor="nw")

        # One tick that matters: where anomalib's calibrated threshold sits.
        x = int(heatmap.THRESHOLD * self.WIDTH)
        canvas.create_line(x, 0, x, self.HEIGHT, fill="#0b0b0b", width=1)
        canvas.create_text(2, self.HEIGHT + 1, anchor="nw",
                           text="0.00" if style == "jet" else f"{floor:.2f}",
                           fill="#898781", font=("Segoe UI", 7))
        canvas.create_text(x, self.HEIGHT + 1, anchor="n", text="0.50",
                           fill="#52514e", font=("Segoe UI", 7))
        canvas.create_text(self.WIDTH, self.HEIGHT + 1, anchor="ne", text="1.00",
                           fill="#898781", font=("Segoe UI", 7))
