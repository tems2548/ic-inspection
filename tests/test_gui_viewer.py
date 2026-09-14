"""The photo canvas, on a real (hidden) Tk window.

These need a display, so they skip where there is none. They are here because
of one bug: switching to the anomalib panel grid and back left the grid on
screen. `show()` treated "same path" as "same picture already loaded", but the
grid and the capture it is built from share a path.
"""

from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image", reason="pillow required")
pytest.importorskip("PIL.ImageTk", reason="pillow's Tk bridge required")
tk = pytest.importorskip("tkinter", reason="tkinter required")


@pytest.fixture(scope="module")
def root():
    """One hidden window for the whole module.

    Creating and destroying a Tk root per test is flaky -- it intermittently
    raises TclError, which showed up here as tests skipping at random.
    """
    try:
        window = tk.Tk()
    except tk.TclError as exc:                      # no display
        pytest.skip(f"no Tk display: {exc}")
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def view(root):
    """A fresh ImageView on the shared window, gone again after the test."""
    from icinspect.gui.viewer import ImageView

    widget = ImageView(root)
    widget.pack()
    root.update()
    yield widget
    widget.destroy()


@pytest.fixture
def photo(tmp_path) -> Path:
    path = tmp_path / "chip.png"
    Image.new("RGB", (40, 30), (90, 90, 90)).save(path)
    return path


def test_switching_away_from_the_panel_grid_shows_the_photo_again(view, photo):
    """The bug: any view after 'anomalib panels' kept showing the panels."""
    view.show(photo)
    assert view._image.size == (40, 30)

    grid = Image.new("RGB", (400, 30), (10, 10, 10))
    view.show_composite(grid, key=str(photo))
    assert view._image.size == (400, 30), "the grid is on screen"

    view.show(photo)
    assert view._image.size == (40, 30), "switching back re-reads the capture"


def test_redrawing_the_same_photo_keeps_the_zoom(view, photo):
    view.show(photo)
    view.zoom_to((0, 0, 10, 10))
    zoomed = view._scale
    view.show(photo, overlays=[])
    assert view._scale == zoomed, "toggling boxes must not throw the zoom away"


def test_a_different_photo_snaps_back_to_fit(view, photo, tmp_path):
    other = tmp_path / "other.png"
    Image.new("RGB", (40, 30), (200, 200, 200)).save(other)
    view.show(photo)
    view.zoom_to((0, 0, 10, 10))
    view.show(other)
    assert view._fit, "a new photo is fitted, not left at the last one's zoom"


def test_redrawing_the_same_grid_keeps_the_zoom(view, photo):
    grid = Image.new("RGB", (400, 30), (10, 10, 10))
    view.show_composite(grid, key=str(photo))
    view.zoom_to((0, 0, 40, 30))
    zoomed = view._scale
    view.show_composite(grid, key=str(photo))
    assert view._scale == zoomed


def test_clearing_the_view_forgets_what_was_loaded(view, photo):
    view.show(photo)
    view.show(None)
    assert view._image is None
    view.show(photo)
    assert view._image is not None, "the same photo can be shown again after clearing"


def test_an_unreadable_file_does_not_wedge_the_view(view, tmp_path, photo):
    broken = tmp_path / "truncated.png"
    broken.write_bytes(b"not really a png")
    view.show(broken)
    assert view._image is None, "a bad capture clears rather than crashing"
    view.show(photo)
    assert view._image is not None, "and the next good one still loads"
