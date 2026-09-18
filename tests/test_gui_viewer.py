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


# -- drawing a box by hand -------------------------------------------------


def _event(x, y):
    from types import SimpleNamespace

    return SimpleNamespace(x=x, y=y)


@pytest.fixture
def drawing(view, photo):
    """The photo shown at 1:1 at the canvas origin, in drawing mode."""
    view.show(photo)
    view._scale, view._offset, view._fit = 1.0, (0.0, 0.0), False
    changes = []
    view.begin_box_edit(None, on_change=changes.append)
    return view, changes


def test_dragging_on_the_photo_draws_a_box(drawing):
    view, changes = drawing
    view._on_press(_event(5, 6))
    view._on_drag(_event(25, 16))
    view._on_release(_event(25, 16))
    assert view.edit_box == (5, 6, 25, 16)
    assert changes[-1] == (5, 6, 25, 16), "the caller hears about it on release"


def test_a_backwards_drag_still_makes_a_box(drawing):
    view, _ = drawing
    view._on_press(_event(25, 16))
    view._on_drag(_event(5, 6))
    view._on_release(_event(5, 6))
    assert view.edit_box == (5, 6, 25, 16)


def test_the_box_cannot_leave_the_photo(drawing):
    view, _ = drawing
    view._on_press(_event(30, 20))
    view._on_drag(_event(300, 200))                  # the photo is 40x30
    view._on_release(_event(300, 200))
    assert view.edit_box == (30, 20, 40, 30)


def test_dragging_a_corner_resizes_and_the_inside_moves(drawing):
    view, _ = drawing
    view.begin_box_edit((10, 10, 30, 20))
    view._on_press(_event(30, 20))                   # bottom-right corner
    view._on_drag(_event(34, 26))
    view._on_release(_event(34, 26))
    assert view.edit_box == (10, 10, 34, 26)

    view._on_press(_event(20, 18))                   # inside
    view._on_drag(_event(22, 16))
    view._on_release(_event(22, 16))
    assert view.edit_box == (12, 8, 36, 24)


def test_a_drag_outside_the_box_starts_a_new_one(drawing):
    view, _ = drawing
    view.begin_box_edit((0, 0, 8, 8))
    view._on_press(_event(20, 20))
    view._on_drag(_event(30, 25))
    view._on_release(_event(30, 25))
    assert view.edit_box == (20, 20, 30, 25)


def test_ending_the_edit_hands_the_box_back_and_restores_panning(drawing):
    view, _ = drawing
    view._on_press(_event(5, 6))
    view._on_drag(_event(25, 16))
    view._on_release(_event(25, 16))
    assert view.end_box_edit() == (5, 6, 25, 16)
    assert not view.editing
    before = view._offset
    view._on_press(_event(0, 0))
    view._on_drag(_event(10, 0))
    assert view._offset == (before[0] + 10, before[1]), "the left button pans again"


def test_the_box_does_not_follow_a_different_photo(drawing, tmp_path):
    view, _ = drawing
    view.begin_box_edit((10, 10, 30, 20))
    other = tmp_path / "other.png"
    Image.new("RGB", (80, 60), (200, 200, 200)).save(other)
    view.show(other)
    assert view.edit_box is None, "it was drawn in another photo's pixels"
