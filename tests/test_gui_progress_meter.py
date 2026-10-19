"""The progress bar under Inspect and Train, and the count that feeds it.

The meter needs a display and skips without one; the batch counting and the
service's step sequence do not.
"""

from types import SimpleNamespace

import pytest

tk = pytest.importorskip("tkinter", reason="tkinter required")


@pytest.fixture(scope="module")
def root():
    try:
        window = tk.Tk()
    except tk.TclError as exc:                      # no display
        pytest.skip(f"no Tk display: {exc}")
    window.withdraw()
    yield window
    window.destroy()


def test_meter_pulses_then_counts_and_says_time_left(root):
    from icinspect.gui.widgets import ProgressMeter

    meter = ProgressMeter(root)
    meter.start()
    assert str(meter.bar.cget("mode")) == "indeterminate", "no count yet: it pulses"
    meter.advance((0, 40))
    meter._stage_started -= 10                      # ten more seconds for ten photos
    meter.advance((10, 40))
    assert str(meter.bar.cget("mode")) == "determinate"
    assert float(meter.bar.cget("value")) == 10 and float(meter.bar.cget("maximum")) == 40
    text = meter.label.cget("text")
    assert text.startswith("25%") and "0:30 left" in text
    assert meter.fraction == 0.25


def test_meter_restarts_its_rate_on_a_new_stage(root):
    from icinspect.gui.widgets import ProgressMeter

    meter = ProgressMeter(root)
    meter.start()
    meter.advance((40, 44))
    meter.advance((2, 10))                          # validation: a new counter
    assert meter.label.cget("text") == "20%", "no rate yet within the new stage"


def test_meter_finished_and_stopped(root):
    from icinspect.gui.widgets import ProgressMeter

    meter = ProgressMeter(root)
    meter.start()
    meter.advance((3, 10))
    meter.stop(finished=True)
    assert meter.label.cget("text").startswith("100%   in ")
    meter.start()
    meter.advance((3, 10))
    meter.stop()
    assert meter.label.cget("text") == "" and float(meter.bar.cget("value")) == 0


def test_meter_ignores_an_empty_total_and_clamps(root):
    from icinspect.gui.widgets import ProgressMeter

    meter = ProgressMeter(root)
    meter.start()
    meter.advance((0, 0))
    assert str(meter.bar.cget("mode")) == "indeterminate"
    meter.advance((12, 10))
    assert meter.label.cget("text") == "100%"


# -- the count behind it -----------------------------------------------------------


def test_batch_images_reads_paths_or_the_tensor():
    from icinspect.anomaly.common import batch_images

    assert batch_images(SimpleNamespace(image_path=["a.jpg", "b.jpg", "c.jpg"])) == 3
    assert batch_images({"image_path": ["a.jpg"]}) == 1
    assert batch_images(SimpleNamespace(image_path=None, image=SimpleNamespace(shape=(8, 3, 250, 250)))) == 8
    assert batch_images(object()) == 1


class _Recorder:
    def __init__(self):
        self.steps, self.statuses = [], []

    def step(self, done, total):
        self.steps.append((done, total))

    def status(self, text):
        self.statuses.append(text)

    def log(self, text=""):
        pass

    def partial(self, payload):
        pass

    def check(self):
        pass


def test_one_bar_covers_scoring_then_reading(monkeypatch, tmp_path):
    """Scoring steps 0..2n out of 3n (a photo scored counts double), reading
    continues 2n..3n: never backwards."""
    pytest.importorskip("torch", reason="the service imports torch")
    from icinspect.gui.service import Inspector, Options

    images = []
    for name in ("a.png", "b.png", "c.png"):
        path = tmp_path / name
        path.write_bytes(b"")
        images.append(path)

    inspector = Inspector()

    def fake_score(self, imgs, opts, prog, total=None, weight=1):
        for done in range(len(imgs) + 1):
            prog.step(done * weight, total)
        return {}

    monkeypatch.setattr(Inspector, "score_anomaly", fake_score)
    monkeypatch.setattr(Inspector, "part_cutoff", lambda self, opts=None: 0.5)
    monkeypatch.setattr(Inspector, "part_cutoff_fitted", lambda self, opts=None: False)
    prog = _Recorder()
    results = inspector.inspect(images, Options(check_anomaly=True, check_ocr=False,
                                                ignore_background=False), prog)
    assert len(results) == 3
    done = [d for d, _ in prog.steps]
    assert done == sorted(done) and prog.steps[-1] == (9, 9)
    assert {t for _, t in prog.steps} == {9}
    assert (6, 9) in prog.steps and (7, 9) in prog.steps, "reading picks up where scoring ended"
    assert any(s.startswith("Deciding 3/3") for s in prog.statuses)
