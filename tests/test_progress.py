"""Progress lines from a training child, and the bar the window makes of them."""

import io
import sys
import threading
import queue

from icinspect.progress import Reporter, format_progress, hms, parse_progress


def test_a_progress_line_round_trips():
    line = format_progress(120, 565, "epoch 3/20  loss 0.04")
    assert line == "[progress] 120/565 epoch 3/20  loss 0.04"
    assert parse_progress(line) == (120, 565, "epoch 3/20  loss 0.04")


def test_lightning_bars_are_understood_too():
    assert parse_progress("Fitting ━━━━━━━━━━━━━━━━━━━━ 12/44 0:00:03 • 0:00:08 3.9it/s") == (12, 44, "Fitting")
    assert parse_progress("Epoch 0/0 ━━━━━━━━━━━━━━━ 44/44 0:00:12 • 0:00:00") == (44, 44, "Epoch 0/0")
    assert parse_progress("Predicting ----------------------------------- 7/7 0:00:02") == (7, 7, "Predicting")


def test_ordinary_lines_are_not_progress():
    assert parse_progress("[ocr] epoch   3/20  loss 0.0421  (12.3s)") is None
    assert parse_progress("loaded 7/7 fonts") is None
    assert parse_progress("") is None


def test_hms():
    assert hms(0) == "0:00"
    assert hms(65) == "1:05"
    assert hms(3725) == "1:02:05"


def test_reporter_rate_limits_but_always_prints_the_last_step():
    out = io.StringIO()
    reporter = Reporter(total=1000, every=3600, stream=out)
    for done in range(1, 1001):
        reporter.step(done, f"step {done}")
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("[progress] 1/1000 step 1")
    assert lines[-1] == "[progress] 1000/1000 step 1000"
    assert len(lines) == 2, "everything between was too soon after the last one"


def test_reporter_adds_an_eta_once_it_can_tell():
    out = io.StringIO()
    reporter = Reporter(total=10, every=0, stream=out)
    reporter.started -= 10                 # ten seconds ago
    reporter.step(5, "half")
    assert "eta 0:10" in out.getvalue()


# -- the window's side: streaming a child's output ------------------------------


class Recorder:
    """A stand-in for `gui.runner.Progress` that keeps what it was told."""

    def __init__(self):
        self.events = []
        self.cancelled = False

    def log(self, text=""):
        self.events.append(("log", text))

    def status(self, text):
        self.events.append(("status", text))

    def step(self, done, total):
        self.events.append(("step", (done, total)))


def test_streaming_turns_progress_into_steps_and_keeps_real_lines():
    from icinspect.gui.runner import _stream

    child = io.StringIO(
        "[ocr] device=cuda\n"
        "[progress] 10/100 epoch 1/2  batch 10/50\n"
        "Fitting ━━━━ 3/44 0:00:01\rFitting ━━━━ 44/44 0:00:12\n"
        "[ocr] best model : crnn_best.pt\n"
        "tail without newline"
    )
    rec = Recorder()
    _stream(rec, child)
    assert rec.events == [
        ("log", "[ocr] device=cuda"),
        ("step", (10, 100)), ("status", "epoch 1/2  batch 10/50"),
        ("step", (3, 44)), ("status", "Fitting"),
        ("step", (44, 44)), ("status", "Fitting"),
        ("log", "Fitting ━━━━ 44/44 0:00:12"),
        ("log", "[ocr] best model : crnn_best.pt"),
        ("log", "tail without newline"),
    ]


def test_run_command_streams_a_real_child(tmp_path):
    from icinspect.gui.runner import Cancelled, Progress, run_command

    script = tmp_path / "child.py"
    script.write_text(
        "import sys\n"
        "print('[progress] 1/4 warming up')\n"
        "sys.stdout.write('Fitting ---- 2/4\\rFitting ---- 4/4\\n')\n"
        "print('done')\n",
        encoding="utf-8",
    )
    q = queue.Queue()
    prog = Progress(q, threading.Event())
    code = run_command(prog, [sys.executable, str(script)])
    assert code == 0
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert ("progress", (1, 4)) in events
    assert ("progress", (4, 4)) in events
    assert ("log", "done") in events
    assert not any(kind == "log" and text.startswith("[progress]") for kind, text in events)


def test_patchcore_progress_callback_reports_each_stage(capsys):
    """The callback on a tiny lightning run: one counter per stage, in order."""
    torch = __import__("pytest").importorskip("torch")
    lightning = __import__("pytest").importorskip("lightning")
    from torch.utils.data import DataLoader, TensorDataset

    from icinspect.anomaly.common import progress_callback

    class Tiny(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 1)

        def training_step(self, batch, _idx):
            x, y = batch
            return torch.nn.functional.mse_loss(self.layer(x), y)

        def validation_step(self, batch, _idx):
            pass

        def test_step(self, batch, _idx):
            pass

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    data = TensorDataset(torch.zeros(12, 2), torch.zeros(12, 1))
    loader = DataLoader(data, batch_size=4)
    trainer = lightning.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False,
                                enable_checkpointing=False, enable_progress_bar=False,
                                enable_model_summary=False, callbacks=[progress_callback()],
                                num_sanity_val_steps=0)
    trainer.fit(Tiny(), loader, loader)
    trainer.test(Tiny(), loader)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[progress]")]
    parsed = [parse_progress(ln) for ln in lines]
    assert (3, 3, "extracting features 3/3") in [(d, t, txt.split("  eta")[0]) for d, t, txt in parsed]
    assert any(txt.startswith("validating 3/3") for _, _, txt in parsed)
    assert any(txt.startswith("testing 3/3") for _, _, txt in parsed)
