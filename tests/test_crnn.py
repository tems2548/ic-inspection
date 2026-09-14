"""CRNN shape contract and a CTC overfit smoke test. Skipped without torch."""

import pytest

torch = pytest.importorskip("torch", reason="torch required for model tests")

from icinspect.ocr.charset import Charset  # noqa: E402
from icinspect.ocr.model import CRNN, build_model  # noqa: E402

SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-+./ "


@pytest.fixture(scope="module")
def charset() -> Charset:
    return Charset(SYMBOLS)


def test_forward_shape_is_time_major(charset):
    model = CRNN(num_classes=charset.num_classes, rnn_hidden=32, rnn_layers=1)
    out = model(torch.zeros(2, 1, 32, 160))
    assert out.shape == (39, 2, charset.num_classes)   # T = 160//4 - 1


@pytest.mark.parametrize(("width", "expected_t"), [(64, 15), (128, 31), (160, 39), (256, 63)])
def test_output_length_matches_forward(width, expected_t, charset):
    model = CRNN(num_classes=charset.num_classes, rnn_hidden=16, rnn_layers=1)
    assert model.output_length(width) == expected_t
    assert model(torch.zeros(1, 1, 32, width)).shape[0] == expected_t


def test_non_32_height_is_rejected(charset):
    with pytest.raises(ValueError, match="img_height=32"):
        CRNN(num_classes=charset.num_classes, img_height=64)


def test_build_model_honours_config(charset):
    cfg = {
        "image": {"height": 32, "width": 160, "grayscale": True},
        "model": {"rnn_hidden": 48, "rnn_layers": 1, "dropout": 0.0},
    }
    model = build_model(cfg, charset.num_classes)
    assert model.fc.out_features == charset.num_classes
    assert len(model.rnn) == 1                 # stacked as single-layer LSTMs, see model.py
    assert model.rnn[0].hidden_size == 48


def test_ctc_loss_is_finite_and_backprops(charset):
    torch.manual_seed(0)
    model = CRNN(num_classes=charset.num_classes, rnn_hidden=32, rnn_layers=1)
    images = torch.randn(3, 1, 32, 160)
    targets = torch.cat([torch.tensor(charset.encode(t)) for t in ("STM32", "LM358N", "7412")])
    target_lengths = torch.tensor([5, 6, 4])

    log_probs = model(images).log_softmax(2)
    input_lengths = torch.full((3,), log_probs.size(0), dtype=torch.long)
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(log_probs, targets, input_lengths, target_lengths)

    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.slow
def test_overfits_a_single_batch(charset):
    """The training signal is wired correctly if one batch can be memorised."""
    torch.manual_seed(0)
    model = CRNN(num_classes=charset.num_classes, rnn_hidden=64, rnn_layers=1, dropout=0.0)
    texts = ["STM32F4", "LM358N"]

    # Distinct, learnable inputs: a fixed random image per label.
    images = torch.stack([torch.randn(1, 32, 160) for _ in texts])
    targets = torch.cat([torch.tensor(charset.encode(t)) for t in texts])
    target_lengths = torch.tensor([len(t) for t in texts])

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(700):
        optimizer.zero_grad()
        log_probs = model(images).log_softmax(2)
        input_lengths = torch.full((len(texts),), log_probs.size(0), dtype=torch.long)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        best = model(images).argmax(dim=2).permute(1, 0).tolist()
    decoded = [charset.ctc_greedy_decode(frames) for frames in best]
    assert decoded == texts, f"expected {texts}, decoded {decoded} (loss {loss.item():.4f})"


def test_lstm_stack_avoids_cudnn_dropout_state(charset):
    """Multi-layer cuDNN LSTMs with dropout>0 crash Windows/CUDA teardown (0xC0000409)."""
    model = CRNN(num_classes=charset.num_classes, rnn_hidden=32, rnn_layers=3, dropout=0.5)
    assert len(model.rnn) == 3
    for layer in model.rnn:
        assert layer.num_layers == 1
        assert layer.dropout == 0.0
    assert model(torch.zeros(1, 1, 32, 160)).shape == (39, 1, charset.num_classes)
