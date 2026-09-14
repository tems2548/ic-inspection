"""Fit PatchCore on defect-free IC images and evaluate on the test split.

    python -m icinspect.anomaly.train_patchcore
    python -m icinspect.anomaly.train_patchcore --config patchcore.yaml --no-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from icinspect.anomaly.common import build_datamodule, build_engine, build_model
from icinspect.config import load_yaml, resolve_output, resolve_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train PatchCore on the IC dataset.")
    ap.add_argument("--config", default="patchcore.yaml", help="config file (configs/ or absolute)")
    ap.add_argument("--paths", default="paths.yaml")
    ap.add_argument("--no-test", action="store_true", help="skip evaluation after fitting")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_yaml(args.config)
    paths = resolve_paths(args.paths)
    paths.require("anomaly_root")

    out_dir = resolve_output(cfg.get("trainer", {}).get("default_root_dir", "results/patchcore"))
    out_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(cfg, paths)
    model = build_model(cfg)
    engine = build_engine(cfg, out_dir)

    print(f"[patchcore] dataset : {paths.anomaly_root}")
    print(f"[patchcore] results : {out_dir}")
    engine.fit(model=model, datamodule=datamodule)

    ckpt = getattr(getattr(engine.trainer, "checkpoint_callback", None), "best_model_path", "")
    summary: dict[str, object] = {"checkpoint": ckpt, "dataset_root": str(paths.anomaly_root)}

    if not args.no_test:
        try:
            results = engine.test(model=model, datamodule=datamodule)
            summary["test"] = results
            for row in results or []:
                for k, v in row.items():
                    print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")
        except Exception as exc:  # a missing/empty test split should not lose the model
            print(f"[patchcore] evaluation skipped: {exc}")
            summary["test_error"] = str(exc)

    summary_path = out_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[patchcore] checkpoint: {ckpt or '(see ' + str(out_dir) + ')'}")
    print(f"[patchcore] summary   : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
