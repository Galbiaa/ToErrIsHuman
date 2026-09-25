"""Step 1 verification: predict_proba[:, 1] must correspond to error-rating = 1.

For a binary XGBoost classifier, predict_proba columns follow clf.classes_
order.  Labels are {0, 1}, so classes_ must be exactly [0, 1]; column 1 is
then P(error-rating = 1), i.e. the "error" probability.

This script:
- loads every saved D model (weighted + unweighted) WITHOUT retraining;
- asserts n_classes_ == 2 and list(classes_) == [0, 1];
- sanity-checks that predict_proba rows sum to 1 and argmax matches predict().

Writes a per-file report to outputs/D/predict_proba_class_verification.csv
and raises if any model violates the required class order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# the script lives in scratch/ but imports modules from ../src
_SRC_SIBLING = _SRC.parent / "src"
if _SRC_SIBLING.is_dir() and str(_SRC_SIBLING) not in sys.path:
    sys.path.insert(0, str(_SRC_SIBLING))

from build_D_features import d_feature_allowlist
from data import load_config, project_path
from reproducibility import bootstrap_run_reproducibility, set_global_seed

EXPECTED_CLASSES = [0, 1]


def check_model_file(path: Path) -> dict:
    """Verify class order for one saved D model dict."""
    payload = joblib.load(path)
    if "model" not in payload:
        raise ValueError(f"{path.name}: not a D model dict (missing 'model' key)")

    clf = payload["model"]
    classes = list(clf.classes_)
    n_classes = int(clf.n_classes_)
    ok = (n_classes == 2) and (classes == EXPECTED_CLASSES)

    return {
        "file": path.name,
        "run": str(payload.get("run_name", "")),
        "outer_fold": str(payload.get("outer_fold", "")),
        "protocol": str(payload.get("protocol", "")),
        "scale_pos_weight": str(payload.get("scale_pos_weight", "")),
        "n_classes": n_classes,
        "classes_": repr(classes),
        "class_order_ok": ok,
    }


def proba_row_checks(clf, X: np.ndarray, n_rows: int = 5):
    """Sanity checks on predict_proba output shape / summing / argmax."""
    if X is None or len(X) == 0:
        return "no-feature-input"
    probs = clf.predict_proba(X[:n_rows])
    preds = clf.predict(X[:n_rows])
    sums_ok = bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-6))
    argmax_ok = bool(np.array_equal(probs.argmax(axis=1), preds))
    return f"sums_to_1={sums_ok} argmax_matches_predict={argmax_ok}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify D predict_proba[:, 1] == P(error-rating=1) on saved models."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--smoke-features",
        default=None,
        help="Optional CSV with D features (e.g. feature_smoke_from_outer_oof.csv) "
        "for the sum-to-1 / argmax sanity check.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_dir = cfg_path.parent
    cfg = load_config(cfg_path)

    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    set_global_seed(seed)
    bootstrap_run_reproducibility(cfg, base_dir=base_dir)

    model_dir = project_path(cfg["models"]["d_dir"], base_dir)
    files = sorted(model_dir.glob("scenario_D_fold_*.joblib"))
    if not files:
        raise FileNotFoundError(f"No scenario_D_fold_*.joblib found in {model_dir}")

    # Optional feature input for probability-row sanity checks.
    X_probe = None
    if args.smoke_features:
        smoke_path = Path(args.smoke_features)
        if not smoke_path.is_file():
            smoke_path = project_path(args.smoke_features, base_dir)
        allow = d_feature_allowlist(cfg)
        feats = pd.read_csv(smoke_path)
        X_probe = feats.loc[:, allow].to_numpy(dtype=float)

    rows = []
    for path in files:
        info = check_model_file(path)
        payload = joblib.load(path)
        clf = payload["model"]
        if X_probe is not None:
            info["proba_sanity"] = proba_row_checks(clf, X_probe)
        rows.append(info)
        status = "OK" if info["class_order_ok"] else "FAIL"
        print(
            f"[{status}] {path.name}: classes_={info['classes_']} "
            f"n_classes={info['n_classes']}"
        )

    report = pd.DataFrame(rows)
    report_path = project_path(cfg["outputs"]["d_dir"], base_dir) / "predict_proba_class_verification.csv"
    report.to_csv(report_path, index=False)

    bad = report.loc[~report["class_order_ok"]]
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} D model(s) violated expected class order [0, 1]: "
            f"{bad['file'].tolist()}"
        )

    print(f"\nAll {len(files)} D models have classes_ == [0, 1]: predict_proba[:, 1] is P(error-rating=1).")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()