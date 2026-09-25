"""Inference service behind the Streamlit app (no web-framework imports here).

Same fold-aligned wiring as ``src/infer_new_case.py``:

    diagnostic_fold_k.pt                    -> p_i = P(TARGET = 1 | images)
    build_d_feature_frame                   -> fold-safe features (rater profiles from historical cases)
    preprocessor_fold_k + scenario_D_fold_k -> P(error-rating = 1)

The five D probabilities are averaged; components of the same fold stay paired.

Two ways to feed a case:

* ``dataset`` (demo): the case exists in IMAGE-GROUND-TRUTH.xlsx and
  USER-INFOS-CORRECTED.xlsx. Diagnostic fold k saw 4/5 of the cases during
  training, so this mode checks the wiring and is **not** a performance estimate.
* ``upload``: three MRI files come from the browser and one synthetic USER row is
  built in memory from the context entered in the UI. The queried case is removed
  from ``train_case_ids``, so no statistic computed on it can reach the features;
  the placeholder ``error-rating`` of the synthetic row only fills a column that
  the feature builder requires to exist and is never read.

TARGET is never an input of solution D: the allowlist is enforced by
``methodology_guards.assert_d_features`` inside ``build_d_feature_frame`` and
again by ``extract_xy`` right before the preprocessor.

This module never writes to ``outputs/`` or ``models/``: the web app must not
overwrite the artefacts of the documented run.
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from build_D_features import build_d_feature_frame, d_feature_allowlist, extract_xy
from data import (
    get_image_paths,
    load_config,
    load_diagnostic_case_table,
    load_user_infos,
    project_path,
)
from generate_folds import load_fold_assignment
from infer_new_case import load_diagnostic_from_checkpoint, predict_case_probability
from train_diagnostic import get_device

# Operating threshold used for the "hard" decision columns.
DEFAULT_THRESHOLD = 0.5
# Provenance tag for AI probabilities fed to D (see methodology_guards).
PROBABILITY_SOURCE = "nested_test"
# Placeholder written in the synthetic USER row of an uploaded case. Never read:
# the queried case is excluded from train_case_ids, so its labels cannot enter
# the fold-safe rater statistics, and it is never a feature of the model.
SYNTHETIC_ERROR_RATING = 0
# Prefix of the temporary directories this module is allowed to delete.
UPLOAD_DIR_PREFIX = "teih_web_"


@dataclass(frozen=True)
class DecisionInput:
    """Rater context of one case-rater decision: the inputs of solution D."""

    rater_id: int
    rating: int
    rating_confidence: float
    case_difficulty: float
    rater_expertise: float
    # Ground-truth label, available only for dataset cases and never used as a
    # feature (kept for the optional "wiring check" panel).
    error_rating: Optional[int] = None

    def as_user_row(self, case_id: int) -> dict:
        return {
            "id": f"{int(case_id)}-{int(self.rater_id)}",
            "case_id": int(case_id),
            "rater_id": int(self.rater_id),
            "rating-confidence": float(self.rating_confidence),
            "case-difficulty": float(self.case_difficulty),
            "rater-expertise": float(self.rater_expertise),
            "rater-accuracy": np.nan,
            "rater-confidence": np.nan,
            "rating": int(self.rating),
            "error-rating": int(
                SYNTHETIC_ERROR_RATING if self.error_rating is None else self.error_rating
            ),
        }


@dataclass
class InferenceContext:
    """Read-only view over config, data tables and fold assignment."""

    cfg: dict
    base_dir: Path
    cases: pd.DataFrame  # case_id, target, *_path  (from load_diagnostic_case_table)
    user: pd.DataFrame   # one row per case-rater decision, TARGET stripped
    folds: pd.DataFrame  # case_id, fold
    allow: List[str]     # solution D allowlist (11 features)

    # --- paths -----------------------------------------------------------
    @property
    def n_folds(self) -> int:
        return int(self.cfg["validation"]["n_splits"])

    @property
    def orientations(self) -> List[str]:
        return list(self.cfg["images"]["orientations"])

    @property
    def extension(self) -> str:
        return str(self.cfg["images"].get("extension", "jpg"))

    @property
    def image_dir(self) -> Path:
        return project_path(self.cfg["data"]["image_dir"], self.base_dir)

    @property
    def diagnostic_model_dir(self) -> Path:
        return project_path(self.cfg["models"]["diagnostic_dir"], self.base_dir)

    @property
    def d_model_dir(self) -> Path:
        return project_path(self.cfg["models"]["d_dir"], self.base_dir)

    @property
    def folds_dir(self) -> Path:
        return project_path(self.cfg["outputs"]["folds_dir"], self.base_dir)

    def diagnostic_path(self, fold: int) -> Path:
        return self.diagnostic_model_dir / f"diagnostic_fold_{int(fold)}.pt"

    def preprocessor_path(self, fold: int) -> Path:
        return self.d_model_dir / f"preprocessor_fold_{int(fold)}.joblib"

    def d_model_path(self, fold: int) -> Path:
        return self.d_model_dir / f"scenario_D_fold_{int(fold)}.joblib"

    # --- convenience -----------------------------------------------------
    @property
    def n_cases(self) -> int:
        return int(self.cases["case_id"].nunique())

    @property
    def n_decisions(self) -> int:
        return int(len(self.user))

    def fold_of(self, case_id: int) -> Optional[int]:
        hit = self.folds.loc[self.folds["case_id"].astype(int) == int(case_id), "fold"]
        return int(hit.iloc[0]) if len(hit) else None


def load_inference_context(cfg: dict, base_dir: str | Path) -> InferenceContext:
    """Load everything the inference path needs (no heavy artefacts)."""
    base_dir = Path(base_dir)
    user = load_user_infos(cfg, base_dir=base_dir)
    # TARGET must never travel with the decision table used by D.
    drop = [c for c in user.columns if str(c).upper() == "TARGET"]
    if drop:
        user = user.drop(columns=drop)
    folds = load_fold_assignment(
        project_path(cfg["outputs"]["folds_dir"], base_dir) / "case_fold_assignment.csv"
    )
    folds = folds.copy()
    folds["case_id"] = folds["case_id"].astype(int)
    folds["fold"] = folds["fold"].astype(int)
    return InferenceContext(
        cfg=cfg,
        base_dir=base_dir,
        cases=load_diagnostic_case_table(cfg, base_dir=base_dir),
        user=user,
        folds=folds,
        allow=d_feature_allowlist(cfg),
    )


def load_config_and_context(config_path: str | Path) -> Tuple[dict, InferenceContext]:
    """Convenience wrapper: config file -> (cfg, context)."""
    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    return cfg, load_inference_context(cfg, base_dir=cfg_path.parent)


def artifact_status(ctx: InferenceContext) -> dict:
    """Presence of every artefact inference depends on (for the UI status panel)."""
    folds: List[dict] = []
    missing: List[str] = []
    for fold in range(1, ctx.n_folds + 1):
        diag = ctx.diagnostic_path(fold)
        pre = ctx.preprocessor_path(fold)
        d_model = ctx.d_model_path(fold)
        folds.append(
            {
                "fold": fold,
                "diagnostic_checkpoint": diag,
                "preprocessor": pre,
                "d_model": d_model,
                "ok": all(p.is_file() for p in (diag, pre, d_model)),
            }
        )
        for label, path in (
            ("checkpoint diagnostico", diag),
            ("preprocessor D", pre),
            ("modello D", d_model),
        ):
            if not path.is_file():
                missing.append(f"{label} fold {fold}: {path}")

    user_path = project_path(ctx.cfg["data"]["user_infos_file"], ctx.base_dir)
    gt_path = project_path(ctx.cfg["data"]["ground_truth_file"], ctx.base_dir)
    fold_assignment = ctx.folds_dir / "case_fold_assignment.csv"
    data_files = {
        "USER-INFOS-CORRECTED.xlsx": (user_path, user_path.is_file()),
        "IMAGE-GROUND-TRUTH.xlsx": (gt_path, gt_path.is_file()),
        "case_fold_assignment.csv": (fold_assignment, fold_assignment.is_file()),
    }
    for label, (path, ok) in data_files.items():
        if not ok:
            missing.append(f"{label}: {path}")

    return {
        "folds": folds,
        "data_files": data_files,
        "missing": missing,
        "all_ok": not missing,
    }


def environment_info(ctx: InferenceContext) -> dict:
    """Small run/environment summary shown in the sidebar."""
    return {
        "device": str(get_device(ctx.cfg["diagnostic"].get("device", "auto"))),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "n_cases": ctx.n_cases,
        "n_decisions": ctx.n_decisions,
        "n_folds": ctx.n_folds,
        "protocollo_D": ctx.cfg["solution_d"].get("protocol"),
        "seed": ctx.cfg.get("reproducibility", {}).get("seed"),
    }


def case_catalog(ctx: InferenceContext) -> pd.DataFrame:
    """One row per case: fold, number of decisions, raters, MRI availability."""
    fold_map = dict(zip(ctx.folds["case_id"].astype(int), ctx.folds["fold"].astype(int)))
    target_map = dict(zip(ctx.cases["case_id"].astype(int), ctx.cases["target"].astype(int)))
    grouped = {int(cid): sub for cid, sub in ctx.user.groupby("case_id", sort=False)}
    rows: List[dict] = []
    for case_id in ctx.cases["case_id"].astype(int).tolist():
        sub = grouped.get(int(case_id))
        paths = get_image_paths(case_id, ctx.image_dir, ctx.orientations, ctx.extension)
        rows.append(
            {
                "case_id": int(case_id),
                "fold": fold_map.get(int(case_id)),
                "n_decisions": int(len(sub)) if sub is not None else 0,
                "rater_ids": sorted(int(x) for x in sub["rater_id"]) if sub is not None else [],
                "has_all_images": all(p.is_file() for p in paths),
                # Ground truth: used by the UI only as an explicit demo check.
                "target": target_map.get(int(case_id)),
            }
        )
    return pd.DataFrame(rows)


def decisions_of_case(ctx: InferenceContext, case_id: int) -> pd.DataFrame:
    """Decision rows of one case, with the context actually used as D inputs."""
    sub = ctx.user.loc[ctx.user["case_id"].astype(int) == int(case_id)].copy()
    cols = [
        "id",
        "rater_id",
        "rating",
        "rating-confidence",
        "case-difficulty",
        "rater-expertise",
        "error-rating",
    ]
    present = [c for c in cols if c in sub.columns]
    return sub[present].sort_values("rater_id").reset_index(drop=True)


def context_ranges(ctx: InferenceContext) -> Dict[str, Tuple[int, int]]:
    """Observed ranges of the four context inputs (fallback when unavailable)."""
    fallback = {
        "rating-confidence": (1, 5),
        "case-difficulty": (1, 4),
        "rater-expertise": (2, 4),
    }
    out: Dict[str, Tuple[int, int]] = {}
    for col, (lo, hi) in fallback.items():
        if col in ctx.user.columns:
            values = pd.to_numeric(ctx.user[col], errors="coerce").dropna()
            if len(values):
                out[col] = (int(values.min()), int(values.max()))
                continue
        out[col] = (lo, hi)
    return out


def suggest_new_case_id(ctx: InferenceContext) -> int:
    """First free case id after the ones already in the dataset."""
    known = set(ctx.cases["case_id"].astype(int)) | set(ctx.user["case_id"].astype(int))
    return int(max(known) + 1) if known else 1


def validate_new_case_id(ctx: InferenceContext, case_id: int) -> None:
    """An uploaded case must not collide with a case of the dataset."""
    case_id = int(case_id)
    in_gt = case_id in set(ctx.cases["case_id"].astype(int))
    in_user = case_id in set(ctx.user["case_id"].astype(int))
    if in_gt or in_user:
        raise ValueError(
            f"case_id={case_id} esiste già nel dataset (ground truth: {in_gt}, "
            f"decisioni rater: {in_user}). Usa un id libero oppure la modalità "
            "'caso nel dataset'."
        )


def save_uploaded_images(
    uploads: Dict[str, bytes | Path | str],
    case_id: int,
    *,
    orientations: Sequence[str],
    extension: str = "jpg",
    root: str | Path | None = None,
) -> Path:
    """Write browser uploads to a temp dir with the pipeline naming convention.

    Files are always re-encoded as RGB JPEG (``case{i}{orientation}.{extension}``)
    so uploaded previews follow the same preprocessing path as the dataset images.
    """
    from PIL import Image

    if root is None:
        root = Path(tempfile.mkdtemp(prefix=UPLOAD_DIR_PREFIX))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    missing = [o for o in orientations if uploads.get(o) is None]
    if missing:
        raise ValueError(f"Immagini mancanti per le viste: {missing}")

    for orientation in orientations:
        payload = uploads[orientation]
        source: io.BytesIO | Path
        if isinstance(payload, (bytes, bytearray)):
            source = io.BytesIO(bytes(payload))
        else:
            source = Path(payload)
        with Image.open(source) as img:
            img.convert("RGB").save(
                root / f"case{int(case_id)}{orientation}.{extension}",
                format="JPEG",
                quality=95,
            )
    return root


def cleanup_upload_dir(path: str | Path | None) -> None:
    """Delete a temp upload dir created by :func:`save_uploaded_images`."""
    if path is None:
        return
    path = Path(path)
    if not path.exists():
        return
    if not path.name.startswith(UPLOAD_DIR_PREFIX):
        # Safety: never delete outside the directories we created.
        raise ValueError(f"Refusing to delete untracked directory: {path}")
    shutil.rmtree(path, ignore_errors=True)


def _user_table_with_decisions(
    ctx: InferenceContext,
    case_id: int,
    decisions: Sequence[DecisionInput],
    *,
    is_new_case: bool,
) -> pd.DataFrame:
    """USER table used by the fold-safe feature builder.

    The rows of the queried case are replaced by the context entered in the UI, so
    the app can run transparent "what-if" variations. Everything else is the
    historical table and is the only source of the rater profiles.
    """
    if not decisions:
        raise ValueError("Nessuna decisione selezionata")
    new_rows = pd.DataFrame([d.as_user_row(case_id) for d in decisions])
    if new_rows["rater_id"].duplicated().any():
        raise ValueError("rater_id duplicato tra le decisioni selezionate")

    base = ctx.user.copy()
    if is_new_case:
        base = base.loc[base["case_id"].astype(int) != int(case_id)]
    else:
        # Keep the other decisions of this case (they stay out of train_case_ids
        # anyway, being part of the queried case).
        base = base.loc[~base["id"].isin(new_rows["id"])]
    return pd.concat([base, new_rows], ignore_index=True)


def _historical_decision_counts(ctx: InferenceContext, train_case_ids: Sequence[int]) -> pd.Series:
    """Number of historical decisions per rater (transparency column)."""
    mask = ctx.user["case_id"].astype(int).isin([int(c) for c in train_case_ids])
    return ctx.user.loc[mask].groupby("rater_id").size()


def run_inference(
    ctx: InferenceContext,
    *,
    case_id: int,
    decisions: Sequence[DecisionInput],
    image_dir: str | Path | None = None,
    is_new_case: bool = False,
    device_spec: str | None = None,
    on_fold: Optional[Callable[[int, int], None]] = None,
    model_provider: Optional[Callable[[int, Path], torch.nn.Module]] = None,
    d_asset_provider: Optional[Callable[[int, Path, Path], Tuple[object, dict]]] = None,
) -> dict:
    """Fold-aligned ensemble inference for one case (one or more raters).

    Parameters
    ----------
    ctx:
        Result of :func:`load_inference_context`.
    case_id:
        Case to score.
    decisions:
        Rater context rows to score (the D feature inputs).
    image_dir:
        Where the three MRI files live (default: the dataset image dir).
    is_new_case:
        ``True`` for uploaded cases: the queried id must not exist in the dataset
        (see :func:`validate_new_case_id`) and the ground truth is unknown.
    device_spec:
        ``auto`` / ``cpu`` / ``cuda``; default from ``config.yaml``.
    on_fold:
        Optional progress callback ``(fold, n_folds)``.
    model_provider / d_asset_provider:
        Optional cached loaders (the Streamlit app passes cached ones so repeated
        runs do not reload the 5 checkpoints and the 5 D models from disk).

    Returns
    -------
    dict with per-fold rows, per-rater ensemble summary, feature importances,
    fold-level AI probabilities and a JSON-ready payload (``result["json"]``).
    """
    cfg = ctx.cfg
    case_id = int(case_id)
    if is_new_case:
        validate_new_case_id(ctx, case_id)
    image_dir_path = Path(image_dir) if image_dir is not None else ctx.image_dir
    device_spec = device_spec if device_spec is not None else cfg["diagnostic"].get("device", "auto")
    device = get_device(device_spec)

    # Label placeholder: DiagnosticDataset needs a target column; it is only used
    # to build the batch and never reaches a feature matrix.
    target = 0
    hit = ctx.cases.loc[ctx.cases["case_id"].astype(int) == case_id, "target"]
    if len(hit):
        target = int(hit.iloc[0])
    case_row = pd.DataFrame({"case_id": [case_id], "target": [target]})

    user = _user_table_with_decisions(ctx, case_id, decisions, is_new_case=is_new_case)
    train_case_ids = [
        c for c in ctx.folds["case_id"].astype(int).tolist() if int(c) != case_id
    ]
    history_counts = _historical_decision_counts(ctx, train_case_ids)
    history_counts.index = [int(i) for i in history_counts.index]

    allow = list(ctx.allow)
    rater_ids = sorted(int(d.rater_id) for d in decisions)

    per_fold_rows: List[pd.DataFrame] = []
    case_probability_rows: List[dict] = []
    importance_series: List[pd.Series] = []
    model_meta: List[dict] = []

    for fold in range(1, ctx.n_folds + 1):
        diag_path = ctx.diagnostic_path(fold)
        pre_path = ctx.preprocessor_path(fold)
        d_path = ctx.d_model_path(fold)
        for path in (diag_path, pre_path, d_path):
            if not path.is_file():
                raise FileNotFoundError(f"Artefatto mancante per il fold {fold}: {path}")

        owned_model = model_provider is None
        model = (
            load_diagnostic_from_checkpoint(diag_path, device)
            if owned_model
            else model_provider(fold, diag_path)
        )
        try:
            p_i = predict_case_probability(model, case_row, cfg, image_dir_path, device)
        finally:
            if owned_model:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        probs = pd.DataFrame({"case_id": [case_id], "ai_probability_class_1": [p_i]})
        frame = build_d_feature_frame(
            user,
            probs,
            train_case_ids,
            config=cfg,
            probability_source=PROBABILITY_SOURCE,
            fold_col=None,
        )
        frame = frame.loc[frame["case_id"].astype(int) == case_id].copy()
        frame = frame.loc[frame["rater_id"].astype(int).isin(rater_ids)]
        if frame.empty:
            raise RuntimeError(
                "Nessuna riga di decisione per il caso/rater richiesti dopo il calcolo "
                "delle feature fold-safe."
            )

        X, _y, names = extract_xy(frame, cfg)
        if list(names) != allow:
            raise RuntimeError(f"Ordine feature inatteso: {list(names)} != {allow}")

        if d_asset_provider is None:
            preprocessor = joblib.load(pre_path)
            payload = joblib.load(d_path)
        else:
            preprocessor, payload = d_asset_provider(fold, pre_path, d_path)
        classifier = payload["model"]
        if list(getattr(classifier, "classes_", [0, 1])) != [0, 1]:
            raise RuntimeError(
                f"Ordine classi inatteso: {getattr(classifier, 'classes_', None)}; la "
                "colonna 1 di predict_proba non sarebbe P(error-rating=1)."
            )
        threshold_youden = float(payload.get("threshold_youden", DEFAULT_THRESHOLD))
        transformed = preprocessor.transform(np.asarray(X, dtype=float))
        probability = classifier.predict_proba(transformed)[:, 1]

        block = frame[["id", "case_id", "rater_id", "rating"]].copy()
        for column in allow:
            block[column] = frame[column].to_numpy()
        block["rater_historical_decisions"] = (
            block["rater_id"].astype(int).map(history_counts).astype("Int64")
        )
        block["fold"] = fold
        block["ai_probability_class_1"] = p_i
        block["q_ir"] = np.where(block["rating"].to_numpy(dtype=int) == 0, p_i, 1.0 - p_i)
        block["d_probability"] = probability
        block["threshold_youden"] = threshold_youden
        block["ai_predicted_class"] = int(p_i >= DEFAULT_THRESHOLD)
        block["decision_at_0_5"] = (block["d_probability"] >= DEFAULT_THRESHOLD).astype(int)
        block["decision_at_youden"] = (block["d_probability"] >= threshold_youden).astype(int)
        block["y_true_demo"] = (
            np.nan if is_new_case else frame["error-rating"].to_numpy(dtype=float)
        )
        per_fold_rows.append(block)

        case_probability_rows.append({"fold": fold, "p_i": float(p_i)})
        model_meta.append(
            {
                "fold": fold,
                "run_name": payload.get("run_name"),
                "protocol": payload.get("protocol"),
                "threshold_youden": threshold_youden,
                "scale_pos_weight": payload.get("scale_pos_weight"),
                "train_probability_source": payload.get("train_probability_source"),
                "test_probability_source": payload.get("test_probability_source"),
            }
        )
        importances = getattr(classifier, "feature_importances_", None)
        if importances is not None and len(importances) == len(allow):
            importance_series.append(pd.Series(importances, index=allow, name=f"fold_{fold}"))

        if on_fold is not None:
            on_fold(fold, ctx.n_folds)

    fold_frame = pd.concat(per_fold_rows, ignore_index=True)
    case_probabilities = pd.DataFrame(case_probability_rows)

    summary = (
        fold_frame.groupby("rater_id", as_index=False)
        .agg(
            rating=("rating", "first"),
            ai_probability_mean=("ai_probability_class_1", "mean"),
            ai_probability_std=("ai_probability_class_1", "std"),
            q_ir_mean=("q_ir", "mean"),
            d_probability_mean=("d_probability", "mean"),
            d_probability_std=("d_probability", "std"),
            d_probability_min=("d_probability", "min"),
            d_probability_max=("d_probability", "max"),
            threshold_youden_mean=("threshold_youden", "mean"),
            folds_flagging_0_5=("decision_at_0_5", "sum"),
            folds_flagging_youden=("decision_at_youden", "sum"),
            rater_accuracy_foldsafe=("rater-accuracy", "first"),
            rater_confidence_foldsafe=("rater-confidence", "first"),
            rater_historical_decisions=("rater_historical_decisions", "first"),
            y_true_demo=("y_true_demo", "first"),
        )
        .sort_values("d_probability_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary["ai_predicted_class"] = (summary["ai_probability_mean"] >= DEFAULT_THRESHOLD).astype(int)
    summary["decision_at_0_5"] = (summary["d_probability_mean"] >= DEFAULT_THRESHOLD).astype(int)
    summary["decision_at_youden_mean"] = (
        summary["d_probability_mean"] >= summary["threshold_youden_mean"]
    ).astype(int)
    summary["delta_vs_q_ir"] = summary["d_probability_mean"] - summary["q_ir_mean"]

    if importance_series:
        importance = pd.concat(importance_series, axis=1)
        importance["mean"] = importance.mean(axis=1)
        importance = (
            importance.sort_values("mean", ascending=False)
            .reset_index()
            .rename(columns={"index": "feature"})
        )
    else:
        importance = pd.DataFrame({"feature": allow, "mean": np.nan})

    notes = [
        "Ensemble dei 5 fold: le componenti dello stesso fold (diagnostico, preprocessor, "
        "modello D) restano abbinate e le 5 probabilità vengono mediate.",
        "I profili rater (rater-accuracy, rater-confidence) sono ricalcolati fold-safe sui "
        "casi storici, escludendo il caso in esame: nessuna statistica del caso da predire "
        "entra nelle feature.",
        "TARGET non è mai una feature di D: l'allowlist è imposta da methodology_guards.",
        "Su un caso già presente nel dataset il risultato non è una stima di prestazione: "
        "parte dei modelli diagnostici ha visto quel caso in training.",
    ]
    if is_new_case:
        notes.append(
            "Caso caricato: il TARGET diagnostico è sconosciuto e non viene mai richiesto; "
            "la probabilità d'errore mostrata è una stima del modello, non una verifica."
        )

    result = {
        "case_id": case_id,
        "is_new_case": bool(is_new_case),
        "mode": "upload" if is_new_case else "dataset",
        "image_dir": str(image_dir_path),
        "device": str(device),
        "n_folds": int(ctx.n_folds),
        "n_raters": int(len(rater_ids)),
        "feature_allowlist": allow,
        "probability_source": PROBABILITY_SOURCE,
        "case_fold_assignment": ctx.fold_of(case_id),
        "case_probabilities": case_probabilities,
        "p_i_mean": float(case_probabilities["p_i"].mean()),
        "p_i_std": float(case_probabilities["p_i"].std(ddof=0)),
        "fold_frame": fold_frame,
        "decision_summary": summary,
        "importance": importance,
        "model_meta": pd.DataFrame(model_meta),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }
    result["json"] = result_to_json(result)
    return result


def result_to_json(result: dict) -> dict:
    """JSON-ready payload (shape aligned with ``infer_new_case.py``)."""
    fold_frame: pd.DataFrame = result["fold_frame"]
    decisions: List[dict] = []
    for row in result["decision_summary"].to_dict(orient="records"):
        sub = fold_frame.loc[fold_frame["rater_id"].astype(int) == int(row["rater_id"])]
        sub = sub.sort_values("fold")
        decisions.append(
            {
                "rater_id": int(row["rater_id"]),
                "rating": int(row["rating"]),
                "ai_probability_per_fold": [float(x) for x in sub["ai_probability_class_1"]],
                "ai_probability_mean": float(row["ai_probability_mean"]),
                "q_ir_per_fold": [float(x) for x in sub["q_ir"]],
                "q_ir_mean": float(row["q_ir_mean"]),
                "d_probability_per_fold": [float(x) for x in sub["d_probability"]],
                "d_probability_ensemble": float(row["d_probability_mean"]),
                "d_probability_std": _maybe_float(row["d_probability_std"]),
                "threshold_youden_per_fold": [float(x) for x in sub["threshold_youden"]],
                "threshold_youden_mean": float(row["threshold_youden_mean"]),
                "folds_flagging_error_at_0_5": int(row["folds_flagging_0_5"]),
                "folds_flagging_error_at_youden": int(row["folds_flagging_youden"]),
                "rater_accuracy_foldsafe": _maybe_float(row["rater_accuracy_foldsafe"]),
                "rater_confidence_foldsafe": _maybe_float(row["rater_confidence_foldsafe"]),
                "rater_historical_decisions": _maybe_int(row["rater_historical_decisions"]),
            }
        )

    per_fold_details = []
    for row in fold_frame.to_dict(orient="records"):
        per_fold_details.append(
            {
                "fold": int(row["fold"]),
                "rater_id": int(row["rater_id"]),
                "rating": int(row["rating"]),
                "ai_probability_class_1": float(row["ai_probability_class_1"]),
                "q_ir": float(row["q_ir"]),
                "d_probability": float(row["d_probability"]),
                "threshold_youden": float(row["threshold_youden"]),
                "features": {
                    name: _maybe_float(row[name]) for name in result["feature_allowlist"]
                },
            }
        )

    return {
        "case_id": int(result["case_id"]),
        "mode": result["mode"],
        "n_folds": int(result["n_folds"]),
        "device": result["device"],
        "image_dir": result["image_dir"],
        "probability_source": result["probability_source"],
        "feature_allowlist": list(result["feature_allowlist"]),
        "case_fold_assignment": result["case_fold_assignment"],
        "ai_probability_per_fold": [
            {"fold": int(r["fold"]), "p_i": float(r["p_i"])}
            for r in result["case_probabilities"].to_dict(orient="records")
        ],
        "ai_probability_mean": float(result["p_i_mean"]),
        "mean_d_probability_over_decisions": float(
            result["decision_summary"]["d_probability_mean"].mean()
        ),
        "decisions": decisions,
        "per_fold_details": per_fold_details,
        "notes": list(result["notes"]),
        "generated_at": result["generated_at"],
    }


def _maybe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def _maybe_int(value) -> Optional[int]:
    number = _maybe_float(value)
    return None if number is None else int(number)
