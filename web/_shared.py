"""Shared helpers for the Streamlit pages of the interactive inference app.

Streamlit reads this module from ``web/Home.py`` and ``web/pages/*.py``; it keeps
the pages thin and free of pipeline details (those live in ``src/web_inference.py``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for _path in (str(REPO_ROOT), str(SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data import project_path  # noqa: E402
from infer_new_case import load_diagnostic_from_checkpoint  # noqa: E402
from train_diagnostic import get_device  # noqa: E402
from web_inference import InferenceContext, load_config_and_context  # noqa: E402

CONFIG_ENV_VAR = "TEIH_CONFIG"
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
REPORT_FIGURE_DIR = REPO_ROOT / "INFO" / "report" / "figure"

PAGE_TITLE = "ToErrIsHuman — inferenza interattiva"
PAGE_ICON = "🧠"

# Numbers of the documented run (INFO/report_consegna.md, PDF "Rapporto").
# Used as fixed reference values in the explanatory copy; where an artefact is
# available the pages read the live number instead.
DOCUMENTED_RUN = {
    "n_cases": 427,
    "n_decisions": 5551,
    "n_raters": 13,
    "error_prevalence": 0.1904,
    "majority_accuracy": 0.8096,
    "diagnostic_auroc": 0.77,
    "diagnostic_auprc": 0.73,
    "diagnostic_accuracy_0_5": 0.710,
    "q_ir_auroc": 0.7159,
    "q_ir_auprc": 0.4158,
    "q_ir_brier": 0.2101,
    "q_ir_bss": -0.3629,
    "q_ir_mean_probability": 0.3622,
    "d_auroc": 0.6327,
    "d_auprc": 0.3247,
    "d_brier": 0.1520,
    "d_ece": 0.0653,
    "d_bss": 0.0139,
    "d_mean_probability": 0.1582,
    "d_recall_0_5": 0.1277,
    "d_precision_0_5": 0.4945,
    "d_unweighted_auroc": 0.6516,
    "d_unweighted_brier": 0.1577,
    "d_fast_auroc": 0.7028,
}

CALIBRATION_WARNING = (
    "Le probabilità non sono percentuali cliniche: nel run documentato D è "
    f"marginalmente migliore di una costante (BSS {DOCUMENTED_RUN['d_bss']:.3f}) e "
    f"predice in media {DOCUMENTED_RUN['d_mean_probability']:.3f} contro una frequenza "
    f"osservata di errore di {DOCUMENTED_RUN['error_prevalence']:.4f}. La discriminazione "
    f"di D (AUROC {DOCUMENTED_RUN['d_auroc']:.3f}) resta inferiore a quella della baseline "
    f"q_ir (AUROC {DOCUMENTED_RUN['q_ir_auroc']:.3f}): il guadagno è sulla qualità "
    "probabilistica, non sul ranking."
)


def setup_page(page_title: str, *, layout: str = "wide") -> None:
    """``st.set_page_config`` wrapper with the project defaults."""
    st.set_page_config(
        page_title=f"{page_title} · ToErrIsHuman",
        page_icon=PAGE_ICON,
        layout=layout,
        initial_sidebar_state="expanded",
    )


def reset_import_path() -> None:
    """Make ``src`` importable when a page runs from a different working dir."""
    for path in (str(REPO_ROOT), str(SRC_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


def config_path() -> Path:
    """Config file used by the app (``TEIH_CONFIG`` env var can override it)."""
    return Path(os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG)).expanduser()


@st.cache_data(show_spinner=False)
def cached_context(config_file: str) -> InferenceContext:
    """Cached config + data tables + fold assignment (fast, no model loads)."""
    _cfg, ctx = load_config_and_context(Path(config_file))
    return ctx


def get_context() -> InferenceContext:
    return cached_context(str(config_path()))


@st.cache_resource(show_spinner=False)
def _diagnostic_model(checkpoint: str, device_spec: str, mtime: float):
    """One cached DiagnosticNet per fold (invalidated when the file changes)."""
    del mtime  # part of the cache key only
    return load_diagnostic_from_checkpoint(Path(checkpoint), get_device(device_spec))


@st.cache_resource(show_spinner=False)
def _d_assets(preprocessor: str, model: str, pre_mtime: float, model_mtime: float):
    """One cached (preprocessor, D payload) pair per fold."""
    import joblib

    del pre_mtime, model_mtime  # part of the cache key only
    return joblib.load(preprocessor), joblib.load(model)


def model_provider(device_spec: str):
    """Cached diagnostic loader compatible with ``web_inference.run_inference``."""

    def provider(fold: int, checkpoint: Path):
        return _diagnostic_model(str(checkpoint), device_spec, checkpoint.stat().st_mtime)

    return provider


def d_asset_provider():
    """Cached D loader compatible with ``web_inference.run_inference``."""

    def provider(fold: int, preprocessor: Path, model: Path):
        return _d_assets(
            str(preprocessor),
            str(model),
            preprocessor.stat().st_mtime,
            model.stat().st_mtime,
        )

    return provider


def clear_caches() -> None:
    cached_context.clear()
    _diagnostic_model.clear()
    _d_assets.clear()


# --------------------------------------------------------------------- paths
def outputs_dir(ctx: InferenceContext) -> Path:
    return project_path(ctx.cfg["outputs"]["root_dir"], ctx.base_dir)


def output_subdir(ctx: InferenceContext, key: str) -> Path:
    return project_path(ctx.cfg["outputs"][key], ctx.base_dir)


def models_dir(ctx: InferenceContext) -> Path:
    return project_path(ctx.cfg["models"]["diagnostic_dir"], ctx.base_dir).parent


# ---------------------------------------------------------------- artefacts
@st.cache_data(show_spinner=False)
def _cached_table(path: str, mtime: float) -> Optional[pd.DataFrame]:
    del mtime  # cache key only
    file = Path(path)
    if not file.is_file():
        return None
    return pd.read_csv(file)


@st.cache_data(show_spinner=False)
def _cached_json(path: str, mtime: float):
    del mtime  # cache key only
    file = Path(path)
    if not file.is_file():
        return None
    return json.loads(file.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _cached_bytes(path: str, mtime: float) -> Optional[bytes]:
    del mtime  # cache key only
    file = Path(path)
    return file.read_bytes() if file.is_file() else None


def table(path: Path) -> Optional[pd.DataFrame]:
    """CSV artefact as DataFrame, re-read only when the file changes."""
    path = Path(path)
    return _cached_table(str(path), path.stat().st_mtime) if path.is_file() else None


def payload(path: Path):
    """JSON artefact (dict or list), re-read only when the file changes."""
    path = Path(path)
    return _cached_json(str(path), path.stat().st_mtime) if path.is_file() else None


def image_bytes(path: Path) -> Optional[bytes]:
    path = Path(path)
    return _cached_bytes(str(path), path.stat().st_mtime) if path.is_file() else None


def missing_notice(path: Path, command: str, label: Optional[str] = None) -> None:
    """Consistent message when a pipeline artefact has not been produced yet."""
    st.warning(
        f"Artefatto non disponibile: `{path}`"
        + (f" ({label})" if label else "")
        + f"\n\nGeneralo con:\n\n```bash\n{command}\n```"
    )


# ---------------------------------------------------------------- formatting
def fmt_prob(value, digits: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "n.d."
    except (TypeError, ValueError):
        return "n.d."
    return f"{float(value):.{digits}f}"


def fmt_signed(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n.d."
    return f"{float(value):+.{digits}f}"


def decision_label(probability, threshold: float = 0.5) -> str:
    if probability is None or pd.isna(probability):
        return "n.d."
    return "errore probabile" if float(probability) >= threshold else "errore improbabile"


def fold_list(ctx: InferenceContext) -> list:
    return list(range(1, ctx.n_folds + 1))


# ------------------------------------------------------------------- layout
def sidebar_status(ctx: InferenceContext) -> None:
    """Artifact/environment panel shared by every page."""
    from web_inference import artifact_status, environment_info

    status = artifact_status(ctx)
    env = environment_info(ctx)

    with st.sidebar:
        st.markdown("### Stato del run")
        if status["all_ok"]:
            st.success("Modelli e dati presenti (5 fold completi)")
        else:
            st.error(f"{len(status['missing'])} artefatti mancanti")
            with st.expander("Dettaglio"):
                for item in status["missing"]:
                    st.code(item, language="text")
                st.markdown(
                    "Rigenera la pipeline con:\n\n```bash\n"
                    "python -u src/generate_folds.py --config config.yaml\n"
                    "python -u src/train_diagnostic.py --config config.yaml\n"
                    "python -u src/train_D.py --config config.yaml --protocol nested\n```"
                )
        st.caption(
            f"device **{env['device']}** · torch {env['torch']} · "
            f"CUDA {'sì' if env['cuda_available'] else 'no'}"
        )
        st.caption(
            f"{env['n_cases']} casi · {env['n_decisions']} decisioni · "
            f"{env['n_folds']} fold · protocollo D **{env['protocollo_D']}** · seed {env['seed']}"
        )
        st.caption(f"config: `{config_path()}`")
        if st.button("Ricarica cache artefatti", use_container_width=True):
            clear_caches()
            st.rerun()


def figure_gallery(
    paths: Sequence[Path],
    captions: Optional[Sequence[str]] = None,
    *,
    columns: int = 2,
) -> None:
    """Show existing PNG artefacts, skipping (and reporting) missing ones."""
    found = []
    for index, path in enumerate(paths):
        data = image_bytes(Path(path))
        caption = captions[index] if captions and index < len(captions) else Path(path).name
        if data is not None:
            found.append((data, caption))
    if not found:
        st.info("Nessuna figura disponibile fra gli artefatti richiesti.")
        return
    for start in range(0, len(found), columns):
        cols = st.columns(columns)
        for col, (data, caption) in zip(cols, found[start : start + columns]):
            with col:
                st.image(data, caption=caption, use_container_width=True)
