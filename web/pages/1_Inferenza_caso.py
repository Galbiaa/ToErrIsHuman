"""Inferenza su un caso: pagina principale dell'app Streamlit.

Flusso (identico a ``src/infer_new_case.py``, via ``src/web_inference.py``):

    checkpoint diagnostico fold k -> p_i
    feature fold-safe (profili rater sui casi storici) -> preprocessor k + D k -> P(errore)
    media delle 5 probabilità, componenti dello stesso fold sempre appaiate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd
import streamlit as st

import _shared as ui
from data import get_image_paths
from web_inference import (
    DEFAULT_THRESHOLD,
    DecisionInput,
    case_catalog,
    cleanup_upload_dir,
    context_ranges,
    decisions_of_case,
    run_inference,
    save_uploaded_images,
    suggest_new_case_id,
)

MODE_DATASET = "Caso nel dataset (demo)"
MODE_UPLOAD = "Nuovo caso (upload MRI)"
ALL_RATERS = "tutti i rater del caso"
UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
RESULT_KEY = "inferenza_ultimo_risultato"

CONTEXT_FIELDS = [
    ("rating-confidence", "rating-confidence", "Confidenza dichiarata dal rater"),
    ("case-difficulty", "case-difficulty", "Difficoltà percepita del caso"),
    ("rater-expertise", "rater-expertise", "Expertise dichiarata del rater"),
]


def _context_editor(rows: pd.DataFrame, ranges, key: str, *, single: bool) -> pd.DataFrame:
    """Editable rater-context table (what-if inputs of solution D)."""
    confidence = ranges["rating-confidence"]
    difficulty = ranges["case-difficulty"]
    expertise = ranges["rater-expertise"]
    editor = st.data_editor(
        rows,
        key=key,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "rater_id": st.column_config.NumberColumn("rater", disabled=True, format="%d"),
            "rating": st.column_config.SelectboxColumn(
                "rating (0/1)",
                options=[0, 1],
                required=True,
                help="Diagnosi del rater: 0 = negativa, 1 = positiva. È un input di D.",
            ),
            "rating-confidence": st.column_config.NumberColumn(
                "rating-confidence", min_value=confidence[0], max_value=confidence[1],
                step=1, format="%d",
            ),
            "case-difficulty": st.column_config.NumberColumn(
                "case-difficulty", min_value=difficulty[0], max_value=difficulty[1],
                step=1, format="%d",
            ),
            "rater-expertise": st.column_config.NumberColumn(
                "rater-expertise", min_value=expertise[0], max_value=expertise[1],
                step=1, format="%d",
            ),
        },
    )
    if single:
        st.caption(
            "Valori precompilati dai dati Excel: puoi modificarli per una prova "
            "what-if. Il profilo rater usato dal modello resta quello fold-safe."
        )
    return editor


def _decisions_from_frame(frame: pd.DataFrame) -> list:
    return [
        DecisionInput(
            rater_id=int(row["rater_id"]),
            rating=int(row["rating"]),
            rating_confidence=float(row["rating-confidence"]),
            case_difficulty=float(row["case-difficulty"]),
            rater_expertise=float(row["rater-expertise"]),
        )
        for _, row in frame.iterrows()
    ]


def dataset_inputs(ctx):
    """Case/rater selection for a case already present in the dataset."""
    catalog = case_catalog(ctx)
    labels = [
        f"caso {int(row.case_id)} — fold {int(row.fold)} · {int(row.n_decisions)} decisioni"
        + ("" if row.has_all_images else " · MRI mancanti")
        for row in catalog.itertuples()
    ]
    selected_label = st.selectbox("Caso", labels, index=0)
    case_id = int(catalog.iloc[labels.index(selected_label)]["case_id"])

    subset = decisions_of_case(ctx, case_id)
    rater_choice = st.selectbox(
        "Rater", [ALL_RATERS] + [f"rater {int(r)}" for r in subset["rater_id"]], index=0
    )
    single = rater_choice != ALL_RATERS
    if single:
        subset = subset.loc[subset["rater_id"].astype(int) == int(rater_choice.split()[-1])]

    rows = subset[
        ["rater_id", "rating", "rating-confidence", "case-difficulty", "rater-expertise"]
    ].copy()
    rows["rating"] = rows["rating"].astype(int)

    st.markdown("**Contesto della decisione** (input tabellari di D)")
    with st.form(f"form_dataset_{case_id}_{rater_choice}"):
        edited = _context_editor(
            rows, context_ranges(ctx), f"ctx_dataset_{case_id}_{rater_choice}", single=single
        )
        submitted = st.form_submit_button("Esegui inferenza fold-allineata", type="primary")

    if not submitted:
        return None
    suffix = "" if not single else f" · rater {rater_choice.split()[-1]}"
    return {
        "case_id": case_id,
        "decisions": _decisions_from_frame(edited),
        "image_dir": None,
        "is_new_case": False,
        "label": f"caso {case_id}{suffix}",
    }


def upload_inputs(ctx):
    """Three MRI uploads + rater context for a case outside the dataset."""
    st.caption(
        "Le tre viste vengono ricodificate in JPEG RGB in una cartella temporanea e "
        "seguono lo stesso preprocessing di validation/test "
        "(resize 256 → center crop 224 → normalizzazione ImageNet)."
    )
    columns = st.columns(len(ctx.orientations))
    uploads: dict = {}
    for column, orientation in zip(columns, ctx.orientations):
        with column:
            st.markdown(f"**{orientation}**")
            uploaded = st.file_uploader(
                f"MRI {orientation}",
                type=UPLOAD_TYPES,
                key=f"upload_{orientation}",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                uploads[orientation] = uploaded.getvalue()
                st.image(uploads[orientation], use_container_width=True)

    known_raters = sorted(int(r) for r in ctx.user["rater_id"].unique())
    ranges = context_ranges(ctx)

    with st.form("form_upload"):
        col_a, col_b = st.columns(2)
        case_id = int(
            col_a.number_input(
                "case_id del nuovo caso",
                min_value=1,
                value=int(suggest_new_case_id(ctx)),
                step=1,
                help="Deve essere libero: gli id già presenti nel dataset vengono rifiutati.",
            )
        )
        rater_id = int(
            col_b.number_input(
                "rater_id",
                min_value=0,
                value=int(known_raters[0]),
                step=1,
                help="Senza decisioni storiche non esiste un profilo rater: il preprocessor "
                "imputa la mediana vista in training.",
            )
        )
        col_1, col_2, col_3 = st.columns(3)
        rating = int(
            col_1.selectbox(
                "rating (0/1)", options=[0, 1], index=0,
                help="Diagnosi del rater: 0 = negativa, 1 = positiva. È un input di D.",
            )
        )
        rating_confidence = int(
            col_2.number_input(
                "rating-confidence",
                min_value=int(ranges["rating-confidence"][0]),
                max_value=int(ranges["rating-confidence"][1]),
                value=int(ranges["rating-confidence"][1]),
                step=1,
            )
        )
        case_difficulty = int(
            col_3.number_input(
                "case-difficulty",
                min_value=int(ranges["case-difficulty"][0]),
                max_value=int(ranges["case-difficulty"][1]),
                value=int(ranges["case-difficulty"][0]),
                step=1,
            )
        )
        col_4, _col_5 = st.columns(2)
        rater_expertise = int(
            col_4.number_input(
                "rater-expertise",
                min_value=int(ranges["rater-expertise"][0]),
                max_value=int(ranges["rater-expertise"][1]),
                value=int(ranges["rater-expertise"][1]),
                step=1,
            )
        )
        submitted = st.form_submit_button("Esegui inferenza fold-allineata", type="primary")

    missing = [orientation for orientation in ctx.orientations if orientation not in uploads]
    if missing:
        st.info(f"Carica le tre viste MRI per procedere (mancano: {', '.join(missing)}).")
        return None
    if not submitted:
        return None
    return {
        "case_id": case_id,
        "decisions": [
            DecisionInput(
                rater_id, rating, rating_confidence, case_difficulty, rater_expertise
            )
        ],
        "uploads": uploads,
        "is_new_case": True,
        "label": f"nuovo caso {case_id} · rater {rater_id}",
    }
def _get_risk_level(prob: float, threshold_youden: float) -> tuple[str, str, str]:
    """Determina il livello di rischio, il colore/tipo di alert e la spiegazione sintetica."""
    if prob >= 0.50 or prob >= threshold_youden + 0.15:
        return (
            "RISCHIO ERRORE ELEVATO",
            "error",
            f"La stima di probabilità d'errore ({prob:.1%}) supera la soglia critica. "
            "È fortemente raccomandata una seconda lettura o una revisione collegiale del caso.",
        )
    elif prob >= threshold_youden or prob >= 0.35:
        return (
            "RISCHIO ERRORE MEDIO",
            "warning",
            f"La stima di probabilità d'errore ({prob:.1%}) supera la soglia operativa ottimale Youden ({threshold_youden:.1%}). "
            "Si consiglia cautela e verifica approfondita della decisione clinica.",
        )
    else:
        return (
            "RISCHIO ERRORE BASSO",
            "success",
            f"La stima di probabilità d'errore ({prob:.1%}) è contenuta e inferiore alla soglia operativa ({threshold_youden:.1%}). "
            "Il modello indica concordanza favorevole tra segnale MRI e rater.",
        )


def _render_executive_summary(result):
    summary = result["decision_summary"]
    st.markdown("## Valutazione Finale Rischio Errore")
    
    if len(summary) == 1:
        row = summary.iloc[0]
        prob = float(row["d_probability_mean"])
        thr = float(row["threshold_youden_mean"])
        level, alert_type, desc = _get_risk_level(prob, thr)
        
        box_style = {
            "error": "background-color: rgba(255, 75, 75, 0.12); border-left: 6px solid #ff4b4b;",
            "warning": "background-color: rgba(255, 170, 0, 0.12); border-left: 6px solid #ffa500;",
            "success": "background-color: rgba(0, 180, 80, 0.12); border-left: 6px solid #00b450;",
        }[alert_type]
        
        st.markdown(
            f"""
            <div style="{box_style} padding: 18px 22px; border-radius: 8px; margin-bottom: 24px;">
                <h2 style="margin: 0 0 6px 0; font-size: 1.55rem; letter-spacing: 0.5px;">{level}</h2>
                <p style="margin: 0; font-size: 1.05rem; opacity: 0.95;">{desc}</p>
                <div style="margin-top: 14px; font-weight: 500; font-size: 0.95rem;">
                    Probabilità d'errore Soluzione D: <b>{prob:.1%}</b> (±{float(row['d_probability_std']):.1%}) &nbsp;|&nbsp; 
                    Soglia operativa Youden: <b>{thr:.1%}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        mean_p = float(summary["d_probability_mean"].mean())
        mean_thr = float(summary["threshold_youden_mean"].mean())
        level, alert_type, desc = _get_risk_level(mean_p, mean_thr)
        
        box_style = {
            "error": "background-color: rgba(255, 75, 75, 0.12); border-left: 6px solid #ff4b4b;",
            "warning": "background-color: rgba(255, 170, 0, 0.12); border-left: 6px solid #ffa500;",
            "success": "background-color: rgba(0, 180, 80, 0.12); border-left: 6px solid #00b450;",
        }[alert_type]
        
        n_elevato = sum(summary["d_probability_mean"] >= 0.50)
        n_youden = sum(summary["d_probability_mean"] >= summary["threshold_youden_mean"])
        
        st.markdown(
            f"""
            <div style="{box_style} padding: 18px 22px; border-radius: 8px; margin-bottom: 24px;">
                <h2 style="margin: 0 0 6px 0; font-size: 1.55rem; letter-spacing: 0.5px;">{level} (Media rater: {mean_p:.1%})</h2>
                <p style="margin: 0; font-size: 1.05rem; opacity: 0.95;">{desc}</p>
                <div style="margin-top: 14px; font-weight: 500; font-size: 0.95rem;">
                    {n_youden} rater su {len(summary)} oltre la soglia Youden &nbsp;|&nbsp;
                    {n_elevato} rater su {len(summary)} oltre soglia 50%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



def _render_metrics_panel(result):
    summary = result["decision_summary"]
    st.markdown("### Risultato dell'inferenza ensemble (5 fold)")
    
    if len(summary) == 1:
        row = summary.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "P(errore) Soluzione D",
            f"{row['d_probability_mean']:.3f}",
            delta=f"±{row['d_probability_std']:.3f} (std)",
            delta_color="off",
            help="Media delle probabilità predette dai 5 modelli D (uno per fold).",
        )
        c2.metric(
            "p_i diagnostico (AI)",
            f"{result['p_i_mean']:.3f}",
            delta=f"±{result['p_i_std']:.3f}",
            delta_color="off",
            help="Probabilità predetta di patologia dalla ResNet-18 sulle 3 viste MRI.",
        )
        c3.metric(
            "q_ir baseline",
            f"{row['q_ir_mean']:.3f}",
            help="Baseline puramente diagnostica q_ir = p_i * (1-y_ir) + (1-p_i) * y_ir",
        )
        delta_q = row["delta_vs_q_ir"]
        c4.metric(
            "Δ (D - q_ir)",
            f"{delta_q:+.3f}",
            delta_color="inverse" if delta_q > 0 else "normal",
            help="Scarto rispetto alla baseline q_ir. D integra anche il contesto e il profilo rater.",
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        mean_d = summary["d_probability_mean"].mean()
        max_rater = summary.loc[summary["d_probability_mean"].idxmax()]
        min_rater = summary.loc[summary["d_probability_mean"].idxmin()]
        
        c1.metric("P(errore) media sul caso", f"{mean_d:.3f}")
        c2.metric("p_i diagnostico (AI)", f"{result['p_i_mean']:.3f}")
        c3.metric("Rater a rischio max", f"Rater {int(max_rater['rater_id'])} ({max_rater['d_probability_mean']:.3f})")
        c4.metric("Rater a rischio min", f"Rater {int(min_rater['rater_id'])} ({min_rater['d_probability_mean']:.3f})")


def _render_summary_table(result, ctx):
    st.markdown("#### Tabella decisioni per rater")
    summary = result["decision_summary"].copy()
    
    is_dataset = not result["is_new_case"] and (int(result["case_id"]) in set(ctx.cases["case_id"].astype(int)))
    if is_dataset:
        with st.expander("Verifica cablaggio dati (Ground Truth del dataset)", expanded=False):
            u_sub = ctx.user[ctx.user["case_id"].astype(int) == int(result["case_id"])][["rater_id", "rating", "error-rating"]].drop_duplicates()
            c_sub = ctx.cases[ctx.cases["case_id"].astype(int) == int(result["case_id"])]
            gt_sub = u_sub.copy()
            if not c_sub.empty and "target" in c_sub.columns:
                gt_sub["TARGET"] = int(c_sub["target"].iloc[0])
            st.dataframe(gt_sub, hide_index=True, use_container_width=True)
            st.caption(
                "Nota metodologica: TARGET e error-rating sono mostrati solo per verifica retrospettiva "
                "e sono rigorosamente esclusi durante la pipeline di feature extraction per evitare data leakage."
            )

    display_df = pd.DataFrame({
        "Rater ID": summary["rater_id"].astype(int),
        "Rating": summary["rating"].astype(int),
        "P(errore) D": summary["d_probability_mean"].apply(lambda v: f"{v:.4f}"),
        "Std Folds": summary["d_probability_std"].apply(lambda v: f"{v:.4f}"),
        "q_ir": summary["q_ir_mean"].apply(lambda v: f"{v:.4f}"),
        "Soglia Youden": summary["threshold_youden_mean"].apply(lambda v: f"{v:.4f}"),
        "Allarme @0.5": summary["folds_flagging_0_5"].apply(lambda n: f"{n}/{result['n_folds']} fold"),
        "Allarme @Youden": summary["folds_flagging_youden"].apply(lambda n: f"{n}/{result['n_folds']} fold"),
        "Acc. storica rater": summary["rater_accuracy_foldsafe"].apply(
            lambda v: f"{v:.3f}" if pd.notna(v) else "n.d."
        ),
        "N dec. storiche": summary["rater_historical_decisions"].apply(
            lambda v: f"{int(round(v))}" if pd.notna(v) else "0"
        ),
    })
    st.dataframe(display_df, hide_index=True, use_container_width=True)


def _render_fold_details(result):
    st.markdown("#### Ispezione dettagliata per fold")
    summary = result["decision_summary"]
    fold_df_all = result["fold_frame"]
    raters = [int(r) for r in summary["rater_id"]]
    selected_rater = st.selectbox("Seleziona Rater da ispezionare", raters, key="fold_inspect_rater")
    
    fold_df = fold_df_all[fold_df_all["rater_id"] == selected_rater].copy()
    
    c_chart, c_table = st.columns([1, 1])
    with c_chart:
        chart_data = fold_df[["fold", "ai_probability_class_1", "q_ir", "d_probability"]].copy()
        chart_data.columns = ["Fold", "p_i (AI)", "q_ir (baseline)", "P(errore) D"]
        st.bar_chart(chart_data.set_index("Fold"))
        st.caption("Confronto p_i, q_ir e P(errore) D nei 5 fold.")
        
    with c_table:
        fold_table_disp = pd.DataFrame({
            "Fold": fold_df["fold"],
            "p_i": fold_df["ai_probability_class_1"].apply(lambda v: f"{v:.4f}"),
            "q_ir": fold_df["q_ir"].apply(lambda v: f"{v:.4f}"),
            "P(errore) D": fold_df["d_probability"].apply(lambda v: f"{v:.4f}"),
            "Soglia Youden": fold_df["threshold_youden"].apply(lambda v: f"{v:.4f}"),
            "Err @0.5": fold_df["decision_at_0_5"].apply(lambda v: "Sì" if v == 1 else "No"),
            "Err @Youden": fold_df["decision_at_youden"].apply(lambda v: "Sì" if v == 1 else "No"),
        })
        st.dataframe(fold_table_disp, hide_index=True, use_container_width=True)

    with st.expander(f"Features e Importanze per Rater {selected_rater}", expanded=False):
        st.markdown("**Importanza media delle feature nel modello XGBoost (5 fold)**")
        if "importance" in result and not result["importance"].empty:
            st.dataframe(result["importance"], hide_index=True, use_container_width=True)
        else:
            st.info("Importanze non disponibili.")


def _render_mri_views(result, ctx):
    st.markdown("#### Viste MRI del caso")
    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]
    
    if not result["is_new_case"]:
        imgs = get_image_paths(result["case_id"], ctx.image_dir)
        for col, orient in zip(cols, ctx.orientations):
            p = imgs.get(orient)
            with col:
                st.markdown(f"**{orient}**")
                if p and p.is_file():
                    st.image(str(p), use_container_width=True)
                else:
                    st.caption("Immagine non trovata")
    else:
        st.caption("Immagini caricate dall'utente (elaborate dal preprocessore diagnostico).")


def render_result(result, ctx):
    # 1. Risposta finale sintetica in primo piano
    _render_executive_summary(result)
    
    # 2. Sezione dettagliata a comparsa / cliccabile
    with st.expander("🔍 Mostra dettagli tecnici, metriche dei fold e dati completi", expanded=False):
        _render_metrics_panel(result)
        st.divider()
        _render_summary_table(result, ctx)
        st.divider()
        _render_fold_details(result)
        st.divider()
        _render_mri_views(result, ctx)
        st.divider()
        
        st.info(f"**Avvertenza di calibrazione e affidabilità:**\n\n{ui.CALIBRATION_WARNING}")
        
        st.markdown("#### Esporta risultati")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Scarica JSON inferenza",
                data=json.dumps(result["json"], indent=2),
                file_name=f"infer_case_{result['case_id']}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_dl2:
            csv_data = result["decision_summary"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Scarica CSV riassuntivo",
                data=csv_data,
                file_name=f"infer_case_{result['case_id']}_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )


ui.setup_page("Inferenza su un caso")
ctx = ui.get_context()
ui.sidebar_status(ctx)

st.title("Inferenza fold-allineata su un caso")
st.caption(
    "Stima della probabilità di errore umano (Soluzione D) con ensemble a 5 fold. "
    "Le feature contestuali e i profili storici del rater sono calcolati in modo fold-safe "
    "senza alcuna perdita di informazione (leakage)."
)

mode = st.radio("Modalità di inferenza", [MODE_DATASET, MODE_UPLOAD], horizontal=True)

input_data = None
if mode == MODE_DATASET:
    input_data = dataset_inputs(ctx)
else:
    input_data = upload_inputs(ctx)

if input_data is not None:
    progress_bar = st.progress(0, text="Avvio inferenza ensemble...")
    
    def on_fold_progress(fold_idx, total_folds):
        progress_bar.progress(fold_idx / total_folds, text=f"Calcolo inferenza su Fold {fold_idx}/{total_folds}...")

    temp_upload_dir = None
    try:
        if input_data.get("is_new_case"):
            temp_upload_dir = save_uploaded_images(
                input_data["uploads"],
                input_data["case_id"],
                orientations=ctx.orientations,
            )
            img_dir = temp_upload_dir
        else:
            img_dir = None
            
        result = run_inference(
            ctx,
            case_id=input_data["case_id"],
            decisions=input_data["decisions"],
            image_dir=img_dir,
            is_new_case=input_data.get("is_new_case", False),
            device_spec="auto",
            model_provider=ui.model_provider("auto"),
            d_asset_provider=ui.d_asset_provider(),
            on_fold=on_fold_progress,
        )
        progress_bar.empty()
        st.session_state[RESULT_KEY] = result
        st.success(f"Inferenza completata con successo per {input_data['label']}!")
    except Exception as exc:
        progress_bar.empty()
        st.error(f"Errore durante l'inferenza: {exc}")
    finally:
        if temp_upload_dir is not None:
            cleanup_upload_dir(temp_upload_dir)

if RESULT_KEY in st.session_state:
    render_result(st.session_state[RESULT_KEY], ctx)
