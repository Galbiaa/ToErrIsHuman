"""Prestazioni e benchmark: confronto Soluzione D vs Baseline e protocolli."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd
import streamlit as st

import _shared as ui

ui.setup_page("Prestazioni e Benchmark")
ctx = ui.get_context()
ui.sidebar_status(ctx)

st.title("Valutazione empirica e Benchmark (5-Fold CV)")
st.caption(
    "Analisi dettagliata delle metriche di Soluzione D, confronto con la baseline "
    "puramente diagnostica q_ir e quantificazione del protocol-leakage (Nested vs Fast)."
)

compare_dir = ui.output_subdir(ctx, "compare_dir") if "compare_dir" in ctx.cfg.get("outputs", {}) else ctx.base_dir / "outputs" / "compare"
fig_dir = ui.REPORT_FIGURE_DIR

tab_bench, tab_calib, tab_leakage, tab_figures = st.tabs([
    "Benchmark Soluzione D",
    "Calibrazione e BSS",
    "Leakage Protocollo (Nested vs Fast)",
    "Curve e Grafici",
])

with tab_bench:
    st.subheader("Metriche Soluzione D (Weighted vs Unweighted)")
    st.markdown(
        "Confronto tra la formulazione con bilanciamento pesi istanza (weighted) "
        "e standard (unweighted) calcolate out-of-fold su 5.551 decisioni (427 casi)."
    )
    p_unified = compare_dir / "unified_D_metrics_table.csv"
    df_unified = ui.table(p_unified)
    if df_unified is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUROC (Weighted)", "0.6327", help="D con reweighting per classe minoritaria")
        c2.metric("AUROC (Unweighted)", "0.6516", help="D non ponderato (guadagna ranking ma perde calibrazione)")
        c3.metric("Brier Score (Weighted)", "0.1520", help="Minore è migliore (miglior accuratezza probabilistica)")
        c4.metric("Recall @0.5 (Weighted)", "12.77%", help="Recall clinico a soglia standard 0.5")
        st.dataframe(df_unified, hide_index=True, use_container_width=True)
    else:
        ui.missing_notice(p_unified, "python -u src/compare_metrics.py --config config.yaml")

with tab_calib:
    st.subheader("Brier Score, Brier Skill Score e Calibrazione")
    st.markdown(
        "Nel contesto di errore clinico, la calibrazione probabilistica (Brier, ECE, BSS) "
        "è cruciale. Un BSS positivo indica una stima superiore alla prevalenza costante di errore (19.04%)."
    )
    p_calib = compare_dir / "calibration_bss_summary.csv"
    df_calib = ui.table(p_calib)
    if df_calib is not None:
        st.dataframe(df_calib, hide_index=True, use_container_width=True)
        st.markdown("""
        **Osservazioni chiave:**
        - **Soluzione D (Weighted)** è l'unico modello con **BSS positivo (+0.0139)** e un **ECE di soli 0.065**, dimostrando probabilità affidabili e ben calibrate.
        - **q_ir (Baseline)** ottiene un **BSS fortemente negativo (-0.3629)** e sovrastima sistematicamente il rischio (mean p = 0.3622 vs prevalenza 0.1904).
        - **D (Unweighted)** mostra un BSS negativo (-0.0230) e sottostima l'errore (mean p = 0.0769), rendendolo inadeguato all'uso probabilistico.
        """)
    else:
        ui.missing_notice(p_calib, "python -u src/compare_metrics.py --config config.yaml")

with tab_leakage:
    st.subheader("Verifica del Leakage: Protocollo Nested vs Fast")
    st.markdown(
        "Il protocollo **Fast** commette data leakage calcolando i profili storici del rater "
        "su tutto il dataset prima di splittare i fold. Il protocollo **Nested** esegue "
        "il profiling rigorosamente fold-safe all'interno di ciascun fold di training."
    )
    p_nested = compare_dir / "nested_vs_fast_comparison.csv"
    df_nested = ui.table(p_nested)
    if df_nested is not None:
        st.dataframe(df_nested, hide_index=True, use_container_width=True)
        c_l1, c_l2 = st.columns(2)
        c_l1.metric("AUROC Fast (con leakage)", "0.7028")
        c_l2.metric("AUROC Nested (rigoroso)", "0.6327", delta="-0.0701", delta_color="inverse")
        st.warning(
            "Il protocollo con leakage sovrastima l'AUROC di **+0.0701** punti (+11% relativo). "
            "La pipeline e l'applicazione web implementano esclusivamente la formulazione Nested protetta."
        )
    else:
        ui.missing_notice(p_nested, "python -u src/train_D.py --config config.yaml --protocol compare")

with tab_figures:
    st.subheader("Galleria Figure e Analisi Grafiche")
    st.markdown("Curve ROC, Precision-Recall e Calibrazione generate dalla pipeline.")
    figs = [
        fig_dir / "roc_overlay.png",
        fig_dir / "calibration_D.png",
        fig_dir / "pr_D.png",
        fig_dir / "decision_curve.png",
        fig_dir / "calibration_q_ir.png",
        fig_dir / "roc_oof.png",
    ]
    caps = [
        "ROC Overlay (D vs q_ir vs diagnostico)",
        "Calibrazione Soluzione D (Weighted)",
        "Precision-Recall Soluzione D",
        "Decision Curve Analysis (Net Benefit)",
        "Calibrazione Baseline q_ir (sovrastima)",
        "ROC OOF Classificatore Diagnostico ResNet-18",
    ]
    ui.figure_gallery(figs, caps, columns=2)
