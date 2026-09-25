"""Home page: panoramica ToErrIsHuman e stato del sistema."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamlit as st

import _shared as ui
from web_inference import artifact_status, environment_info

ui.setup_page("Home")
ctx = ui.get_context()
ui.sidebar_status(ctx)

st.title("ToErrIsHuman 🧠")
st.subheader("Piattaforma di Stima dell'Errore Medico (Soluzione D) con Leakage Prevention")

st.markdown("""
Benvenuto nella dashboard interattiva di **ToErrIsHuman**. 
Il sistema consente di stimare la probabilità di errore umano commesso da radiologi 
durante la diagnosi di risonanze magnetiche al ginocchio (ACL tear), combinando 
un modello diagnostico deep learning tri-planare con gradient boosting contestuale (XGBoost).
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("🎯 **Inferenza Interattiva**\n\nEsegui inferenza fold-safe su un caso del dataset o caricando tre nuove viste MRI (assiale, coronale, sagittale).")
with col2:
    st.info("📊 **Benchmark & Metriche**\n\nConfronta le performance di Soluzione D con la baseline q_ir e analizza l'impatto del leakage (Nested vs Fast).")
with col3:
    st.info("🛡️ **Metodologia & Guardie**\n\nApprofondisci i protocolli anticontaminazione, l'allineamento dei fold e i limiti operativi del sistema.")

st.divider()

st.markdown("### Stato degli Artefatti e dell'Ambiente")
env = environment_info(ctx)
status = artifact_status(ctx)

c_e1, c_e2, c_e3, c_e4 = st.columns(4)
c_e1.metric("Casi Dataset", f"{env['n_cases']}")
c_e2.metric("Decisioni Totali", f"{env['n_decisions']}")
c_e3.metric("Folds Cross-Validation", f"{env['n_folds']}")
c_e4.metric("Dispositivo Hardware", f"{env['device'].upper()}")

if status["all_ok"]:
    st.success("Tutti i 5 fold diagnostici e i modelli di Soluzione D sono correttamente compilati e pronti all'uso.")
else:
    st.warning(f"Attenzione: {len(status['missing'])} artefatti non trovati.")

st.markdown("### Flusso di Lavoro Rapido")
st.markdown("""
1. Seleziona **Inferenza su un caso** dal menu laterale.
2. Scegli se esplorare retrospettivamente un caso clinico reale o caricare una nuova triade di immagini.
3. Regola o inserisci le feature contestuali del rater (confidenza, difficoltà percepita, expertise).
4. Avvia l'inferenza ensemble a 5 fold e visualizza la scomposizione delle probabilità e l'allineamento tra modelli.
""")
