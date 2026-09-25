"""Metodologia, protocolli anticontaminazione e limiti operativi."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st

import _shared as ui

ui.setup_page("Metodologia e Limiti")
ctx = ui.get_context()
ui.sidebar_status(ctx)

st.title("Metodologia, Prevenzione Leakage e Limiti Operativi")
st.caption("Guida metodologica e approfondimento tecnico su Soluzione D.")

tab_pipe, tab_leak, tab_limits = st.tabs([
    "Architettura della Pipeline",
    "Prevenzione Data Leakage",
    "Limiti e Caveat Clinici",
])

with tab_pipe:
    st.markdown("### Architettura a due stadi (Two-Stage Pipeline)")
    st.markdown("""
    L'obiettivo del progetto **ToErrIsHuman** è stimare la probabilità condizionata che un medico radiologo (rater $r$) 
    commetta un errore diagnostico su un dato volume di risonanza magnetica (caso $i$):
    $$P(E_{ir} = 1 \\mid X_i, y_{ir}, C_{ir})$$
    
    La soluzione è strutturata in due stadi distinti:
    
    1. **Stadio 1 (Rete Diagnostica CNN):**
       - Un modello ResNet-18 tri-planare elabora congiuntamente le tre orientazioni MRI standard (assiale, coronale, sagittale).
       - Produce la probabilità predetta $p_i = P(Y_i = 1 \\mid X_i)$ di presenza della patologia.
       - Viene addestrato e validato con una 5-Fold Stratified Cross-Validation a livello di paziente/caso (nessun paziente condiviso tra train e val).
       
    2. **Stadio 2 (Modello di Errore Tabellare XGBoost - Soluzione D):**
       - Integra la probabilità oggettiva dell'algoritmo $p_i$, la decisione espressa dal medico $y_{ir}$, le variabili di contesto $C_{ir}$ 
         (confidenza, difficoltà percepita, expertise) e le statistiche storiche del medico.
       - Viene addestrato su una formulazione ponderata per contrastare il forte sbilanciamento delle classi (frequenza errore ~19%).
    """)

with tab_leak:
    st.markdown("### Garanzie e Guardie Anticontaminazione (Data Leakage)")
    st.markdown("""
    Nei modelli che stimano l'errore umano, il rischio di data leakage è estremamente elevato:
    
    1. **Esclusione rigorosa di TARGET ed error-rating:**
       - All'interno del modulo `src/methodology_guards.py` e di `src/build_D_features.py`, una whitelist rigorosa 
         scarta preventivamente qualsiasi colonna contenente la verità di terra prima dell'estrazione delle feature.
       
    2. **Profiling rater strettamente Nested (Fold-Safe):**
       - L'accuratezza storica di un rater non viene mai calcolata sull'intero dataset. Viene calcolata unicamente 
         sui casi presenti nel fold di addestramento (`train_case_ids`), escludendo integralmente i casi di test o i nuovi casi.
       - Come dimostrato nel confronto sperimentale, l'assenza di questa misura (protocollo Fast) provocherebbe 
         un'ottimistica sovrastima dell'AUROC di oltre +0.07.
         
    3. **Allineamento dei componenti dell'Ensemble:**
       - In fase di inferenza su un nuovo caso, l'algoritmo non mescola componenti casuali: per ciascun fold $k \\in \\{1..5\\}$, 
         il modello diagnostico $k$ produce $p_i^{(k)}$, il quale viene passato al preprocessore e al modello $D^{(k)}$ 
         istruiti esattamente su quello stesso fold.
    """)

with tab_limits:
    st.markdown("### Limiti Operativi e Avvertenze d'Uso")
    st.error(f"**Attenzione sull'uso clinico:**\n\n{ui.CALIBRATION_WARNING}")
    
    st.markdown("""
    - **Discriminazione vs Calibrazione:**
      La Soluzione D raggiunge un'AUROC di **0.6327**, inferiore a quella della baseline teorica $q_{ir}$ (**0.7159**). 
      Tuttavia, $q_{ir}$ produce probabilità ampiamente distorte (sovrastima grave, BSS = -0.36), mentre Soluzione D è l'unica 
      ad ottenere un BSS positivo (+0.0139) e una probabilità media coerente con la realtà clinica (15.8% vs 19.0%).
      
    - **Soglie decisionali e Trade-off Clinico:**
      Alla soglia convenzionale di 0.5, la recall sull'errore è limitata (12.77%), mentre la specificità è molto alta (96.9%). 
      Per applicazioni di screening dell'errore (second-reader o doppio controllo prioritario), è consigliabile adottare 
      la soglia ottimizzata di Youden (~0.19 - 0.22).
      
    - **Generalizzazione a nuovi rater:**
      Se un nuovo medico senza storico effettua una lettura, il modello non dispone di feature di profilo rater (accuratezza media). 
      In questo scenario, il preprocessore imputa la mediana della popolazione vista in fase di addestramento.
    """)
