"""Verifica di due punti sollevati sul rapporto (read-only, non riaddestra nulla).

PUNTO 1 - Definizione di q_ir.
  q_ir = p_i se R=0, 1-p_i se R=1 identifica P(Y != R | X, R) solo se
  P(Y | X, R) = P(Y | X). Qui si mostra empiricamente che non coincidono.

PUNTO 2 - Statistiche rater fold-safe (rater-accuracy / rater-confidence).
  L'implementazione usa leave-one-out sulla DECISIONE. Domanda: nel pool di
  riferimento della riga (case_i, rater_r) entrano altre decisioni dello
  STESSO caso i? Si confronta con un leave-one-CASE-out esplicito e si
  ispeziona la composizione dei pool (train e val/test) piu' la feature
  importanza nei modelli salvati.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import (  # noqa: E402
    load_config,
    load_ground_truth,
    load_user_infos,
    project_path,
)
from generate_folds import load_fold_assignment  # noqa: E402
from methodology_guards import fold_safe_rater_stats  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

cfg = load_config(ROOT / "config.yaml")
user = load_user_infos(cfg, base_dir=ROOT)
gt = load_ground_truth(cfg, base_dir=ROOT)
folds = load_fold_assignment(
    project_path(cfg["outputs"]["folds_dir"], ROOT) / "case_fold_assignment.csv"
)
oof = pd.read_csv(project_path(cfg["outputs"]["diagnostic_dir"], ROOT) / "oof_predictions.csv")

bar = "=" * 78
print(bar)
print("A) STRUTTURA DELLA TABELLA USER (decide l'equivalenza LOO vs LOCO)")
print(bar)
n_dup = int(user.duplicated(subset=["case_id", "rater_id"]).sum())
per_pair = user.groupby(["case_id", "rater_id"]).size()
print(f"righe={len(user)}  casi={user['case_id'].nunique()}  rater={user['rater_id'].nunique()}")
print(f"coppie (case_id, rater_id) duplicate : {n_dup}")
print(f"decisioni per coppia (rater, case)   : min={per_pair.min()} max={per_pair.max()}")
cov = user.groupby("rater_id")["case_id"].nunique()
print(f"casi coperti per rater               : min={cov.min()} max={cov.max()}")
print(
    "\n=> Se ogni coppia (rater, case) ha UNA sola decisione, escludere la singola\n"
    "   decisione equivale a escludere l'intero caso: il leave-one-out implementato\n"
    "   coincide con un leave-one-case-out sul pool di riferimento."
)


def loco_stats(df: pd.DataFrame, train_case_ids) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-CASE-out esplicito (versione 'brutale' ma verificabile)."""
    cases = set(int(x) for x in train_case_ids)
    case = df["case_id"].to_numpy()
    rater = df["rater_id"].to_numpy()
    err = df["error-rating"].to_numpy(dtype=float)
    conf = df["rating-confidence"].to_numpy(dtype=float)
    in_train = np.array([int(c) in cases for c in case], dtype=bool)
    acc = np.full(len(df), np.nan)
    cf = np.full(len(df), np.nan)
    for r in np.unique(rater):
        m = (rater == r) & in_train
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        ref_case, ref_err, ref_conf = case[idx], err[idx], conf[idx]
        tot_e, tot_c, n = ref_err.sum(), ref_conf.sum(), float(len(idx))
        ev = (rater == r) & ~in_train
        if ev.any():
            acc[ev] = 1.0 - tot_e / n
            cf[ev] = tot_c / n
        for j in np.flatnonzero(m):
            keep = ref_case != case[j]
            k = int(keep.sum())
            if k == 0:
                continue
            acc[j] = 1.0 - ref_err[keep].sum() / k
            cf[j] = ref_conf[keep].sum() / k
    return acc, cf
print()
print(bar)
print("B) LOO IMPLEMENTATO vs LEAVE-ONE-CASE-OUT, E COMPOSIZIONE DEI POOL")
print(bar)

n_splits = int(cfg["validation"]["n_splits"])
summary_rows = []
for fold in range(1, n_splits + 1):
    train_cases = folds.loc[folds["fold"] != fold, "case_id"].astype(int).tolist()
    test_cases = set(folds.loc[folds["fold"] == fold, "case_id"].astype(int))

    impl = fold_safe_rater_stats(user, train_cases)
    acc_loco, conf_loco = loco_stats(user, train_cases)

    a_impl = impl["rater_accuracy_foldsafe"].to_numpy()
    c_impl = impl["rater_confidence_foldsafe"].to_numpy()
    in_train = user["case_id"].isin(train_cases).to_numpy()

    d_acc = float(np.nanmax(np.abs(a_impl - acc_loco)))
    d_conf = float(np.nanmax(np.abs(c_impl - conf_loco)))

    siblings_train = 0   # righe di train con un 'fratello' dello stesso caso nel pool
    siblings_eval = 0    # righe di test  con un 'fratello' dello stesso caso nel pool
    pool_train_only = True
    for r in user["rater_id"].unique():
        rm = (user["rater_id"] == r).to_numpy()
        pool_tr = user.loc[rm & in_train]      # pool usato per le righe di train (LOO)
        pool_ev = user.loc[rm & in_train]      # pool usato per le righe di test (train-only)
        if set(pool_tr["case_id"]) & test_cases:
            pool_train_only = False
        for j in user.index[rm & in_train]:
            siblings_train += int((pool_tr["case_id"] == user.at[j, "case_id"]).sum()) - 1
        for j in user.index[rm & ~in_train]:
            siblings_eval += int((pool_ev["case_id"] == user.at[j, "case_id"]).sum())

    summary_rows.append(
        {
            "fold": fold,
            "train_rows": int(in_train.sum()),
            "test_rows": int((~in_train).sum()),
            "max|diff| rater-accuracy": d_acc,
            "max|diff| rater-confidence": d_conf,
            "righe train con fratello stesso-caso nel pool": siblings_train,
            "righe test con fratello stesso-caso nel pool": siblings_eval,
            "pool di riferimento train-only": pool_train_only,
        }
    )

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
print(
    "\n'righe train con fratello stesso-caso nel pool' = 0 significa: la statistica del\n"
    "rater per la riga (case_i, rater_r) NON contiene nessuna decisione del caso i."
)

print()
print(bar)
print("C) PUNTO 1 - q_ir NON E' P(Y != R | X, R): evidenza empirica")
print(bar)

from sklearn.metrics import roc_auc_score  # noqa: E402

merged = user.merge(gt, on="case_id", how="left", validate="many_to_one")
merged = merged.merge(
    oof[["case_id", "ai_probability_class_1"]], on="case_id", how="left", validate="many_to_one"
)
rating = merged["rating"].to_numpy()
gt_y = merged["TARGET"].to_numpy()
err = merged["error-rating"].to_numpy()
p_i = merged["ai_probability_class_1"].to_numpy()
q = np.where(rating == 0, p_i, 1.0 - p_i)

print(f"n={len(merged)}  prevalenza TARGET={gt_y.mean():.4f}  tasso errori osservato={err.mean():.4f}")
print(f"media q_ir={q.mean():.4f}   media p_i={p_i.mean():.4f}")
print()
print("  R=0 : media q_ir (= media p_i)     = %.4f   |  P(errore vero | R=0) = P(Y=1|R=0) = %.4f"
      % (q[rating == 0].mean(), gt_y[rating == 0].mean()))
print("  R=1 : media q_ir (= 1 - media p_i) = %.4f   |  P(errore vero | R=1) = P(Y=0|R=1) = %.4f"
      % (q[rating == 1].mean(), 1.0 - gt_y[rating == 1].mean()))
print()
print("Per l'identita' q_ir = P(Y != R | X, R) servirebbe P(Y | X, R) = P(Y | X), cioe'")
print("che il rating non aggiunga informazione su Y oltre alle immagini. Empiricamente:")
print("  P(Y=1 | R=0) = %.4f   vs   P(Y=1) = %.4f" % (gt_y[rating == 0].mean(), gt_y.mean()))
print("  P(Y=1 | R=1) = %.4f" % gt_y[rating == 1].mean())
print("  AUROC(rating -> TARGET) = %.4f   (il rating da solo, senza immagini, e' informativo)"
      % roc_auc_score(gt_y, rating))

print()
print(bar)
print("D) QUANTO I MODELLI D SALVATI SI APPOGGIANO A rater-accuracy / rater-confidence")
print(bar)
import joblib  # noqa: E402

model_dir = project_path(cfg["models"]["d_dir"], ROOT)
imp_rows = []
for fold in range(1, n_splits + 1):
    path = model_dir / f"scenario_D_fold_{fold}.joblib"
    if not path.is_file():
        continue
    payload = joblib.load(path)
    names = list(payload["feature_names"])
    imp = np.asarray(payload["model"].feature_importances_, dtype=float)
    order = np.argsort(-imp)
    imp_rows.append(
        {
            "fold": fold,
            "top1": f"{names[order[0]]} ({imp[order[0]]:.3f})",
            "top2": f"{names[order[1]]} ({imp[order[1]]:.3f})",
            "rater-accuracy": round(float(imp[names.index("rater-accuracy")]), 4),
            "rater-confidence": round(float(imp[names.index("rater-confidence")]), 4),
            "spw": payload.get("scale_pos_weight"),
        }
    )
print(pd.DataFrame(imp_rows).to_string(index=False))
print("\n(importanze 'gain' normalizzate di XGBoost, modello nested weighted salvato)")

