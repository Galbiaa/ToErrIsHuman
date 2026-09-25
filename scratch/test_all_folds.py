import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

base = Path(r"c:\Users\feede\Desktop\Uni\ToErrIsHuman")

p_ai = 0.107
rating = 0
confidence = 4.0
difficulty = 3.0
expertise = 5.0

ai_pred_class = 1 if p_ai >= 0.5 else 0
disagreement = 1 if rating != ai_pred_class else 0
p_wrong = p_ai if rating == 0 else (1.0 - p_ai)
margin = abs(p_ai - 0.5)

probs = []
thrs = []

for fold in range(1, 6):
    pre = joblib.load(base / f"models/D/preprocessor_fold_{fold}.joblib")
    payload = joblib.load(base / f"models/D/scenario_D_fold_{fold}.joblib")
    clf = payload["model"]
    thr = float(payload.get("threshold_youden", 0.5))
    thrs.append(thr)

    # In app.py:
    # "rater-accuracy": 0.80
    # "rater-confidence": float(confidence) = 4.0
    df = pd.DataFrame([{
        "rating-confidence": confidence,
        "case-difficulty": difficulty,
        "rater-expertise": expertise,
        "rater-accuracy": 0.80,
        "rater-confidence": confidence,
        "rating": rating,
        "ai_probability_class_1": p_ai,
        "ai_predicted_class": ai_pred_class,
        "rater_ai_disagreement": disagreement,
        "ai_probability_rater_wrong": p_wrong,
        "ai_margin": margin
    }])

    X_t = pre.transform(df.values)
    p_err = clf.predict_proba(X_t)[:, 1][0]
    probs.append(p_err)
    print(f"Fold {fold}: prob={p_err:.4f}, thr={thr:.4f}")

print(f"Mean prob: {np.mean(probs):.4f}")
print(f"Mean threshold: {np.mean(thrs):.4f}")
