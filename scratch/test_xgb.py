import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

base = Path(r"c:\Users\feede\Desktop\Uni\ToErrIsHuman")
pre = joblib.load(base / "models/D/preprocessor_fold_1.joblib")
payload = joblib.load(base / "models/D/scenario_D_fold_1.joblib")
clf = payload["model"]

print("Feature names in model:", payload.get("feature_names"))

p_ai = 0.107
rating = 0
confidence = 4.0
difficulty = 3.0
expertise = 5.0

ai_pred_class = 1 if p_ai >= 0.5 else 0
disagreement = 1 if rating != ai_pred_class else 0
p_wrong = p_ai if rating == 0 else (1.0 - p_ai)
margin = abs(p_ai - 0.5)

# Build test DataFrame
df = pd.DataFrame([{
    "rating-confidence": confidence,
    "case-difficulty": difficulty,
    "rater-expertise": expertise,
    "rater-accuracy": 0.82,
    "rater-confidence": 4.10,
    "rating": rating,
    "ai_probability_class_1": p_ai,
    "ai_predicted_class": ai_pred_class,
    "rater_ai_disagreement": disagreement,
    "ai_probability_rater_wrong": p_wrong,
    "ai_margin": margin
}])

X_t = pre.transform(df.values)
prob = clf.predict_proba(X_t)[:, 1][0]
print(f"Prob for fold 1: {prob:.4f}")
