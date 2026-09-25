"""Sanity checks for src/web_inference.py (service layer of the Streamlit app).

Verifies that the fold-aligned ensemble exposed to the web app reproduces the
official CLI inference (`src/infer_new_case.py` -> outputs/analysis/infer_case_1.json),
that the uploaded-case path works without touching the dataset, and that TARGET
never reaches the D feature matrix.

Run from the repository root:  .venv\\Scripts\\python.exe -u tmp/verify_web_inference.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import web_inference as wi  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {label}{'' if not detail else ' -> ' + detail}")
    if not condition:
        raise AssertionError(label)


def by_rater(frame, column: str) -> dict:
    return {
        int(row["rater_id"]): float(row[column])
        for row in frame.to_dict(orient="records")
    }


def main() -> None:
    _cfg, ctx = wi.load_config_and_context(ROOT / "config.yaml")

    status = wi.artifact_status(ctx)
    check(
        "artefatti completi (5 fold x diagnostico/preprocessor/D)",
        status["all_ok"],
        f"{len(status['missing'])} mancanti",
    )
    check("allowlist D ha 11 feature", len(ctx.allow) == 11, str(ctx.allow))
    check(
        "nessuna feature proibita nell'allowlist",
        not any(f == "TARGET" or f.upper() == "TARGET" for f in ctx.allow),
    )
    print(f"      ambiente: {wi.environment_info(ctx)}")

    catalog = wi.case_catalog(ctx)
    check("catalogo: una riga per caso", len(catalog) == ctx.n_cases == 427, str(len(catalog)))
    check("catalogo: tutte le MRI presenti", bool(catalog["has_all_images"].all()))
    check(
        "catalogo: 5551 decisioni totali",
        int(catalog["n_decisions"].sum()) == 5551,
        str(int(catalog["n_decisions"].sum())),
    )

    decisions_case1 = wi.decisions_of_case(ctx, 1)
    check("caso 1: 13 decisioni rater", len(decisions_case1) == 13, str(len(decisions_case1)))
    check("range di contesto letti dai dati", "rating-confidence" in wi.context_ranges(ctx))
    check("suggerimento id libero oltre il dataset", wi.suggest_new_case_id(ctx) > 427)
    try:
        wi.validate_new_case_id(ctx, 1)
        check("id esistente rifiutato in modalita upload", False)
    except ValueError:
        check("id esistente rifiutato in modalita upload", True)

    # ---------------------------------------------------------------- dataset
    inputs = [
        wi.DecisionInput(
            rater_id=int(row["rater_id"]),
            rating=int(row["rating"]),
            rating_confidence=float(row["rating-confidence"]),
            case_difficulty=float(row["case-difficulty"]),
            rater_expertise=float(row["rater-expertise"]),
            error_rating=int(row["error-rating"]),
        )
        for _, row in decisions_case1.iterrows()
    ]
    progress: list = []
    result = wi.run_inference(
        ctx, case_id=1, decisions=inputs, on_fold=lambda f, n: progress.append(f)
    )
    check("progress callback per ogni fold", progress == [1, 2, 3, 4, 5], str(progress))
    check("griglia per fold completa (13 rater x 5 fold)", len(result["fold_frame"]) == 65)
    check("nessuna colonna TARGET nella griglia", "TARGET" not in result["fold_frame"].columns)
    check(
        "q_ir coerente con ai_probability_rater_wrong",
        bool(
            np.allclose(
                result["fold_frame"]["q_ir"],
                result["fold_frame"]["ai_probability_rater_wrong"],
            )
        ),
    )
    probs = result["fold_frame"]["d_probability"].to_numpy()
    check("probabilita D in [0,1]", bool(np.all((probs >= 0) & (probs <= 1))))

    reference_path = ROOT / "outputs" / "analysis" / "infer_case_1.json"
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        ref_d = {int(r["rater_id"]): float(r["d_probability_ensemble"]) for r in reference["ensemble_decisions"]}
        ref_p = {int(r["rater_id"]): float(r["ai_probability_mean"]) for r in reference["ensemble_decisions"]}
        got_d = by_rater(result["decision_summary"], "d_probability_mean")
        got_p = by_rater(result["decision_summary"], "ai_probability_mean")
        delta_d = max(abs(ref_d[k] - got_d[k]) for k in ref_d)
        delta_p = max(abs(ref_p[k] - got_p[k]) for k in ref_p)
        check(
            "ensemble D identico a src/infer_new_case.py (case 1)",
            delta_d < 1e-6,
            f"delta massimo {delta_d:.2e} su {len(ref_d)} rater",
        )
        check(
            "p_i identica a src/infer_new_case.py (case 1)",
            delta_p < 1e-6,
            f"delta massimo {delta_p:.2e}",
        )
    else:
        print("[SKIP] outputs/analysis/infer_case_1.json assente: confronto CLI saltato")

    excel_global = ctx.user.set_index("rater_id")["rater-accuracy"]
    foldsafe = result["decision_summary"].set_index("rater_id")["rater_accuracy_foldsafe"]
    gap = float((excel_global - foldsafe).abs().max())
    check(
        "rater-accuracy fold-safe sostituisce la media globale Excel",
        gap > 1e-6,
        f"scarto massimo {gap:.4f}",
    )
    check(
        "profilo rater calcolato su casi storici (n>0)",
        bool(result["decision_summary"]["rater_historical_decisions"].notna().all()),
    )
    check("payload JSON serializzabile", isinstance(json.dumps(result["json"]), str))

    # ----------------------------------------------------------------- upload
    free_id = wi.suggest_new_case_id(ctx)
    upload_dir = Path(tempfile.mkdtemp(prefix=wi.UPLOAD_DIR_PREFIX))
    try:
        primary = Path(result["image_dir"])
        uploads = {
            orientation: (primary / f"case1{orientation}.{ctx.extension}").read_bytes()
            for orientation in ctx.orientations
        }
        written = wi.save_uploaded_images(
            uploads,
            free_id,
            orientations=ctx.orientations,
            extension=ctx.extension,
            root=upload_dir,
        )
        check(
            "immagini caricate scritte con il naming della pipeline",
            all(
                (written / f"case{free_id}{o}.{ctx.extension}").is_file()
                for o in ctx.orientations
            ),
            str(written),
        )
        uploaded = wi.run_inference(
            ctx,
            case_id=free_id,
            decisions=[wi.DecisionInput(0, 0, 5, 2, 4), wi.DecisionInput(3, 1, 4, 3, 4)],
            image_dir=written,
            is_new_case=True,
        )
        check(
            "caso caricato: ground truth assente (y_true_demo NaN)",
            bool(uploaded["fold_frame"]["y_true_demo"].isna().all()),
        )
        check("caso caricato: 2 rater x 5 fold", len(uploaded["fold_frame"]) == 10)
        check(
            "caso caricato: nessuna feature NaN (imputer del preprocessor)",
            bool(uploaded["fold_frame"][ctx.allow].notna().all().all()),
        )
        check(
            "caso caricato: probabilita D in [0,1]",
            bool(uploaded["fold_frame"]["d_probability"].between(0, 1).all()),
        )
        repeated = wi.run_inference(
            ctx,
            case_id=free_id,
            decisions=[wi.DecisionInput(0, 0, 5, 2, 4)],
            image_dir=written,
            is_new_case=True,
        )
        single = uploaded["decision_summary"].loc[
            uploaded["decision_summary"]["rater_id"] == 0, "d_probability_mean"
        ]
        check(
            "inferenza deterministica su richieste ripetute",
            bool(np.allclose(repeated["decision_summary"]["d_probability_mean"], single)),
        )
    finally:
        wi.cleanup_upload_dir(upload_dir)
        check("cartella temporanea di upload rimossa", not upload_dir.exists(), str(upload_dir))

    print("\nTutte le verifiche superate.")


if __name__ == "__main__":
    main()
