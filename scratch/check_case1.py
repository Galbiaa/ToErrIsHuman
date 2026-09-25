import sys
from pathlib import Path
base = Path(r"c:\Users\feede\Desktop\Uni\ToErrIsHuman")
sys.path.insert(0, str(base / "src"))

from data import load_config, load_user_infos
cfg = load_config(base / "config.yaml")
user = load_user_infos(cfg, base_dir=base)
case1 = user[user["case_id"] == 1]
print("Case 1 in user table:")
print(case1[["id", "rater_id", "rating", "error-rating", "rating-confidence", "case-difficulty", "rater-expertise"]])
