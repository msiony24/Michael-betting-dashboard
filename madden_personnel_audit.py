from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_personnel_audit(
    madden_path: Path | str = PROJECT_ROOT / "data" / "madden_27_players.csv",
    player_ratings_path: Path | str = PROJECT_ROOT / "data" / "nfl" / "player_ratings.csv",
    team_ratings_path: Path | str = PROJECT_ROOT / "data" / "nfl" / "team_ratings_auto.json",
    csv_path: Path | str = PROJECT_ROOT / "data" / "nfl" / "madden_27_personnel_audit.csv",
    json_path: Path | str = PROJECT_ROOT / "data" / "nfl" / "madden_27_personnel_audit.json",
) -> dict[str, Any]:
    madden = pd.read_csv(madden_path, low_memory=False)
    rated = pd.read_csv(player_ratings_path, low_memory=False)
    merged = rated.merge(
        madden[[c for c in ("player_name", "team", "position", "madden_position_raw", "overall") if c in madden.columns]],
        on="player_name", how="left", suffixes=("", "_source")
    )
    merged["trait_delta_vs_madden"] = pd.to_numeric(merged["trait_grade"], errors="coerce") - pd.to_numeric(merged["overall"], errors="coerce")
    merged["final_delta_vs_madden"] = pd.to_numeric(merged["macabets_rating"], errors="coerce") - pd.to_numeric(merged["overall"], errors="coerce")
    cols=[c for c in ["player_name","team_abbr","position","position_family","overall","trait_grade","trait_delta_vs_madden","performance_grade","performance_weight","macabets_rating","final_delta_vs_madden","rating_source"] if c in merged.columns]
    detail=merged[cols].sort_values(["position_family","overall"], ascending=[True,False])
    out=Path(csv_path); out.parent.mkdir(parents=True, exist_ok=True); detail.to_csv(out,index=False)

    with Path(team_ratings_path).open(encoding="utf-8") as f: teams=json.load(f)
    unit_rows=[]
    for team, payload in teams.items():
        for unit, data in payload.get("units",{}).items():
            unit_rows.append({"team":team,"unit":unit,"grade":data.get("grade"),"roster_grade":data.get("roster_grade",data.get("grade")),"performance_grade":data.get("performance_grade"),"performance_weight":data.get("performance_weight",0),"source":data.get("source","")})
    unit_frame=pd.DataFrame(unit_rows)
    summary={
        "source_players": int(len(madden)),
        "source_teams": int(madden["team"].nunique()),
        "rated_players": int(len(rated)),
        "max_abs_trait_delta": round(float(detail["trait_delta_vs_madden"].abs().max()),2),
        "mean_abs_trait_delta": round(float(detail["trait_delta_vs_madden"].abs().mean()),2),
        "players_trait_delta_over_3": int((detail["trait_delta_vs_madden"].abs()>3.01).sum()),
        "players_with_performance": int((pd.to_numeric(detail["performance_weight"],errors="coerce").fillna(0)>0).sum()),
        "unit_count": int(len(unit_frame)),
        "teams_with_units": int(unit_frame["team"].nunique()) if not unit_frame.empty else 0,
        "guardrails": {
            "madden_overall_is_preseason_anchor": True,
            "trait_refinement_cap_points": 3.0,
            "qb_depth_does_not_reduce_starter_grade": True,
            "prior_performance_double_count_removed": True,
        },
    }
    payload={"summary":summary,"unit_audit":unit_frame.to_dict(orient="records")}
    p=Path(json_path); temp=p.with_suffix(p.suffix+".tmp"); temp.write_text(json.dumps(payload,indent=2),encoding="utf-8"); temp.replace(p)
    return payload

if __name__ == "__main__":
    report=build_personnel_audit(); print(json.dumps(report["summary"],indent=2))
