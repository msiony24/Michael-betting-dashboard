from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    ROOT / "audit" / "results_phase2" / "tennis_phase2_predictions.csv",
    ROOT / "audit" / "phase2_predictions_input.csv",
]
IN = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])
OUT = ROOT / "audit" / "results_phase4"
OUT.mkdir(parents=True, exist_ok=True)

TRANSITION_MULT = 2.125
PRESSURE_MULT = 0.975
SECONDARY_CAP = 0.04


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    pred = p >= 0.5
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_confidence": float(np.mean(np.maximum(p, 1 - p))),
    }


def logloss_each(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_each(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return (p - y) ** 2


def bootstrap_paired(y, p_a, p_b, reps=3000, seed=9804):
    """Return paired A-B metric differences. Negative means A is better."""
    y = np.asarray(y, float)
    p_a = np.asarray(p_a, float)
    p_b = np.asarray(p_b, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    ll_a, ll_b = logloss_each(y, p_a), logloss_each(y, p_b)
    br_a, br_b = brier_each(y, p_a), brier_each(y, p_b)
    dll = ll_a - ll_b
    dbr = br_a - br_b
    vals_ll = np.empty(reps)
    vals_br = np.empty(reps)
    vals_acc = np.empty(reps)
    ca = ((p_a >= .5) == y).astype(float)
    cb = ((p_b >= .5) == y).astype(float)
    dacc = ca - cb
    for i in range(reps):
        idx = rng.integers(0, n, n)
        vals_ll[i] = dll[idx].mean()
        vals_br[i] = dbr[idx].mean()
        vals_acc[i] = dacc[idx].mean()
    def row(metric, point, arr):
        return {
            "metric": metric,
            "point_difference_a_minus_b": float(point),
            "ci_2_5": float(np.quantile(arr, .025)),
            "ci_97_5": float(np.quantile(arr, .975)),
            "probability_a_better": float(np.mean(arr < 0)) if metric != "accuracy" else float(np.mean(arr > 0)),
        }
    return [
        row("log_loss", dll.mean(), vals_ll),
        row("brier", dbr.mean(), vals_br),
        row("accuracy", dacc.mean(), vals_acc),
    ]


def add_candidates(df):
    d = df.copy()
    raw_adj = (
        TRANSITION_MULT * d.adj_transition.astype(float)
        + PRESSURE_MULT * d.adj_pressure.astype(float)
    )
    d["v098_secondary_adjustment"] = np.clip(raw_adj, -SECONDARY_CAP, SECONDARY_CAP)
    d["p_v098_safe"] = np.clip(
        d.p_core.astype(float) + d["v098_secondary_adjustment"],
        .05, .95,
    )
    # Rounded implementation candidate. This is intentionally nearly identical to the tuned safe spec.
    rounded_adj = np.clip(2.0 * d.adj_transition.astype(float) + 1.0 * d.adj_pressure.astype(float), -SECONDARY_CAP, SECONDARY_CAP)
    d["p_v098_rounded"] = np.clip(d.p_core.astype(float) + rounded_adj, .05, .95)
    # Production-proxy from the Phase 2 historical approximation. It includes all tested secondary
    # factors and the v0.97 shrink. It does not include every UI/manual context factor in engine/tennis.py,
    # so it must be labeled as a proxy, not an exact recreation of every live analysis.
    if "p_all_tested_shrunk" in d:
        d["p_v097_proxy"] = d.p_all_tested_shrunk.astype(float)
    return d


def calibration_table(g, col):
    conf = np.maximum(g[col].to_numpy(float), 1 - g[col].to_numpy(float))
    chosen_correct = ((g[col].to_numpy(float) >= .5) == g.y.to_numpy(float)).astype(float)
    bins = np.array([.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,1.001])
    labels = ["50-55","55-60","60-65","65-70","70-75","75-80","80-85","85-90","90-95","95+"]
    b = pd.cut(conf, bins=bins, labels=labels, right=False, include_lowest=True)
    tmp = pd.DataFrame({"bin":b,"confidence":conf,"correct":chosen_correct})
    out = tmp.groupby("bin", observed=False).agg(n=("correct","size"), mean_confidence=("confidence","mean"), actual_accuracy=("correct","mean")).reset_index()
    out["calibration_gap_actual_minus_confidence"] = out.actual_accuracy - out.mean_confidence
    out.insert(0,"model",col)
    return out


df = pd.read_csv(IN, parse_dates=["date"])
df["year"] = df.date.dt.year
df = add_candidates(df)
models = ["p_core", "p_v098_safe", "p_v098_rounded"]
if "p_v097_proxy" in df:
    models.append("p_v097_proxy")

# Overall + year summary.
rows = []
for period, g in [("ALL",df)] + [(str(y),g) for y,g in df.groupby("year")]:
    base = metrics(g.y, g.p_core)
    for col in models:
        m = metrics(g.y, g[col])
        rows.append({
            "period":period,"model":col,**m,
            "delta_accuracy_vs_core":m["accuracy"]-base["accuracy"],
            "delta_log_loss_vs_core":m["log_loss"]-base["log_loss"],
            "delta_brier_vs_core":m["brier"]-base["brier"],
        })
summary = pd.DataFrame(rows)
summary.to_csv(OUT/"tennis_phase4_head_to_head_summary.csv", index=False)

# Surface and round robustness.
rob = []
for dim in ["surface","round"]:
    for val,g in df.groupby(dim):
        if len(g) < 80:
            continue
        base = metrics(g.y,g.p_core)
        cand = metrics(g.y,g.p_v098_safe)
        rob.append({"dimension":dim,"segment":val,"n":len(g),
                    "core_accuracy":base["accuracy"],"candidate_accuracy":cand["accuracy"],
                    "delta_accuracy":cand["accuracy"]-base["accuracy"],
                    "core_log_loss":base["log_loss"],"candidate_log_loss":cand["log_loss"],
                    "delta_log_loss":cand["log_loss"]-base["log_loss"],
                    "core_brier":base["brier"],"candidate_brier":cand["brier"],
                    "delta_brier":cand["brier"]-base["brier"]})
pd.DataFrame(rob).to_csv(OUT/"tennis_phase4_segment_robustness.csv",index=False)

# Calibration tables.
cal = pd.concat([calibration_table(df,m) for m in ["p_core","p_v098_safe"]], ignore_index=True)
cal.to_csv(OUT/"tennis_phase4_calibration.csv",index=False)

# Adjustment diagnostics.
adj = df.p_v098_safe - df.p_core
adj_rows = [{
    "n":len(df),"mean_adjustment":adj.mean(),"mean_abs_adjustment":adj.abs().mean(),
    "median_abs_adjustment":adj.abs().median(),"p90_abs_adjustment":adj.abs().quantile(.90),
    "p95_abs_adjustment":adj.abs().quantile(.95),"max_abs_adjustment":adj.abs().max(),
    "pct_over_2pp":float((adj.abs()>.02).mean()),"pct_over_4pp":float((adj.abs()>.04).mean()),
    "pct_at_probability_clip":float(((df.p_v098_safe<=.0500001)|(df.p_v098_safe>=.9499999)).mean()),
}]
pd.DataFrame(adj_rows).to_csv(OUT/"tennis_phase4_adjustment_diagnostics.csv",index=False)

# Bootstrap paired confidence for all data and temporal years separately.
boot=[]
comparisons=[("v098_safe","core","p_v098_safe","p_core")]
if "p_v097_proxy" in df:
    comparisons.append(("v098_safe","v097_proxy","p_v098_safe","p_v097_proxy"))
for period,g in [("ALL",df),("2025",df[df.year==2025]),("2026",df[df.year==2026])]:
    for an,bn,ac,bc in comparisons:
        for row in bootstrap_paired(g.y,g[ac],g[bc]):
            boot.append({"period":period,"model_a":an,"model_b":bn,**row})
pd.DataFrame(boot).to_csv(OUT/"tennis_phase4_bootstrap_confidence.csv",index=False)

# Save candidate prediction dataset so future audits can reproduce exact Phase 4 numbers.
keep=["date","surface","round","player_a","player_b","y","p_core","adj_transition","adj_pressure","p_v098_safe","p_v098_rounded"]
if "p_v097_proxy" in df: keep.append("p_v097_proxy")
df[keep].to_csv(OUT/"tennis_phase4_candidate_predictions.csv",index=False)

# Decision table with explicit guardrails.
all_core=metrics(df.y,df.p_core); all_c=metrics(df.y,df.p_v098_safe)
y25=df[df.year==2025]; y26=df[df.year==2026]
m25c=metrics(y25.y,y25.p_core);m25=metrics(y25.y,y25.p_v098_safe)
m26c=metrics(y26.y,y26.p_core);m26=metrics(y26.y,y26.p_v098_safe)
decision = pd.DataFrame([
    {"criterion":"2025 log loss improves","passed":m25["log_loss"]<m25c["log_loss"],"value":m25["log_loss"]-m25c["log_loss"]},
    {"criterion":"2026 log loss improves","passed":m26["log_loss"]<m26c["log_loss"],"value":m26["log_loss"]-m26c["log_loss"]},
    {"criterion":"Overall Brier improves","passed":all_c["brier"]<all_core["brier"],"value":all_c["brier"]-all_core["brier"]},
    {"criterion":"Overall accuracy not worse by >0.25pp","passed":all_c["accuracy"]>=all_core["accuracy"]-.0025,"value":all_c["accuracy"]-all_core["accuracy"]},
    {"criterion":"Candidate adjustment stays surgical (p95 <= 5pp)","passed":float(adj.abs().quantile(.95))<=.05,"value":float(adj.abs().quantile(.95))},
])
decision.to_csv(OUT/"tennis_phase4_release_gate.csv",index=False)

print("INPUT",IN)
print("\nHEAD TO HEAD")
print(summary.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
print("\nADJUSTMENT")
print(pd.DataFrame(adj_rows).to_string(index=False,float_format=lambda x:f"{x:.6f}"))
print("\nRELEASE GATE")
print(decision.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
