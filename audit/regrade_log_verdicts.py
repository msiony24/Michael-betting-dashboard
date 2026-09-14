"""Re-grade a logged analysis history under the new (confidence-free) verdict rules.

Answers the only question that matters about the change: would removing the
confidence gate, and de-vigging the market price, have made or lost money on
predictions that have already been graded?

Input is a CSV export of the Analysis Log (the download button on the log
table). Expected columns, matching the dashboard:

    Date, Sport, Event, Prediction, Confidence, Actual Line, Fair Line,
    Price Assessment, Verdict, Prediction Result

Model probability is recovered from Fair Line, market probability from Actual
Line. The logged Verdict is the OLD rule's real output, so no replay or
approximation of the old thresholds is needed -- it is compared directly
against what the new rules produce on the same row.

Run:
    python audit/regrade_log_verdicts.py path/to/analysis_log.csv
    python audit/regrade_log_verdicts.py log.csv --sport Tennis

Caveat worth stating plainly: this is an in-sample replay on a single history.
A rule set that looks better here has not been validated, only fitted to what
already happened. Treat a good result as "not obviously worse," not as proof.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.bet_math import moneyline_price_quality  # noqa: E402

ACTIONABLE = {"Strong Bet", "Worth Betting", "Lean"}


def parse_american(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("+", "")
    if not text or text.lower() in {"nan", "none", "-"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def american_to_probability(odds):
    odds = float(odds)
    if odds == 0:
        return None
    return (-odds) / ((-odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def payout_units(odds):
    odds = float(odds)
    return odds / 100.0 if odds > 0 else 100.0 / (-odds)


def opposite_american(odds):
    """Approximate the other side's price from one side, assuming a typical hold.

    Only used when the log does not carry both prices. The result feeds the
    de-vig, so it is an assumption, not a measurement -- the column is flagged
    in the output so you can see which rows relied on it.
    """
    p = american_to_probability(odds)
    if p is None:
        return None
    other = max(0.01, min(0.99, (1.0 + ASSUMED_HOLD) - p))
    if other >= 0.5:
        return -round(100.0 * other / (1.0 - other))
    return round(100.0 * (1.0 - other) / other)


ASSUMED_HOLD = 0.045  # typical two-way moneyline hold


def load(path, sport=None):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if sport and "sport" in df.columns:
        df = df[df["sport"].astype(str).str.lower() == sport.lower()]
    return df


def settled(row):
    """True/False if graded, None if not yet settled."""
    for col in ("prediction_correct", "value_call_correct"):
        v = row.get(col)
        if isinstance(v, bool):
            return v
        if pd.notna(v) and str(v).strip().lower() in {"true", "false"}:
            return str(v).strip().lower() == "true"
    status = str(row.get("status", "")).strip().lower()
    if status in {"won", "win", "correct"}:
        return True
    if status in {"lost", "loss", "incorrect"}:
        return False
    return None


def regrade(df, use_no_vig=True):
    rows = []
    skipped = {"unsettled": 0, "no_price": 0, "no_probability": 0}
    for _, r in df.iterrows():
        won = settled(r)
        if won is None:
            skipped["unsettled"] += 1
            continue

        pick = str(r.get("prediction", "")).strip()
        a = str(r.get("participant_a", "")).strip()
        odds_a = parse_american(r.get("market_odds_a"))
        odds_b = parse_american(r.get("market_odds_b"))
        if odds_a is None or odds_b is None:
            skipped["no_price"] += 1
            continue
        picked_a = pick == a
        market = odds_a if picked_a else odds_b
        opponent = odds_b if picked_a else odds_a

        # Model probability, preferring the stored value over the fair line.
        model_p = r.get("predicted_probability")
        model_p = float(model_p) if pd.notna(model_p) else None
        if model_p is None:
            fair = parse_american(r.get("fair_line"))
            model_p = american_to_probability(fair) if fair is not None else None
        if model_p is None:
            skipped["no_probability"] += 1
            continue
        # Stored probability is for the predicted side already.
        if model_p < 0.5 and picked_a is False:
            pass

        report = moneyline_price_quality(
            model_p, market, opponent_odds=opponent if use_no_vig else None
        )
        rows.append(
            {
                "date": r.get("event_date"),
                "sport": r.get("sport"),
                "event": r.get("event_name"),
                "market": market,
                "model_p": model_p,
                "edge_pts": report["edge_points"],
                "old_verdict": str(r.get("recommendation", "")).strip(),
                "new_verdict": report["verdict"],
                "won": won,
                "units": payout_units(market) if won else -1.0,
            }
        )
    return pd.DataFrame(rows), skipped


def summarize(df, verdict_col, label):
    print(f"\n--- {label} ---")
    print(f"{'verdict':16} {'n':>4} {'W-L':>8} {'hit%':>7} {'units':>8} {'ROI':>8}")
    act = df[df[verdict_col].isin(ACTIONABLE)]
    for v in ["Strong Bet", "Worth Betting", "Lean", "Pass", "Complete Pass"]:
        g = df[df[verdict_col] == v]
        if g.empty:
            continue
        w = int(g["won"].sum())
        u = g["units"].sum()
        print(f"{v:16} {len(g):>4} {w:>4}-{len(g)-w:<3} "
              f"{w/len(g)*100:>6.1f}% {u:>+8.2f} {u/len(g)*100:>+7.1f}%")
    if not act.empty:
        w = int(act["won"].sum())
        u = act["units"].sum()
        print(f"{'ACTIONABLE':16} {len(act):>4} {w:>4}-{len(act)-w:<3} "
              f"{w/len(act)*100:>6.1f}% {u:>+8.2f} {u/len(act)*100:>+7.1f}%")


def edge_buckets(df):
    """ROI by size of edge -- the table that should set your threshold."""
    print("\n--- ROI by edge size (all rows, regardless of verdict) ---")
    bins = [(-99, -6), (-6, -2), (-2, 0), (0, 2), (2, 4), (4, 8), (8, 99)]
    print(f"{'edge (pts)':14} {'n':>4} {'hit%':>7} {'units':>8} {'ROI':>8}")
    for lo, hi in bins:
        g = df[(df["edge_pts"] >= lo) & (df["edge_pts"] < hi)]
        if g.empty:
            continue
        w = int(g["won"].sum())
        u = g["units"].sum()
        print(f"{lo:>+4} to {hi:>+4}   {len(g):>4} {w/len(g)*100:>6.1f}% "
              f"{u:>+8.2f} {u/len(g)*100:>+7.1f}%")


def movement(df):
    print("\n--- rows whose verdict changed ---")
    moved = df[df["old_verdict"] != df["new_verdict"]]
    if moved.empty:
        print("none")
        return
    counts = defaultdict(lambda: [0, 0, 0.0])
    for _, r in moved.iterrows():
        k = f"{r['old_verdict']} -> {r['new_verdict']}"
        counts[k][0] += 1
        counts[k][1] += int(r["won"])
        counts[k][2] += r["units"]
    print(f"{'transition':38} {'n':>4} {'W':>4} {'units':>8}")
    for k, (n, w, u) in sorted(counts.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:38} {n:>4} {w:>4} {u:>+8.2f}")
    print(f"\n{len(moved)} of {len(df)} rows changed verdict "
          f"({len(moved)/len(df)*100:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Analysis Log CSV export")
    ap.add_argument("--sport", default=None)
    ap.add_argument("--raw-vig", action="store_true",
                    help="skip the de-vig (compare against raw implied price)")
    args = ap.parse_args()

    df = load(args.csv, args.sport)
    if df.empty:
        raise SystemExit("No rows found.")

    out, skipped = regrade(df, use_no_vig=not args.raw_vig)
    if out.empty:
        raise SystemExit("No rows had both a usable Actual Line and Fair Line.")

    print(f"Rows in export: {len(df)}   skipped: {skipped}")
    print(f"Graded rows: {len(out)}"
          + (f"   sport={args.sport}" if args.sport else "")
          + ("   (raw vig)" if args.raw_vig else "   (de-vigged from both stored sides)"))

    summarize(out, "old_verdict", "AS LOGGED (confidence gate active)")
    summarize(out, "new_verdict", "NEW RULES (confidence removed)")
    edge_buckets(out)
    movement(out)


if __name__ == "__main__":
    main()
