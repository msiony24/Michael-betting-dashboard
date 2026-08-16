# Macabets Automatic Result Settlement — Phase 1 Install

## What this release does

This release adds a trusted backend settlement loop without changing the live prediction model.

- Links pending analyses to provider event IDs using strict participant identity matching.
- Uses API Tennis as the tennis result source.
- Uses The Odds API as the market/closing-odds source when an event can be linked there.
- Captures pre-start moneyline snapshots during the 12 hours before an event.
- Attaches final winner, final score, provider status, settlement source, and raw provider payload.
- Grades moneyline winner predictions automatically.
- Includes tested grading functions for spread and total markets for the next rollout.
- Stores consensus closing probability, closing moneyline, probability CLV, and model edge at close when pre-start odds snapshots exist.
- Leaves retirements, walkovers, cancellations, ties, suspensions, and other exceptional cases for manual review instead of guessing sportsbook rules.
- Preserves manual overrides.
- Writes an append-only settlement audit trail.

## Installation order

### 1. Run the Supabase migration first

Open the Supabase SQL Editor and run:

`supabase_result_settlement_migration.sql`

Do this before uploading/running the Python settlement files.

### 2. Add GitHub Actions secrets

Repository → Settings → Secrets and variables → Actions → New repository secret.

Add:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `API_TENNIS_KEY`
- `THE_ODDS_API_KEY`

Important: the Supabase service-role key belongs in GitHub Actions secrets only. Do not put it in Streamlit secrets or client-facing code.

### 3. Upload/replace these GitHub files

| Local file | GitHub destination | Action |
|---|---|---|
| `analysis_store.py` | `/analysis_store.py` | Replace |
| `settle_results.py` | `/settle_results.py` | New |
| `engine/result_settlement.py` | `/engine/result_settlement.py` | New |
| `engine/settlement_providers.py` | `/engine/settlement_providers.py` | New |
| `tests/test_result_settlement.py` | `/tests/test_result_settlement.py` | New |
| `.github/workflows/settle-results.yml` | `/.github/workflows/settle-results.yml` | New |
| `supabase_result_settlement_migration.sql` | `/supabase_result_settlement_migration.sql` | Optional repository copy after running it |

### 4. First run should be a dry run

GitHub → Actions → **Settle Macabets Results** → Run workflow → set `dry_run` to `true`.

The workflow validates Python syntax and runs the settlement unit tests before it touches any prediction.

### 5. Run the real settlement

Run the workflow again with `dry_run` set to `false`.

After that, the scheduled job checks pending analyses once per hour.

## Important behavior

### Event ID linking

The engine never accepts a surname-only match. Full-name or surname + first-initial identity must agree. Ambiguous matches remain unlinked rather than being guessed.

### Closing line

The free/standard implementation does not invent a historical close after the fact. It records live pre-start snapshots and treats the latest snapshot before the scheduled start as the close. If no pre-start snapshot exists, closing-line fields remain blank.

### CLV definition

`clv_probability = closing no-vig probability - entry no-vig probability`

Positive CLV means the market moved toward the predicted side after Macabets logged the analysis.

`model_edge_at_close = Macabets predicted probability - closing no-vig probability`

That second field tells us whether Macabets still believed there was value at the closing market.

### Value-call correctness

Winner prediction accuracy and value-call outcome are kept separate.

- Strong Bet / Worth Betting / BET: the value call is graded won/lost from the actual result.
- Lean / Pass / Complete Pass: `value_call_correct` stays null. One game cannot prove that declining a wager was mathematically correct.

### Exceptions/manual review

Retirements, walkovers, cancellations, suspensions, ties, and other unusual settlement conditions are not auto-graded. The provider result is attached, `provider_link_status` becomes `needs_review`, and the prediction remains available for manual handling.

## What this release deliberately does not do yet

This is the data-foundation release. It does **not** let Macabets rewrite its own live weights.

The next releases should be built in this order:

1. Performance Center dimensional analytics.
2. Probability calibration report and shadow calibrator.
3. Weight-learning challenger models.
4. Champion/challenger model registry with out-of-sample promotion gates.

That sequencing prevents a small or noisy sample from contaminating the live model.
