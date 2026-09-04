BET TRACKER — TEMPORARY STOIXIMAN QUICK IMPORT V6

Base: OUTRIGHT_HIERARCHY_V5 (includes Shared Picks + Shared By analytics).

What this adds:
- Temporary ⚡ Quick Import tab for the attached Sep 1–3 Stoiximan screenshots.
- 34 review rows, all Pending and hidden from normal tracker views until approved.
- Existing tipster `Chat GPT` is used by default; no tipster is ever created by the helper.
- Tipster, Origin, Confidence, Bookmaker, date, event, market, odds and stake remain editable before approval.
- 5-fold rebound bet is classified as Outright Parlay.
- Standalone Janelle Salaun tournament top-points bets are classified as Outright.
- Same exact selection on the same date is merged where useful using stake-weighted effective odds.
  * Yeun Heo U7.5 Points on 02/09: €25 @ effective 1.878.
  * Janelle Salaun Top Points Scorer on 02/09: €20 @ effective 8.6875.
- Different dates stay separate so date analytics are not distorted.
- Pamela Rosado 3+ @7.00 is the only CHECK row: player inferred from adjacent 2+/4+ cards.

No SQL is required because this reuses the existing needs_review/import_batch columns from the previous import helper.

Deploy: replace app.py and ADD stoiximan_sep_2026_quick_import.json. Other V5 files are included only as a synchronized backup.
