# Basketball Pricing Engine v2.14.0 — cumulative patch over v2.12

Η v2.14 **αντικαθιστά την v2.13**. Αν έχεις ακόμα v2.12, δεν χρειάζεται να περάσεις πρώτα v2.13.

## Τι αλλάζει

### 1) Opponent allowed: stat-specific, learned from WNBA history
Δεν υπάρχει πλέον ένα γενικό hand-tuned `ratio^0.35` ή ένα default 50% opponent weight.

Για κάθε structural stat το engine κατασκευάζει pregame historical observations:
- own offensive prior,
- opponent-allowed prior,
- league prior,
- actual next-game outcome.

Μετά μαθαίνει ξεχωριστή elasticity για:
`FGA_LIVE`, `3P_SHARE`, `FTA`, `TOV`, `OREB_PER_MISS`, `AST_PER_MAKE`, `PF`, `3P_PCT`, `2P_PCT`.

Η slope shrinkάρεται με sample size + standard error + actual RMSE gain. Υπάρχει μόνο μικρό structural floor ώστε ο opponent να μην γίνεται ποτέ τελείως αόρατος από noisy early-season data.

### 2) 3PM / shooting percentages
- Το shrinkage prior για 3P%, 2P%, FT% βγαίνει πλέον από **το loaded WNBA league sample**, όχι από fixed 34.0% / 51.0% / 78.5%.
- Το opponent shooting effect παραμένει πιο shrinked από το shot-selection effect, γιατί raw opponent FG% είναι noisy.
- Το existing roster shooter-mix modifier παραμένει και συνεργάζεται με OUT/minute restrictions.

### 3) Home / away
Δεν υπάρχει hard-coded rule τύπου `away = worse shooting`.

Το location modifier:
- υπολογίζει team home/away split,
- το shrinkάρει προς το WNBA-wide home/away split,
- μένει πολύ μικρό για tactical stats (FGA / 3P share),
- επιτρέπεται να επηρεάσει λίγο περισσότερο shooting efficiency / foul-related stats μόνο αν το data το δείχνει.

Το location μπαίνει πλέον και στα `3P_PCT` / `2P_PCT`, αλλά μόνο data-driven και με μικρό cap.

### 4) Minutes / confirmed OUT
Η v2.13 minutes διόρθωση περιλαμβάνεται ολόκληρη:
- πρώτα επιλέγεται η πραγματική 8-10ish recent rotation αντί να μοιράζονται 200' σε όλο το roster,
- DNPs μετρούν ως 0 στα recent team-game minutes,
- confirmed OUT απελευθερώνει τα healthy projected minutes,
- τα minutes πηγαίνουν μέσω learned teammate replacement matrix,
- position είναι fallback prior, όχι hard rule.

v2.14 ενισχύει το positional fallback και διαβάζει combo/provider positions πιο αξιόπιστα. Σε sparse history:
`same position > adjacent G/F or F/C >>> G/C`.
Αν υπάρχει πραγματικό historical small-ball/replacement evidence, αυτό υπερισχύει.

## GitHub upload
Μέσα στο `Pricing-Basketball/core/` αντικατέστησε μαζί:
- `minutes_engine.py`
- `redistribution.py`
- `matchup.py`
- `team_model.py`

Στο root του repository αντικατέστησε:
- `streamlit_app.py`

Το `tests/v214_smoke.py` είναι μόνο για test και δεν απαιτείται για Streamlit.

## Audit που πρέπει να κοιτάξουμε πρώτο
1. `Opponent-elasticity calibration audit`
2. `Location correction` — πρέπει συνήθως να είναι πολύ κοντά στο 1.00
3. `Full rotation minute audit`
4. `Confirmed OUT minute replacement`
5. `Learned replacement weights`

