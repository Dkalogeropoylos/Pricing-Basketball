# Basketball Pricing Engine v2.12.0 — cumulative patch over v2.11.0

## Τι διορθώνει

### 1) Confirmed OUT που πριν μπορούσε να μην κουνάει το team line
Το exact historical OUT-state παραμένει το πρώτο layer. Αν όμως υπάρχουν 0 ή πολύ λίγα πραγματικά games στο ίδιο state (π.χ. Bonner μόλις έφυγε και η Phoenix δεν έχει παίξει χωρίς αυτή), το μοντέλο **δεν μένει πλέον ουδέτερο**.

Χτίζει τρία 200-minute counterfactual rotations:
- **healthy**: οι selected OUT επιστρέφουν, χωρίς minute restrictions,
- **OUT-only**: οι selected OUT είναι 0', χωρίς minute restrictions,
- **current**: OUT state + shared projected-minute restrictions.

Από τη διαφορά των roster compositions βγαίνει ένα **shrunk roster-state modifier** για FGA, 3P share, FTA, TOV, OREB, AST, PF, DREB, STL, BLK και μικρή shooter-mix efficiency διόρθωση.

Overlap guard: όσο αυξάνει το exact-state historical confidence, το synthetic OUT fallback μικραίνει αυτόματα.

### 2) Shared minute restriction / return
Στο Game Setup υπάρχει νέο **Shared minute overrides / restrictions**. Χρησιμοποίησέ το για κάτι σαν Plum 22–24'. Αυτό γράφεται στο shared game context και επηρεάζει:
- Team Markets,
- 200-minute player rotation,
- Player Props.

Τα local player-tab minute overrides παραμένουν μόνο scenario overrides.

### 3) 3PA / 2PA
Παραμένει η σωστή αλυσίδα:
`possessions -> TOV -> FGA -> 3P share -> 3PA / 2PA`.

Αλλά το opponent 3P-share context δεν είναι πλέον ένα φοβικό +/-6% multiplier. Το μοντέλο εφαρμόζει το **opponent defensive deviation from league** πάνω στο offensive 3P share σε logit space. Έτσι μια πραγματικά ακραία άμυνα στο shot profile μπορεί να μετακινήσει ουσιαστικά το share, χωρίς να αντικαθιστά το offensive identity.

### 4) FTA
Το FTA/poss χρησιμοποιεί πλέον πραγματικό offense-defense log-rate blend. Μια ομάδα που τραβάει πολλές βολές δεν μένει σχεδόν αμετακίνητη όταν παίζει με opponent που suppresses FTA.

### 5) Player Props όταν δεν υπάρχουν exact OUT games
Τα exact joint OUT games παραμένουν primary. Αν δεν υπάρχουν, το μοντέλο χρησιμοποιεί capped **vacated-opportunity redistribution** για usage / 3PA role / FTA role / creation / rebound role.

Τα projected minutes είναι ήδη ξεχωριστό layer, άρα το fallback δεν ξαναμετρά απλώς τα ίδια χαμένα λεπτά. Όσο αυξάνει exact-state confidence, το role fallback shrinkάρει προς 1.00.

## GitHub upload
Από το zip:

Στο `Pricing-Basketball/core/` ανέβασε/αντικατάστησε:
- `availability.py`
- `matchup.py`
- `availability_impact.py` **(ΝΕΟ)**

Στο root `Pricing-Basketball/` αντικατάστησε:
- `streamlit_app.py`

Το `tests/v212_smoke.py` είναι προαιρετικό.

**ΠΡΟΣΟΧΗ:** τα core files πρέπει να μπουν μέσα στο `/core`, όχι στο root.

## A/B test που πρέπει να κάνουμε
1. WSH @ PHX χωρίς OUT/restriction.
2. Bonner = OUT.
3. Bonner = OUT + Plum shared minute override (π.χ. το πραγματικό restriction που θέλεις να χρησιμοποιήσεις).

Στο Model Audit κοίτα:
- Exact OUT-state audit,
- Roster-state bridge,
- Shot architecture,
- Final 3P-share context modifier,
- Final FTA context modifier,
- Player sparse-OUT fallback audit.

Το σημαντικό δεν είναι απλώς να αλλάξει το line· πρέπει να φαίνεται **γιατί** άλλαξε και η ίδια απουσία να εμφανίζεται μία φορά ως historical evidence + μόνο το residual fallback που δεν καλύπτεται από αυτό.
