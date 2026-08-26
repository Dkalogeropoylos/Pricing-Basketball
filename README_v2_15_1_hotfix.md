# Basketball Pricing Engine v2.15.1 hotfix

Apply this hotfix on top of v2.15.0.

## Fix
Streamlit number_input widgets for Team Market AUTO modifiers keep their session_state value after first creation. Previously, changing confirmed OUT players, state shrink K, or shared projected minutes could recompute the roster-state audit but leave the visible/final team modifiers stuck on the previous context.

v2.15.1 adds an AUTO-context fingerprint. When the underlying model context changes, the team modifier widgets are re-seeded from the newly computed AUTO values and any old team/player simulation cache is invalidated. Manual modifier edits remain sticky while the context is unchanged.

No model weights, near-state formulas, opponent ratios, pace logic, or 50k simulation defaults are changed by this hotfix.
