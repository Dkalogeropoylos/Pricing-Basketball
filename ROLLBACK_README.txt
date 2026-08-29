SAFE ROLLBACK TO v2.18.2 + DREB WIRING FIX

Overwrite exactly these files in your repo:
1) streamlit_app.py
2) core/availability.py
3) core/availability_impact.py

This restores the pre-v2.18.3 team roster-state behavior while retaining the v2.18.2 + DREB wiring-fix app.
Do NOT copy v2.18.4 minutes_engine.py or redistribution.py for this rollback.
