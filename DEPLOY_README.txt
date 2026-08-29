v2.18.3 deployment patch

Replace ALL THREE files in the repository, preserving paths:
  /streamlit_app.py
  /core/availability.py
  /core/availability_impact.py

Reason: v2.18.3 streamlit_app.py passes team_rotation_board= into
availability_similarity_weight_maps(). That keyword is implemented in the
v2.18.3 core/availability.py, not in v2.18.2. Using the new app with the old
core raises TypeError at the availability_similarity_weight_maps call.

This patch changes no player-prop logic and is the same v2.18.3 math already
specified; it only ensures the matching core modules are deployed together.
