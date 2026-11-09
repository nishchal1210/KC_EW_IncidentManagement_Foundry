import streamlit as st

from Postgres.postgres_client import (
    get_assignment_timeline,
    get_group_incidents,
    get_group_overview,
)
from Views.common import page_heading


page_heading(
    "Group View",
    "Review active incident clusters, current ownership, and assignment history.",
)

try:
    groups = get_group_overview()
except Exception as exc:
    st.error(f"Unable to load groups: {exc}")
    st.stop()

if groups.empty:
    st.info("No incident groups are available.")
    st.stop()

with st.expander("Groups", expanded=True):
    event = st.dataframe(
        groups,
        width="stretch",
        height=260,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="group_table",
    )

selected_rows = event.selection.rows
if selected_rows:
    group_id = int(groups.iloc[selected_rows[0]]["group_id"])
    selected = groups.iloc[selected_rows[0]]
    summary_cols = st.columns(4)
    summary_cols[0].metric("Group", group_id)
    summary_cols[1].metric("Owner", selected.get("worker_name", "Unassigned"))
    summary_cols[2].metric("Status", selected.get("status", "Unknown"))
    summary_cols[3].metric("Incidents", selected.get("incident_number", 0))

    with st.expander(f"Incidents in Group {group_id}", expanded=True):
        st.dataframe(
            get_group_incidents(group_id),
            width="stretch",
            height=320,
            hide_index=True,
        )
    with st.expander(f"Assignment timeline for Group {group_id}", expanded=False):
        st.dataframe(
            get_assignment_timeline(group_id),
            width="stretch",
            height=220,
            hide_index=True,
        )
else:
    st.caption("Select a group row to view its incidents.")
