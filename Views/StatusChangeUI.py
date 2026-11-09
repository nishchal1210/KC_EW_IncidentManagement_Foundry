import json

import streamlit as st

from Agent.AzureAIInterface import query_foundry_agent
from Postgres.postgres_client import (
    get_assigned_groups,
    get_assignment_timeline,
    get_employees,
    get_group_incidents,
)
from Views.common import configured_agent, json_from_text, page_heading


page_heading(
    "Status Change",
    "Move assigned groups through WIP, pause, closure, cancellation, or reassignment.",
)

try:
    employees = get_employees()
except Exception as exc:
    st.error(f"Unable to load employees: {exc}")
    st.stop()

if employees.empty:
    st.info("No employees are available.")
    st.stop()

employee_map = dict(zip(employees["employee_name"], employees["employee_id"]))
with st.container(border=True):
    employee_name = st.selectbox("Select Employee", list(employee_map))
    employee_id = int(employee_map[employee_name])
    groups = get_assigned_groups(employee_id)

    if groups.empty:
        st.info("This employee has no assigned groups.")
        st.stop()

    with st.form("status_change_form"):
        left, middle = st.columns(2)
        group_id = int(left.selectbox("Select Group ID", groups["group_id"].tolist()))
        new_status = middle.selectbox(
            "Status Change", options=["WIP", "Pause", "Close", "Re-assign", "Cancel"]
        )
        reason = st.text_area("Reason", height=220)
        submitted = st.form_submit_button("Submit", type="primary", width="stretch")

if submitted:
    if not reason.strip():
        st.warning("Enter a reason before submitting the status change.")
    else:
        payload = json.dumps(
            {"group_id": group_id, "new_status": new_status, "reason": reason.strip()}
        )
        try:
            with st.spinner("Applying status change..."):
                result = query_foundry_agent(
                    configured_agent("FOUNDRY_LIFECYCLE_AGENT_NAME"),
                    payload,
                    flow="lifecycle",
                )
            st.session_state["last_lifecycle_result"] = result
            st.success("Lifecycle workflow completed.")
        except Exception as exc:
            st.error(f"Lifecycle workflow failed: {exc}")

if result := st.session_state.get("last_lifecycle_result"):
    st.subheader("Latest lifecycle result")
    parsed = json_from_text(result)
    if isinstance(parsed, dict):
        cols = st.columns(5)
        cols[0].metric("Group", parsed.get("group_id", "None"))
        cols[1].metric("Previous", parsed.get("previous_status", parsed.get("previous_action", "None")))
        cols[2].metric("Current", parsed.get("new_status", "None"))
        cols[3].metric("Assignment", parsed.get("assignment_status", "None"))
        cols[4].metric("Employee", parsed.get("employee_id", "None"))
        with st.expander("Lifecycle response", expanded=False):
            st.json(parsed, expanded=True)
    else:
        st.code(result, language="json")

with st.expander("Current incidents", expanded=True):
    st.dataframe(get_group_incidents(group_id), width="stretch", hide_index=True)

with st.expander("Assignment timeline", expanded=False):
    st.dataframe(get_assignment_timeline(group_id), width="stretch", hide_index=True)
