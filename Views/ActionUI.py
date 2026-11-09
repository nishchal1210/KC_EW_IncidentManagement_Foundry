import json

import streamlit as st

from Agent.AzureAIInterface import query_foundry_agent
from Postgres.postgres_client import (
    get_assigned_groups,
    get_assignment_timeline,
    get_employees,
    get_group_incidents,
)
from Views.common import configured_agent, page_heading


page_heading(
    "Action",
    "Use closed-group evidence to guide troubleshooting and next actions.",
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
    columns = st.columns((2, 1, 2))
    selected_name = columns[0].selectbox("Select Employee", list(employee_map))
    employee_id = int(employee_map[selected_name])

    groups = get_assigned_groups(employee_id)
    if groups.empty:
        columns[1].metric("Assigned Groups", 0)
        columns[2].selectbox("Select Group ID", [], disabled=True)
        st.info("This employee has no assigned groups.")
        st.stop()

    columns[1].metric("Assigned Groups", len(groups))
    group_id = int(columns[2].selectbox("Select Group ID", groups["group_id"].tolist()))


with st.expander("Incidents", expanded=False):
    st.dataframe(get_group_incidents(group_id), width="stretch", hide_index=True)

with st.expander("Assignment timeline", expanded=False):
    st.dataframe(get_assignment_timeline(group_id), width="stretch", hide_index=True)

session_key = f"action:{employee_id}:{group_id}"
history = st.session_state.setdefault(f"messages:{session_key}", [])

for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask for troubleshooting, diagnosis, resolution, or next action"):
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        payload = json.dumps(
            {"group_id": group_id, "question": prompt}
        )
        with st.chat_message("assistant"):
            with st.spinner("Searching historical resolutions..."):
                response = query_foundry_agent(
                    configured_agent("FOUNDRY_ACTION_AGENT_NAME"),
                    payload,
                    flow="action",
                )
            if not response.strip():
                raise RuntimeError(
                    "Action-Agent returned an empty response. Check the Foundry trace for tool approval or final answer output."
                )
            st.markdown(response)
        history.append({"role": "assistant", "content": response})
    except Exception as exc:
        st.error(f"Action workflow failed: {exc}")

if st.button("Clear this chat"):
    st.session_state.pop(f"messages:{session_key}", None)
    st.rerun()
