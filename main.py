import streamlit as st


st.set_page_config(layout="wide", page_title="TechM EquipWise - AI")

parser_agent = st.Page("Views/parser_agent.py", title="Parser Agent", icon=":material/chat_info:")
grouping = st.Page("Views/group_view.py", title="Grouping", icon=":material/table_view:")
action = st.Page("Views/ActionUI.py", title="Action", icon=":material/support_agent:")
status_change = st.Page(
    "Views/StatusChangeUI.py", title="Status Change", icon=":material/change_circle:"
)

navigation = st.navigation(
    {"Agentic AI": [parser_agent, grouping, action, status_change]}
)
navigation.run()
