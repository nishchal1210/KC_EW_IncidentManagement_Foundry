import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from Agent.AzureAIInterface import query_foundry_agent
from Views.common import configured_agent, page_heading


PARSER_ONLY_FIELDS = {
    "timestamp",
    "hw_system",
    "sw_system",
    "issue",
    "source",
    "severity",
}


def _now_timestamp() -> str:
    return datetime.now().strftime("%d-%b-%Y %H:%M:%S")


def _clean_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _selected_row_dict(row: pd.Series) -> dict[str, str]:
    return {str(column): _clean_cell(value) for column, value in row.items()}


def _find_field(row_dict: dict[str, str], *aliases: str) -> str:
    """Return the first non-empty value for any of the given column name aliases (case-insensitive)."""
    lower_map = {k.lower(): v for k, v in row_dict.items()}
    for alias in aliases:
        value = lower_map.get(alias.lower(), "")
        if value:
            return value
    return ""


def _build_csv_row_prompt_from_dict(row_data: dict[str, str]) -> str:
    current_timestamp = _now_timestamp()
    return (
        "One incident row has been selected from an uploaded CSV file.\n\n"
        f"Use this exact timestamp for the incident: {current_timestamp}\n"
        "Important: ignore any date, time, opened_at, created_at, updated_at, "
        "or timestamp value present in the CSV row. Column names can vary across "
        "CSV files, so infer the incident fields from the row values.\n\n"
        "Parse this row, then continue the intake workflow for grouping and assignment.\n\n"
        "Selected CSV row JSON:\n"
        f"{json.dumps(row_data, indent=2)}"
    )


def _build_csv_row_prompt(row: pd.Series) -> str:
    return _build_csv_row_prompt_from_dict(_selected_row_dict(row))


def _looks_like_parser_only_output(value: str) -> bool:
    parsed = _select_workflow_result(value)
    if not parsed:
        return False
    return PARSER_ONLY_FIELDS.issubset(parsed) and "group_id" not in parsed


def _extract_json_objects(value: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    while index < len(value):
        start = value.find("{", index)
        if start == -1:
            break
        try:
            parsed, end = decoder.raw_decode(value[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
        index = start + end
    return objects


def _select_workflow_result(value: str) -> dict[str, Any] | None:
    objects = _extract_json_objects(value)
    if not objects:
        return None
    for item in reversed(objects):
        if "group_id" in item or "assignment_status" in item or "employee_id" in item:
            return item
    return objects[-1]


def _select_parser_result(value: str) -> dict[str, Any] | None:
    for item in _extract_json_objects(value):
        if PARSER_ONLY_FIELDS.intersection(item) and "group_id" not in item:
            return item
    return None


def _format_nullable(value: Any) -> str:
    if value is None or value == "":
        return "None"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _render_workflow_result(raw_result: str) -> None:
    workflow_result = _select_workflow_result(raw_result)
    parser_result = _select_parser_result(raw_result)

    st.subheader("Workflow result")
    if not workflow_result:
        st.warning("The workflow returned text, but no JSON object could be extracted.")
        st.code(raw_result)
        return

    if PARSER_ONLY_FIELDS.issubset(workflow_result) and "group_id" not in workflow_result:
        st.warning(
            "This looks like Agent1 parser output only. Check that "
            "FOUNDRY_INTAKE_AGENT_NAME points to the published workflow, and that "
            "FOUNDRY_INTAKE_AGENT_ENDPOINT is empty unless it is a workflow endpoint."
        )
    else:
        st.success("Incident processed by Agent2.")

    cols = st.columns(5)
    cols[0].metric("Incident", _format_nullable(workflow_result.get("incident_number")))
    cols[1].metric("Group", _format_nullable(workflow_result.get("group_id")))
    cols[2].metric("New group", _format_nullable(workflow_result.get("created_new_group")))
    cols[3].metric("Employee", _format_nullable(workflow_result.get("employee_id")))
    cols[4].metric(
        "Assignment",
        _format_nullable(workflow_result.get("assignment_status") or workflow_result.get("status")),
    )

    st.json(workflow_result, expanded=True)

    if parser_result and parser_result != workflow_result:
        with st.expander("Parsed incident from Agent1", expanded=False):
            st.json(parser_result, expanded=False)

    with st.expander("Raw Foundry response", expanded=False):
        st.code(raw_result)


page_heading(
    "Intake Workflow",
    "Upload an incident CSV, select one row, then process it through the intake workflow.",
)

with st.expander("Incident File", expanded=True):
    with st.container(border=True):
        uploaded_file = st.file_uploader("Incident CSV", type=["csv"])

        if uploaded_file is None:
            st.info("Upload a CSV file to begin incident intake.")
        else:
            try:
                incidents = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Unable to read CSV file: {exc}")
                st.stop()

            if incidents.empty:
                st.warning("The uploaded CSV does not contain any incident rows.")
                st.stop()

            st.caption("Select one incident row to process.")
            event = st.dataframe(
                incidents,
                width="stretch",
                height=320,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="intake_csv_table",
            )

            selected_rows = event.selection.rows
            if not selected_rows:
                st.caption("Select a row from the table above.")
            else:
                selected_row = incidents.iloc[selected_rows[0]]
                row_dict = _selected_row_dict(selected_row)

                # ── Read-only metadata row ─────────────────────────────────
                c1, c2, c3 = st.columns(3)
                c1.text_input(
                    "Source",
                    value=_find_field(row_dict, "source", "origin", "channel", "contact_type"),
                    disabled=True,
                )
                c2.text_input(
                    "Severity",
                    value=_find_field(row_dict, "severity", "priority", "urgency", "impact"),
                    disabled=True,
                )
                c3.text_input("TimeStamp", value=_now_timestamp(), disabled=True)

                # ── Editable description fields ────────────────────────────
                st.markdown("**Short Description**")
                short_desc = st.text_input(
                    "Short Description",
                    value=_find_field(
                        row_dict,
                        "short_description", "short description", "summary",
                        "issue", "subject", "title",
                    ),
                    label_visibility="collapsed",
                    key=f"short_desc_{selected_rows[0]}",
                )

                st.markdown("**Long Description**")
                long_desc = st.text_area(
                    "Long Description",
                    value=_find_field(
                        row_dict,
                        "description", "long_description", "details",
                        "comments", "work_notes", "additional_comments",
                    ),
                    height=160,
                    label_visibility="collapsed",
                    key=f"long_desc_{selected_rows[0]}",
                )

                if st.button("Process Selected Incident", type="primary", width="stretch"):
                    # Merge user edits back into the row dict before sending to agent
                    merged = {**row_dict}
                    if short_desc:
                        merged["Short_description"] = short_desc
                    if long_desc:
                        merged["Description"] = long_desc
                    prompt = _build_csv_row_prompt_from_dict(merged)
                    try:
                        with st.spinner("Parsing, grouping, and assigning the selected incident..."):
                            result = query_foundry_agent(
                                configured_agent("FOUNDRY_INTAKE_AGENT_NAME"),
                                prompt,
                                flow="intake",
                            )

                        parser_result = _select_parser_result(result)
                        if _looks_like_parser_only_output(result) and parser_result:
                            grouping_agent = os.getenv("FOUNDRY_GROUPING_AGENT_NAME", "Agent2").strip() or "Agent2"
                            with st.spinner("Workflow returned parser output only. Sending it to Agent2..."):
                                result = query_foundry_agent(
                                    grouping_agent,
                                    json.dumps(parser_result),
                                    flow="grouping",
                                )
                        st.session_state["last_selected_csv_row"] = merged
                        st.session_state["last_intake_result"] = result
                    except Exception as exc:
                        st.error(f"Intake workflow failed: {exc}")

if selected := st.session_state.get("last_selected_csv_row"):
    with st.expander("Last processed CSV row", expanded=False):
        st.json(selected, expanded=False)

if result := st.session_state.get("last_intake_result"):
    _render_workflow_result(result)
