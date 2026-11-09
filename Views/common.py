import json
import os
from typing import Any

import streamlit as st


def page_heading(title: str, subtitle: str = "") -> None:
    # Main heading (centered)
    st.markdown(
        """
        <h1 style="
            text-align:center;
            color:#E05C5C;
            font-size:2.6rem;
            font-weight:700;
            margin-top:0;
            margin-bottom:1.8rem;
        ">
            TechM EquipWise
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # Page title (left aligned)
    st.title(title)

    if subtitle:
        st.caption(subtitle)

def pill(label: str, tone: str = "") -> None:
    st.caption(label)


def json_from_text(value: str) -> Any:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def configured_agent(env_name: str) -> str:
    value = os.getenv(env_name)
    if not value:
        raise RuntimeError(f"Set {env_name} in the application environment.")
    return value
