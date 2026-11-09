import pandas as pd
import streamlit as st


def PrintChatHistory(chat_history: pd.DataFrame) -> None:
    for _, row in chat_history.dropna(subset=["Type", "Message"]).iterrows():
        role = "user" if row["Type"] == "Request" else "assistant"
        with st.chat_message(role):
            st.markdown(str(row["Message"]))
