from pathlib import Path

import pandas as pd


RAW_DATA_FILE = Path(__file__).resolve().parents[1] / "LocalFiles" / "incident.csv"


def GetRawData() -> list[dict]:
    if not RAW_DATA_FILE.exists():
        return []
    return pd.read_csv(RAW_DATA_FILE).to_dict(orient="records")
