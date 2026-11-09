import json
from typing import Any


def getContext_from_dict(selected_dict: dict[str, Any]) -> str:
    payload = {key: value for key, value in selected_dict.items() if key != "Created"}
    return json.dumps(payload, default=str)
