import argparse
import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import AzureOpenAI


load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")

CLASSIFIER_AGENT_DEFINITION = "You are an incident classifier."
CLASSIFIER_QUERY = "Does the current issue belong to this group?"
CLASSIFIER_GUIDANCE = "Answer only YES or NO."

_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("DEPLOYMENT")


KEY_ALIASES = {
    "incident_number": ("incident_number", "Incident_number", "Incident Number", "number"),
    "timestamp": ("timestamp", "Timestamp", "time", "created_at"),
    "hw_system": ("hw_system", "HW_system", "hardware_system", "Hardware", "hardware"),
    "sw_system": ("sw_system", "SW_system", "software_system", "Software", "software"),
    "issue": ("issue", "Issue", "description", "summary", "short_description"),
    "source": ("source", "Source"),
    "severity": ("severity", "Severity", "priority", "Priority"),
}


def _first_present(data: dict[str, Any], aliases: tuple[str, ...], default: str = "") -> str:
    for key in aliases:
        if key in data and data[key] is not None:
            return str(data[key]).strip()
    return default


def normalize_incident(parsed_incident: dict[str, Any]) -> dict[str, str]:
    """Normalize parser output to the PostgreSQL column names used by the reference app."""
    incident = {
        field: _first_present(parsed_incident, aliases)
        for field, aliases in KEY_ALIASES.items()
    }
    missing = [field for field in ("incident_number", "timestamp", "issue") if not incident[field]]
    if missing:
        raise ValueError(f"Parsed incident is missing required fields: {', '.join(missing)}")
    return incident


def _decode_mcp_result(result: Any) -> Any:
    if not getattr(result, "content", None):
        return None

    values: list[Any] = []
    for item in result.content:
        text = getattr(item, "text", None)
        if text is None:
            continue
        try:
            values.append(json.loads(text))
        except json.JSONDecodeError:
            values.append(text)

    if len(values) == 1:
        return values[0]
    return values


async def call_tool(session: ClientSession, name: str, arguments: dict[str, Any]) -> Any:
    result = await session.call_tool(name, arguments)
    decoded = _decode_mcp_result(result)
    if isinstance(decoded, dict) and decoded.get("error"):
        raise RuntimeError(f"MCP tool {name} failed: {decoded['error']}")
    return decoded


def classify_group(current_issue: str, existing_issues: list[str]) -> bool:
    """
    Use Azure AI Foundry/Azure OpenAI with the same classifier prompt as the
    reference project: Current Issue + Existing Issues in Group -> YES/NO.
    """
    if not _deployment:
        raise RuntimeError("Set AZURE_OPENAI_DEPLOYMENT or DEPLOYMENT for the grouping classifier.")

    foundry_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("SUBSCRIPTION_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("API_VERSION"),
    )

    context = f"""
Current Issue:{current_issue}
Existing Issues in Group:{existing_issues}
"""
    response = foundry_client.chat.completions.create(
        model=_deployment,
        temperature=0,
        max_tokens=8,
        messages=[
            {"role": "system", "content": CLASSIFIER_AGENT_DEFINITION},
            {"role": "user", "content": context},
            {"role": "user", "content": CLASSIFIER_QUERY},
            {"role": "user", "content": CLASSIFIER_GUIDANCE},
        ],
    )

    answer = (response.choices[0].message.content or "").strip().upper()
    return answer.startswith("YES")


async def group_and_insert_incident(parsed_incident: dict[str, Any]) -> dict[str, Any]:
    """
    Second-agent flow:
    1. Find candidate groups by hw_system + sw_system.
    2. Ask Foundry classifier if the issue belongs to each candidate group.
    3. Create a new group if no candidate matches.
    4. Insert incident into PostgreSQL.
    5. Pick the next available employee and assign the group when available.
    """
    incident = normalize_incident(parsed_incident)

    async with sse_client(MCP_SERVER_URL) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()

            candidates = await call_tool(
                session,
                "find_candidate_groups",
                {
                    "hw_system": incident["hw_system"],
                    "sw_system": incident["sw_system"],
                },
            )
            candidates = candidates or []

            selected_group_id: int | None = None
            for group in candidates:
                if classify_group(incident["issue"], group.get("issues") or []):
                    selected_group_id = int(group["group_id"])
                    break

            created_new_group = selected_group_id is None
            inserted = await call_tool(
                session,
                "insert_incident",
                {
                    **incident,
                    "group_id": selected_group_id or 0,
                    "status": "Active",
                },
            )
            selected_group_id = int(inserted["group_id"])

            employee = await call_tool(
                session,
                "get_next_employee",
                {"sw_system": incident["sw_system"]},
            )

            assignment = None
            employee_id = employee.get("employee_id") if isinstance(employee, dict) else None
            if employee_id is not None:
                assignment = await call_tool(
                    session,
                    "assign_employee",
                    {
                        "employee_id": int(employee_id),
                        "group_id": selected_group_id,
                    },
                )

            return {
                "status": "success",
                "incident_number": incident["incident_number"],
                "group_id": selected_group_id,
                "created_new_group": created_new_group,
                "inserted": inserted,
                "employee": employee,
                "assignment": assignment,
            }


def _load_incident_from_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run the incident grouping agent.")
    parser.add_argument("--incident-json", help="Parsed incident as a JSON object string.")
    parser.add_argument("--incident-file", help="Path to a JSON file containing one parsed incident.")
    args = parser.parse_args()

    if args.incident_json:
        return json.loads(args.incident_json)

    if args.incident_file:
        with open(args.incident_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    raise SystemExit("Provide --incident-json or --incident-file.")


async def main() -> None:
    result = await group_and_insert_incident(_load_incident_from_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
