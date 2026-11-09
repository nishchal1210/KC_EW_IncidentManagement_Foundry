import argparse
import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client


load_dotenv()
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")

EXPECTED_TOOLS = {
    "health",
    "find_candidate_groups",
    "insert_incident",
    "get_next_employee",
    "assign_employee",
    "get_employee_groups",
    "get_group_incidents",
    "get_group_summary",
    "get_group_assignment",
    "search_similar_closed_resolutions",
    "update_group_status",
    "reassign_group",
}


def decode(result: Any) -> Any:
    values = []
    for item in result.content:
        text = getattr(item, "text", "")
        try:
            values.append(json.loads(text))
        except json.JSONDecodeError:
            values.append(text)
    return values[0] if len(values) == 1 else values


async def run(group_id: int | None, employee_id: int | None) -> None:
    async with sse_client(MCP_SERVER_URL) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            missing = EXPECTED_TOOLS - names
            if missing:
                raise RuntimeError(f"Missing MCP tools: {sorted(missing)}")
            print(f"PASS tool discovery ({len(names)} tools)")

            health = decode(await session.call_tool("health", {}))
            print("PASS health", json.dumps(health))

            if group_id is not None:
                summary = decode(await session.call_tool("get_group_summary", {"group_id": group_id}))
                incidents = decode(await session.call_tool("get_group_incidents", {"group_id": group_id}))
                print("PASS group summary", json.dumps(summary, default=str))
                print(f"PASS group incidents ({len(incidents or [])} rows)")

            if employee_id is not None:
                groups = decode(
                    await session.call_tool(
                        "get_employee_groups",
                        {"employee_id": employee_id, "include_completed": False},
                    )
                )
                print(f"PASS employee groups ({len(groups or [])} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only MCP smoke test.")
    parser.add_argument("--group-id", type=int)
    parser.add_argument("--employee-id", type=int)
    args = parser.parse_args()
    asyncio.run(run(args.group_id, args.employee_id))


if __name__ == "__main__":
    main()
