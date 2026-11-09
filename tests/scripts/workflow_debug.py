import argparse
import os
from pathlib import Path

from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

from Agent.AzureAIInterface import _get_credential


load_dotenv()


IGNORED_STREAM_EVENTS = {
    "response.in_progress",
    "response.content_part.added",
    "response.content_part.done",
    "response.mcp_list_tools.in_progress",
    "response.mcp_list_tools.completed",
    "response.mcp_call.in_progress",
    "response.mcp_call_arguments.delta",
    "response.mcp_call_arguments.done",
    "response.mcp_call.completed",
}


def _read_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def run_workflow(workflow_name: str, prompt: str, debug: bool = True) -> str:
    endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
    if not endpoint:
        raise RuntimeError("Set FOUNDRY_PROJECT_ENDPOINT.")

    project_client = AIProjectClient(endpoint=endpoint, credential=_get_credential())
    output_parts: list[str] = []

    with project_client:
        openai_client = project_client.get_openai_client()
        conversation = openai_client.conversations.create()
        if debug:
            print(f"Created conversation: {conversation.id}")

        try:
            stream = openai_client.responses.create(
                conversation=conversation.id,
                extra_body={
                    "agent_reference": {
                        "name": workflow_name,
                        "type": "agent_reference",
                    }
                },
                input=prompt,
                stream=True,
                metadata={"x-ms-debug-mode-enabled": "1"},
            )

            for event in stream:
                if event.type == "response.output_text.delta":
                    output_parts.append(event.delta)
                    print(event.delta, end="", flush=True)
                elif event.type == "response.output_text.done":
                    print()
                elif event.type == "response.output_item.added":
                    item = event.item
                    if debug and item.type == "workflow_action":
                        print(
                            f"\nWorkflow item {item.action_id} is {item.status}; "
                            f"previous item: {item.previous_action_id}"
                        )
                    elif item.type == "mcp_approval_request":
                        print("\nMCP APPROVAL REQUIRED")
                        print(f"Tool: {item.name}")
                        print(f"Arguments: {item.arguments}")
                elif event.type == "response.output_item.done":
                    item = event.item
                    if debug and item.type == "workflow_action":
                        print(
                            f"\nWorkflow item {item.action_id} is {item.status}; "
                            f"previous item: {item.previous_action_id}"
                        )
                elif event.type == "response.created" and debug:
                    print(f"\nResponse started: {event.response.id}")
                elif event.type == "response.completed" and debug:
                    print("\nWorkflow response completed")
                elif event.type in IGNORED_STREAM_EVENTS:
                    continue
                elif debug:
                    print(f"\nUnknown event type: {event.type}")
        finally:
            openai_client.conversations.delete(conversation_id=conversation.id)
            if debug:
                print("\nConversation deleted")

    return "".join(output_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a Foundry workflow run for debugging.")
    parser.add_argument(
        "--workflow",
        default=os.getenv("FOUNDRY_INTAKE_AGENT_NAME", "ITIL"),
        help="Published Foundry workflow or agent-reference asset name.",
    )
    # Default path updated to reflect moved sample data location
    parser.add_argument("--prompt-file", default="LocalFiles/prompt.txt")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_workflow(args.workflow, _read_prompt(args.prompt_file), debug=not args.quiet)


if __name__ == "__main__":
    main()
