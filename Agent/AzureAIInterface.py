import argparse
import json
import os
import sys
from functools import lru_cache
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from azure.ai.projects import AIProjectClient
from azure.core.credentials import TokenCredential
from azure.identity import ClientSecretCredential, DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import APITimeoutError, BadRequestError, NotFoundError, OpenAI, PermissionDeniedError


load_dotenv()

AGENT_ENV = {
    "intake": "FOUNDRY_INTAKE_AGENT_NAME",
    "grouping": "FOUNDRY_GROUPING_AGENT_NAME",
    "action": "FOUNDRY_ACTION_AGENT_NAME",
    "investigation": "FOUNDRY_INVESTIGATION_AGENT_NAME",
    "lifecycle": "FOUNDRY_LIFECYCLE_AGENT_NAME",
}

AGENT_ENDPOINT_ENV = {
    "intake": "FOUNDRY_INTAKE_AGENT_ENDPOINT",
    "grouping": "FOUNDRY_GROUPING_AGENT_ENDPOINT",
    "action": "FOUNDRY_ACTION_AGENT_ENDPOINT",
    "investigation": "FOUNDRY_INVESTIGATION_AGENT_ENDPOINT",
    "lifecycle": "FOUNDRY_LIFECYCLE_AGENT_ENDPOINT",
}

DEFAULT_FOUNDRY_AGENT_API_VERSION = "v1"
FOUNDRY_TOKEN_SCOPE = "https://ai.azure.com/.default"


def _env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"')
    return None


def _get_credential() -> TokenCredential:
    tenant_id = _env("AZURE_TENANT_ID", "TENANT_ID")
    client_id = _env("AZURE_CLIENT_ID", "CLIENT_ID")
    client_secret = _env("AZURE_CLIENT_SECRET", "CLIENT_SECRET")
    if tenant_id and client_id and client_secret:
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

    interactive_login = os.getenv("FOUNDRY_INTERACTIVE_LOGIN", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    credential_options = {
        "exclude_interactive_browser_credential": not interactive_login,
    }
    if tenant_id:
        credential_options["interactive_browser_tenant_id"] = tenant_id

    return DefaultAzureCredential(**credential_options)


@lru_cache(maxsize=1)
def _get_project_openai_client():
    endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
    if not endpoint:
        raise RuntimeError("Set FOUNDRY_PROJECT_ENDPOINT.")
    credential = _get_credential()
    project = AIProjectClient(endpoint=endpoint, credential=credential)
    return project.get_openai_client()


@lru_cache(maxsize=8)
def _get_endpoint_openai_client(endpoint: str):
    base_url, query = _split_endpoint_query(endpoint)
    if "api-version" not in query:
        query["api-version"] = _foundry_agent_api_version()

    token_provider = get_bearer_token_provider(_get_credential(), FOUNDRY_TOKEN_SCOPE)
    return OpenAI(base_url=base_url, api_key=token_provider, default_query=query or None)


def _foundry_agent_api_version() -> str:
    return os.getenv("FOUNDRY_AGENT_API_VERSION", DEFAULT_FOUNDRY_AGENT_API_VERSION).strip()


def _split_endpoint_query(endpoint: str) -> tuple[str, dict[str, str]]:
    parsed = urlsplit(endpoint)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    path = parsed.path.rstrip("/")
    if path.endswith("/responses"):
        path = path[: -len("/responses")]
    base_url = urlunsplit(
        (parsed.scheme, parsed.netloc, path, "", parsed.fragment)
    )
    return base_url, query


def _client_for_flow(flow: str | None):
    if flow and (endpoint := os.getenv(AGENT_ENDPOINT_ENV[flow], "").strip()):
        return _get_endpoint_openai_client(endpoint), True
    return _get_project_openai_client(), False


def create_foundry_conversation(flow: str | None = None) -> str:
    """Create a conversation for an interactive Agent 3 chat session."""
    client, _ = _client_for_flow(flow)
    return client.conversations.create().id


def query_foundry_agent(
    agent_name: str,
    prompt: str,
    conversation_id: str | None = None,
    flow: str | None = None,
    timeout: float | None = None,
    debug: bool = False,
) -> str:
    """Invoke a Foundry agent/workflow by reference name and return its text."""
    client, direct_endpoint = _client_for_flow(flow)
    timeout = timeout or float(os.getenv("FOUNDRY_REQUEST_TIMEOUT_SECONDS", "120"))
    created_conversation_id: str | None = None
    request = {
        "input": prompt,
        "timeout": timeout,
    }
    if not direct_endpoint:
        request["extra_body"] = {
            "agent_reference": {"name": agent_name, "type": "agent_reference"}
        }
    if conversation_id:
        request["conversation"] = conversation_id
    else:
        created_conversation_id = client.conversations.create().id
        request["conversation"] = created_conversation_id

    try:
        if debug:
            mode = "direct deployed Agent endpoint" if direct_endpoint else "project endpoint + agent_reference"
            print(
                f"[foundry_client] invoking {agent_name!r} via {mode}; timeout={timeout}s",
                file=sys.stderr,
                flush=True,
            )
        response = client.responses.create(**request)
        if debug:
            print("[foundry_client] Foundry returned a response.", file=sys.stderr, flush=True)
    except APITimeoutError as exc:
        raise RuntimeError(
            f"Foundry request timed out after {timeout} seconds while invoking "
            f"{agent_name!r}. If Preview also hangs, check Agent tool/MCP connectivity "
            "and the Foundry trace for the last running step."
        ) from exc
    except NotFoundError as exc:
        raise RuntimeError(
            f"Foundry returned 404 while invoking {agent_name!r}. If this agent "
            "exists in Foundry, open the agent and confirm that the selected model "
            "deployment exists under Build > Deployments. If you are using the "
            "deployed Agent endpoint, copy the full Responses protocol endpoint "
            f"into {AGENT_ENDPOINT_ENV.get(flow or '', 'FOUNDRY_*_AGENT_ENDPOINT')}. "
            "Then Save the agent and verify it in Preview."
        ) from exc
    except PermissionDeniedError as exc:
        raise RuntimeError(
            "Foundry returned 403. The app authenticated, but that identity is not "
            "allowed to invoke this Foundry agent/project. To avoid browser popups "
            "and match the ZIP, configure AZURE_TENANT_ID, AZURE_CLIENT_ID, and "
            "AZURE_CLIENT_SECRET for a service principal that has access to the "
            "Foundry project, or grant your Azure CLI/managed identity the required "
            "Foundry/Azure AI role."
        ) from exc
    except BadRequestError as exc:
        message = getattr(exc, "message", str(exc))
        if "api version" in message.lower() or "api-version" in message.lower():
            raise RuntimeError(
                "Foundry rejected the API version for the deployed Agent endpoint. "
                f"Set FOUNDRY_AGENT_API_VERSION={DEFAULT_FOUNDRY_AGENT_API_VERSION} "
                "or copy the full Responses endpoint including its api-version query "
                "from the Agent configuration page. Do not use "
                "AZURE_OPENAI_API_VERSION for FOUNDRY_*_AGENT_ENDPOINT calls."
            ) from exc
        raise
    finally:
        if created_conversation_id:
            try:
                client.conversations.delete(conversation_id=created_conversation_id)
            except Exception:
                if debug:
                    print(
                        f"[foundry_client] unable to delete temporary conversation {created_conversation_id}",
                        file=sys.stderr,
                        flush=True,
                    )

    messages: list[dict[str, str | None]] = []

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue

        agent_reference = getattr(
            item,
            "agent_reference",
            None,
        )

        agent_name_from_item = None

        if isinstance(agent_reference, dict):
            agent_name_from_item = agent_reference.get("name")

        elif agent_reference is not None:
            agent_name_from_item = getattr(
                agent_reference,
                "name",
                None,
            )

        text_parts: list[str] = []

        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) != "output_text":
                continue

            text = getattr(
                content,
                "text",
                "",
            )

            if text:
                text_parts.append(text)

        text = "".join(text_parts).strip()

        if text:
            messages.append(
                {
                    "agent": agent_name_from_item,
                    "text": text,
                }
            )


    # --------------------------------------------------
    # DEBUG: SEE EXACT WORKFLOW MESSAGES
    # --------------------------------------------------

    if debug:
        print(
            "\n========== FOUNDRY WORKFLOW MESSAGES ==========",
            file=sys.stderr,
        )

        for index, message in enumerate(messages, start=1):
            print(
                f"\nMessage {index}",
                file=sys.stderr,
            )

            print(
                f"Agent: {message['agent']}",
                file=sys.stderr,
            )

            print(
                message["text"],
                file=sys.stderr,
            )

        print(
            "\n================================================",
            file=sys.stderr,
        )


    # --------------------------------------------------
    # INTAKE WORKFLOW:
    # PREFER AGENT2 FINAL OUTPUT
    # --------------------------------------------------

    if flow == "intake":

        # First preference:
        # explicitly select Agent2 output.

        for message in reversed(messages):

            agent_name = (
                message["agent"] or ""
            ).lower()

            if agent_name == "agent2":
                return str(message["text"])


        # Second preference:
        # select JSON containing grouping/assignment fields.

        for message in reversed(messages):

            text = str(message["text"])

            try:
                parsed = json.loads(
                    text.strip()
                )

            except json.JSONDecodeError:
                continue

            if not isinstance(parsed, dict):
                continue

            if (
                "group_id" in parsed
                or "employee_id" in parsed
                or "assignment_status" in parsed
            ):
                return text


    # --------------------------------------------------
    # OTHER FLOWS / FALLBACK:
    # RETURN LAST AGENT MESSAGE
    # --------------------------------------------------

    if messages:
        return str(messages[-1]["text"])


    # --------------------------------------------------
    # FINAL FALLBACK
    # --------------------------------------------------

    return getattr(
        response,
        "output_text",
        "",
    ) or ""


def _build_prompt(flow: str, args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if flow == "action":
        return json.dumps(
            {"group_id": args.group_id, "question": args.query}
        )
    if flow == "investigation":
        return json.dumps(
            {"employee_id": args.employee_id, "group_id": args.group_id, "query": args.query}
        )
    if flow == "lifecycle":
        return json.dumps(
            {
                "group_id": args.group_id,
                "new_status": args.status,
                "reason": args.reason or "",
            }
        )
    raise SystemExit("For intake, supply --prompt with the raw incident text.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Call one of the ITIL Foundry entry points.")
    parser.add_argument("flow", choices=AGENT_ENV)
    parser.add_argument("--prompt")
    parser.add_argument("--employee-id", type=int)
    parser.add_argument("--group-id", type=int)
    parser.add_argument("--query", default="Summarize this group and recommend diagnostic steps.")
    parser.add_argument("--status", choices=("WIP", "Pause", "Close", "Cancel"))
    parser.add_argument("--reason")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.flow == "action" and args.group_id is None:
        parser.error("action requires --group-id")
    if args.flow == "investigation" and (args.employee_id is None or args.group_id is None):
        parser.error("investigation requires --employee-id and --group-id")
    if args.flow == "lifecycle" and (args.group_id is None or args.status is None):
        parser.error("lifecycle requires --group-id and --status")

    agent_name = os.getenv(AGENT_ENV[args.flow])
    if not agent_name:
        raise RuntimeError(f"Set {AGENT_ENV[args.flow]}.")
    print(
        query_foundry_agent(
            agent_name,
            _build_prompt(args.flow, args),
            flow=args.flow,
            timeout=args.timeout,
            debug=args.debug,
        )
    )


if __name__ == "__main__":
    main()
