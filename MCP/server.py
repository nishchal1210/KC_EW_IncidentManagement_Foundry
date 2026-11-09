import logging
import os
from contextlib import contextmanager
from datetime import date, datetime,timezone
from decimal import Decimal
from functools import lru_cache
from typing import Any

import psycopg2
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import OpenAI
import math


load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("incident-mcp")

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", os.getenv("PORT", "8000")))

mcp = FastMCP(
    "Incident Management PostgreSQL MCP",
    host=MCP_HOST,
    port=MCP_PORT,
)

SCHEMA = os.getenv("POSTGRES_SCHEMA", "ew-kc-schema")
INCIDENT_TABLE = os.getenv("INCIDENT_TABLE", "t_incidents")
ASSIGNMENT_TABLE = os.getenv("ASSIGNMENT_TABLE", "t_group_assignment")
ASSIGNMENT_VIEW = os.getenv("ASSIGNMENT_VIEW", "v_group_assignment")
EMPLOYEE_SKILL_VIEW = os.getenv("EMPLOYEE_SKILL_VIEW", "vEmployeeSkillMapping")


ACTIVE_ASSIGNMENT_STATUSES = ("Assigned", "WIP", "Pause")
TERMINAL_STATUSES = ("Closed", "Cancelled")
STATUS_TRANSITIONS = {
    "active": ("Active", "Assigned", False),
    "assigned": ("Active", "Assigned", False),
    "wip": ("WIP", "WIP", False),
    "pause": ("Pause", "Pause", False),
    "paused": ("Pause", "Pause", False),
    "close": ("Closed", "Closed", True),
    "closed": ("Closed", "Closed", True),
    "cancel": ("Cancelled", "Cancelled", True),
    "cancelled": ("Cancelled", "Cancelled", True),
    "canceled": ("Cancelled", "Cancelled", True),
}

_pool: pool.ThreadedConnectionPool | None = None
_search_client: SearchClient | None = None
_embedding_client: OpenAI | None = None


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"')
    return default

AZURE_SEARCH_ENDPOINT = _env(
    "AZURE_SEARCH_ENDPOINT"
)

AZURE_SEARCH_API_KEY = _env(
    "AZURE_SEARCH_API_KEY"
)

AZURE_SEARCH_INDEX_NAME = _env(
    "AZURE_SEARCH_INDEX_NAME",
    default="ew-closed-incident-resolution-index",
)

EMBEDDING_ENDPOINT = _env(
    "EMBEDDING_ENDPOINT"
)

EMBEDDING_API_KEY = _env(
    "EMBEDDING_API_KEY"
)

EMBEDDING_DEPLOYMENT_NAME = _env(
    "EMBEDDING_DEPLOYMENT_NAME"
)
def _get_search_client() -> SearchClient:
    global _search_client

    if _search_client is None:
        if not AZURE_SEARCH_ENDPOINT:
            raise RuntimeError(
                "AZURE_SEARCH_ENDPOINT is not configured."
            )

        if not AZURE_SEARCH_API_KEY:
            raise RuntimeError(
                "AZURE_SEARCH_API_KEY is not configured."
            )

        if not AZURE_SEARCH_INDEX_NAME:
            raise RuntimeError(
                "AZURE_SEARCH_INDEX_NAME is not configured."
            )

        _search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(
                AZURE_SEARCH_API_KEY
            ),
        )

    return _search_client


def _get_embedding_client() -> OpenAI:
    global _embedding_client

    if _embedding_client is None:
        if not EMBEDDING_ENDPOINT:
            raise RuntimeError(
                "EMBEDDING_ENDPOINT is not configured."
            )

        if not EMBEDDING_API_KEY:
            raise RuntimeError(
                "EMBEDDING_API_KEY is not configured."
            )

        endpoint = EMBEDDING_ENDPOINT.rstrip("/")

        _embedding_client = OpenAI(
            base_url=f"{endpoint}/openai/v1/",
            api_key=EMBEDDING_API_KEY,
        )

    return _embedding_client

def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value

def _cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Cannot compare vectors with different dimensions."
        )

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    norm_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (
        norm_a * norm_b
    )


def _generate_embedding(text: str) -> list[float]:
    text = _clean_text(text)

    if not text:
        raise ValueError(
            "Cannot generate an embedding for empty text."
        )

    if not EMBEDDING_DEPLOYMENT_NAME:
        raise RuntimeError(
            "EMBEDDING_DEPLOYMENT_NAME is not configured."
        )

    response = _get_embedding_client().embeddings.create(
        model=EMBEDDING_DEPLOYMENT_NAME,
        input=text,
    )

    vector = response.data[0].embedding

    if len(vector) != 1536:
        raise ValueError(
            "Embedding dimension mismatch. "
            f"Expected 1536, received {len(vector)}."
        )

    return vector

def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(
            minconn=int(os.getenv("POSTGRES_POOL_MIN", "1")),
            maxconn=int(os.getenv("POSTGRES_POOL_MAX", "8")),
            host=_env("POSTGRES_HOST", "DB_HOST"),
            dbname=_env("POSTGRES_DB", "DB_NAME"),
            user=_env("POSTGRES_USER", "DB_USER"),
            password=_env("POSTGRES_PASSWORD", "DB_PASS"),
            port=_env("POSTGRES_PORT", "DB_PORT", default="5432"),
            sslmode=_env("POSTGRES_SSLMODE", "DB_SSLMODE", default="require"),
            connect_timeout=int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10")),
            application_name=os.getenv("POSTGRES_APP_NAME", "incident-mcp"),
        )
    return _pool


@contextmanager
def db_connection():
    conn = _get_pool().getconn()
    try:
        yield conn
    finally:
        _get_pool().putconn(conn)


def _qualified(name: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(SCHEMA), sql.Identifier(name))


def _fetch_all(query: sql.Composable, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    return [{key: _json_safe(value) for key, value in row.items()} for row in rows]


def _fetch_one(query: sql.Composable, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    rows = _fetch_all(query, params)
    return rows[0] if rows else None


@lru_cache(maxsize=64)
def _columns(table_name: str) -> frozenset[str]:
    query = sql.SQL(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        """
    )
    rows = _fetch_all(query, (SCHEMA, table_name))
    return frozenset(row["column_name"] for row in rows)


@lru_cache(maxsize=64)
def _column_metadata(table_name: str, column_name: str) -> dict[str, Any]:
    query = sql.SQL(
        """
        SELECT column_default, identity_generation, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s AND column_name = %s
        """
    )
    return _fetch_one(query, (SCHEMA, table_name, column_name)) or {}


@lru_cache(maxsize=64)
def _relation_exists(name: str) -> bool:
    query = sql.SQL(
        """
        SELECT 1 AS exists
        FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
        UNION ALL
        SELECT 1 AS exists
        FROM information_schema.views
        WHERE table_schema = %s AND table_name = %s
        LIMIT 1
        """
    )
    return _fetch_one(query, (SCHEMA, name, SCHEMA, name)) is not None


def _first_column(table_name: str, candidates: tuple[str, ...]) -> str | None:
    available = _columns(table_name)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _assignment_time_column() -> str:
    column = _first_column(ASSIGNMENT_TABLE, ("datetime", "assigned_datetime"))
    if not column:
        raise RuntimeError(
            f"{SCHEMA}.{ASSIGNMENT_TABLE} must contain datetime or assigned_datetime."
        )
    return column


def _assignment_comment_column() -> str | None:
    return _first_column(ASSIGNMENT_TABLE, ("comment", "reason"))


def _assignment_id_needs_value() -> bool:
    if "id" not in _columns(ASSIGNMENT_TABLE):
        return False
    metadata = _column_metadata(ASSIGNMENT_TABLE, "id")
    if metadata.get("column_default") or metadata.get("identity_generation"):
        return False
    return metadata.get("is_nullable") == "NO"


def _assignment_sort_sql(alias: str = "tga") -> sql.Composed:
    time_column = _assignment_time_column()
    pieces = [sql.SQL("{}.{} DESC NULLS LAST").format(sql.Identifier(alias), sql.Identifier(time_column))]
    if "id" in _columns(ASSIGNMENT_TABLE):
        pieces.append(sql.SQL("{}.{} DESC").format(sql.Identifier(alias), sql.Identifier("id")))
    return sql.SQL(", ").join(pieces)


def _latest_assignment_cte() -> sql.SQL:
    sort_sql = _assignment_sort_sql("tga")
    return sql.SQL(
        """
        WITH latest_assignment AS (
            SELECT DISTINCT ON (tga.group_id)
                tga.employee_id,
                tga.group_id,
                tga.status,
                tga.{time_column} AS datetime
            FROM {assignment_table} tga
            ORDER BY tga.group_id, {sort_sql}
        )
        """
    ).format(
        assignment_table=_qualified(ASSIGNMENT_TABLE),
        time_column=sql.Identifier(_assignment_time_column()),
        sort_sql=sort_sql,
    )


def _next_group_id(cur) -> int:
    lock_name = f"{SCHEMA}.{INCIDENT_TABLE}.group_id"
    cur.execute(sql.SQL("SELECT pg_advisory_xact_lock(hashtext(%s))"), (lock_name,))
    cur.execute(
        sql.SQL("SELECT COALESCE(MAX(group_id), 0) + 1 AS group_id FROM {table}").format(
            table=_qualified(INCIDENT_TABLE)
        )
    )
    return int(cur.fetchone()["group_id"])


def _next_assignment_id(cur) -> int:
    lock_name = f"{SCHEMA}.{ASSIGNMENT_TABLE}.id"
    cur.execute(sql.SQL("SELECT pg_advisory_xact_lock(hashtext(%s))"), (lock_name,))
    cur.execute(
        sql.SQL("SELECT COALESCE(MAX(id), 0) + 1 AS id FROM {table}").format(
            table=_qualified(ASSIGNMENT_TABLE)
        )
    )
    return int(cur.fetchone()["id"])


def _insert_assignment_event(
    cur,
    *,
    employee_id: int | None,
    group_id: int,
    status: str,
    comment: str = "",
) -> dict[str, Any]:
    time_column = _assignment_time_column()
    comment_column = _assignment_comment_column()
    columns = ["employee_id", "group_id", time_column, "status"]
    values: list[Any] = [employee_id, int(group_id), datetime.utcnow(), _clean_text(status)]

    if comment_column:
        columns.append(comment_column)
        values.append(_clean_text(comment))

    if _assignment_id_needs_value():
        columns.insert(0, "id")
        values.insert(0, _next_assignment_id(cur))

    returning = [
        sql.SQL("employee_id"),
        sql.SQL("group_id"),
        sql.SQL("{} AS datetime").format(sql.Identifier(time_column)),
        sql.SQL("status"),
    ]
    if "id" in _columns(ASSIGNMENT_TABLE):
        returning.insert(0, sql.SQL("id"))
    if comment_column:
        returning.append(sql.SQL("{} AS comment").format(sql.Identifier(comment_column)))

    query = sql.SQL(
        """
        INSERT INTO {table} ({columns})
        VALUES ({placeholders})
        RETURNING {returning}
        """
    ).format(
        table=_qualified(ASSIGNMENT_TABLE),
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in columns),
        placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        returning=sql.SQL(", ").join(returning),
    )
    cur.execute(query, tuple(values))
    row = cur.fetchone()
    return {key: _json_safe(value) for key, value in row.items()}


def _normalize_status(new_status: str) -> tuple[str, str, bool]:
    key = _clean_text(new_status).lower().replace("-", "").replace("_", "").replace(" ", "")
    transition = STATUS_TRANSITIONS.get(key)
    if not transition:
        allowed = "Active, WIP, Pause, Close, Cancel"
        raise ValueError(f"Invalid status {new_status!r}. Use one of: {allowed}.")
    return transition
def _get_closed_group_case(
    group_id: int,
) -> dict[str, Any]:
    group_id = int(group_id)

    comment_column = _assignment_comment_column()

    if not comment_column:
        raise RuntimeError(
            f"{SCHEMA}.{ASSIGNMENT_TABLE} must contain "
            "a comment or reason column."
        )

    time_column = _assignment_time_column()

    query = sql.SQL(
        """
        WITH latest_closed_event AS
        (
            SELECT
                tga.employee_id,
                tga.group_id,
                tga.{time_column} AS resolved_at,
                tga.{comment_column} AS resolution_comment
            FROM {assignment_table} tga
            WHERE tga.group_id = %s
              AND tga.status IN ('Close', 'Closed')
            ORDER BY
                tga.{time_column} DESC NULLS LAST,
                {id_sort}
            LIMIT 1
        )
        SELECT
            ti.group_id,

            ARRAY_AGG(
                DISTINCT ti.hw_system
            ) FILTER (
                WHERE NULLIF(TRIM(ti.hw_system), '') IS NOT NULL
            ) AS hw_systems,

            ARRAY_AGG(
                DISTINCT ti.sw_system
            ) FILTER (
                WHERE NULLIF(TRIM(ti.sw_system), '') IS NOT NULL
            ) AS sw_systems,

            ARRAY_AGG(
                DISTINCT ti.issue
            ) FILTER (
                WHERE NULLIF(TRIM(ti.issue), '') IS NOT NULL
            ) AS issues,

            lce.resolution_comment,
            lce.employee_id AS resolved_by,
            lce.resolved_at

        FROM {incident_table} ti

        JOIN latest_closed_event lce
            ON lce.group_id = ti.group_id

        WHERE ti.group_id = %s
          AND ti.status = 'Closed'

        GROUP BY
            ti.group_id,
            lce.resolution_comment,
            lce.employee_id,
            lce.resolved_at
        """
    ).format(
        incident_table=_qualified(INCIDENT_TABLE),
        assignment_table=_qualified(ASSIGNMENT_TABLE),
        time_column=sql.Identifier(time_column),
        comment_column=sql.Identifier(comment_column),
        id_sort=(
            sql.SQL("tga.id DESC")
            if "id" in _columns(ASSIGNMENT_TABLE)
            else sql.SQL("tga.group_id DESC")
        ),
    )

    case = _fetch_one(
        query,
        (
            group_id,
            group_id,
        ),
    )

    if not case:
        raise ValueError(
            f"No closed resolution data found for group {group_id}."
        )

    resolution_comment = _clean_text(
        case.get("resolution_comment")
    )

    if not resolution_comment:
        raise ValueError(
            f"Closed group {group_id} has no resolution comment."
        )

    return case

def _build_issue_embedding_text(
    case: dict[str, Any],
) -> str:
    hw_systems = case.get("hw_systems") or []
    sw_systems = case.get("sw_systems") or []
    issues = case.get("issues") or []

    hw_text = ", ".join(
        _clean_text(value)
        for value in hw_systems
        if _clean_text(value)
    )

    sw_text = ", ".join(
        _clean_text(value)
        for value in sw_systems
        if _clean_text(value)
    )

    issue_text = "\n".join(
        f"- {_clean_text(issue)}"
        for issue in issues
        if _clean_text(issue)
    )

    return f"""
Hardware systems:
{hw_text}

Software systems:
{sw_text}

Historical incident issues:
{issue_text}
""".strip()

def _index_closed_group(
    group_id: int,
) -> dict[str, Any]:
    case = _get_closed_group_case(
        group_id
    )

    embedding_text = _build_issue_embedding_text(
        case
    )

    issue_vector = _generate_embedding(
        embedding_text
    )

    resolved_at = case["resolved_at"]

    if isinstance(resolved_at, str):
        resolved_at = datetime.fromisoformat(
            resolved_at
        )

    if resolved_at.tzinfo is None:
        resolved_at = resolved_at.replace(
            tzinfo=timezone.utc
        )

    issues = [
        _clean_text(issue)
        for issue in (case.get("issues") or [])
        if _clean_text(issue)
    ]

    document = {
        "id": f"group-{int(case['group_id'])}",

        "group_id": int(
            case["group_id"]
        ),

        "status": "Closed",

        "hw_systems": [
            _clean_text(value)
            for value in (
                case.get("hw_systems") or []
            )
            if _clean_text(value)
        ],

        "sw_systems": [
            _clean_text(value)
            for value in (
                case.get("sw_systems") or []
            )
            if _clean_text(value)
        ],

        "issues": issues,

        "combined_issue_text": " ".join(
            issues
        ),

        "resolution_comment": _clean_text(
            case["resolution_comment"]
        ),

        "resolved_by": (
            int(case["resolved_by"])
            if case.get("resolved_by") is not None
            else None
        ),

        "resolved_at": resolved_at.isoformat(),

        "issue_vector": issue_vector,
    }

    result = (
        _get_search_client()
        .merge_or_upload_documents(
            documents=[document]
        )
    )

    if not result:
        raise RuntimeError(
            f"Azure AI Search returned no indexing result "
            f"for group {group_id}."
        )

    indexing_result = result[0]

    if not indexing_result.succeeded:
        raise RuntimeError(
            f"Unable to index closed group {group_id}: "
            f"{indexing_result.error_message}"
        )

    logger.info(
        "Indexed closed group %s into Azure AI Search.",
        group_id,
    )

    return {
        "group_id": group_id,
        "search_document_id": document["id"],
        "indexed": True,
    }

@mcp.tool()
def health() -> dict[str, Any]:
    """Check that the MCP server, PostgreSQL connection, and expected schema are available."""
    try:
        row = _fetch_one(sql.SQL("SELECT 1 AS ok"))
        return {
    "status": "running",
    "postgres": (
        "ok"
        if row and row["ok"] == 1
        else "unknown"
    ),
    "schema": SCHEMA,
    "assignment_columns": sorted(
        _columns(ASSIGNMENT_TABLE)
    ),
    "assignment_view_available": _relation_exists(
        ASSIGNMENT_VIEW
    ),
    "azure_search_index": AZURE_SEARCH_INDEX_NAME,
    "azure_search_configured": bool(
        AZURE_SEARCH_ENDPOINT
        and AZURE_SEARCH_API_KEY
        and AZURE_SEARCH_INDEX_NAME
    ),
    "embedding_configured": bool(
        EMBEDDING_ENDPOINT
        and EMBEDDING_API_KEY
        and EMBEDDING_DEPLOYMENT_NAME
    ),
}
    except Exception as exc:
        logger.exception("Health check failed")
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def find_candidate_groups(hw_system: str = "", sw_system: str = "") -> list[dict[str, Any]]:
    """Return non-terminal candidate groups that match hardware and software systems."""
    latest_cte = _latest_assignment_cte()
    query = latest_cte + sql.SQL(
        """
        SELECT
            ti.group_id,
            ARRAY_AGG(ti.issue ORDER BY ti."timestamp" DESC NULLS LAST) AS issues,
            ARRAY_AGG(ti.incident_number ORDER BY ti."timestamp" DESC NULLS LAST) AS incident_numbers,
            COUNT(*) AS incident_count,
            MAX(ti."timestamp") AS latest_timestamp,
            la.status AS assignment_status,
            la.employee_id AS employee_id
        FROM {incident_table} ti
        LEFT JOIN latest_assignment la ON la.group_id = ti.group_id
        WHERE ti.status NOT IN ('Closed', 'Cancelled')
          AND COALESCE(la.status, 'Open') NOT IN ('Closed', 'Cancelled')
          AND COALESCE(ti.hw_system, '') = %s
          AND COALESCE(ti.sw_system, '') = %s
          AND ti.group_id IS NOT NULL
        GROUP BY ti.group_id, la.status, la.employee_id
        ORDER BY MAX(ti."timestamp") DESC NULLS LAST, ti.group_id ASC
        """
    ).format(incident_table=_qualified(INCIDENT_TABLE))
    return _fetch_all(query, (_clean_text(hw_system), _clean_text(sw_system)))


@mcp.tool()
def create_group() -> dict[str, int]:
    """Create the next group id using max(group_id)+1 under an advisory lock."""
    with db_connection() as conn:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                group_id = _next_group_id(cur)
            conn.commit()
            return {"group_id": group_id}
        except Exception:
            conn.rollback()
            logger.exception("Unable to create group id")
            raise



@mcp.tool()
def insert_incident(
    timestamp: str,
    hw_system: str,
    sw_system: str,
    issue: str,
    source: str,
    severity: str,
    group_id: int = 0,
    status: str = "Active",
) -> dict[str, Any]:
    """
    Insert an incident into t_incidents.

    PostgreSQL automatically generates incident_number.

    If group_id <= 0:
        - create a new group_id
        - insert the incident
        - append an Open event to t_group_assignment

    If an existing group_id is supplied:
        - insert the incident into that existing group
        - do not create another Open assignment event
    """

    insert_query = sql.SQL(
        """
        INSERT INTO {incident_table}
        (
            "timestamp",
            hw_system,
            sw_system,
            issue,
            source,
            severity,
            status,
            group_id
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        RETURNING
            incident_number,
            group_id,
            status
        """
    ).format(
        incident_table=_qualified(INCIDENT_TABLE)
    )

    with db_connection() as conn:
        try:
            with conn.cursor(
                cursor_factory=RealDictCursor
            ) as cur:

                final_group_id = int(group_id or 0)

                created_new_group = final_group_id <= 0

                # -----------------------------------------
                # CREATE NEW GROUP ID
                # -----------------------------------------

                if created_new_group:
                    final_group_id = _next_group_id(cur)

                # -----------------------------------------
                # INSERT INCIDENT
                # -----------------------------------------

                params = (
                    _clean_text(timestamp),
                    _clean_text(hw_system),
                    _clean_text(sw_system),
                    _clean_text(issue),
                    _clean_text(source),
                    _clean_text(severity),
                    _clean_text(status) or "Active",
                    final_group_id,
                )

                cur.execute(
                    insert_query,
                    params,
                )

                incident_row = cur.fetchone()

                # -----------------------------------------
                # NEW GROUP -> CREATE OPEN EVENT
                # -----------------------------------------

                open_assignment = None

                if created_new_group:
                    open_assignment = _insert_assignment_event(
                        cur,
                        employee_id=None,
                        group_id=final_group_id,
                        status="Open",
                        comment="Group created by intake workflow.",
                    )

            conn.commit()

            return {
                "incident_number": _json_safe(
                    incident_row["incident_number"]
                ),
                "group_id": int(
                    incident_row["group_id"]
                ),
                "created_new_group": created_new_group,
                "status": _json_safe(
                    incident_row["status"]
                ),
                "open_assignment": open_assignment,
            }

        except Exception:
            conn.rollback()

            logger.exception(
                "Unable to insert incident"
            )

            raise


@mcp.tool()
def get_next_employee(sw_system: str, exclude_employee_id: int = 0) -> dict[str, Any]:
    """Return ranked software employee recommendations by maturity, then current tickets."""
    latest_cte = _latest_assignment_cte()
    query = latest_cte + sql.SQL(
        """
        , candidate_skills AS (
            SELECT
                esm.employee_id,
                MAX(esm.employee_name) AS employee_name,
                MAX(COALESCE(esm.maturity, 0)) AS skill_maturity
            FROM {skill_view} esm
            WHERE esm.type = 'Software'
              AND esm.system = %s
              AND (%s = 0 OR esm.employee_id <> %s)
            GROUP BY esm.employee_id
        ),
        active_workload AS (
            SELECT
                la.employee_id,
                COUNT(la.group_id) FILTER (
                    WHERE la.status IN ('Assigned', 'WIP', 'Pause')
                ) AS current_tickets
            FROM latest_assignment la
            WHERE la.employee_id IS NOT NULL
            GROUP BY la.employee_id
        )
        SELECT
            cs.employee_id,
            cs.employee_name,
            cs.skill_maturity,
            COALESCE(aw.current_tickets, 0) AS current_tickets
        FROM candidate_skills cs
        LEFT JOIN active_workload aw ON aw.employee_id = cs.employee_id
        ORDER BY cs.skill_maturity DESC,
                 COALESCE(aw.current_tickets, 0) ASC,
                 cs.employee_id ASC
        LIMIT 3
        """
    ).format(skill_view=_qualified(EMPLOYEE_SKILL_VIEW))

    excluded = int(exclude_employee_id or 0)
    rows = _fetch_all(query, (_clean_text(sw_system), excluded, excluded))
    if not rows:
        return {
            "employee_id": None,
            "employee_name": None,
            "skill_maturity": None,
            "current_tickets": None,
            "active_tasks": None,
            "top_3": [],
            "selection_policy": (
                "Eligible software employees are ranked by SkillMaturity descending, "
                "then CurrentTickets ascending, then employee_id ascending."
            ),
        }

    top_3 = [
        {
            "priority": index + 1,
            "EmployeeID": row["employee_id"],
            "Name": row.get("employee_name"),
            "SkillMaturity": row.get("skill_maturity"),
            "CurrentTickets": row.get("current_tickets"),
            "reasoning": (
                "Selected from employees matching the incident software system. "
                f"SkillMaturity={row.get('skill_maturity')}; "
                f"CurrentTickets={row.get('current_tickets')}."
            ),
        }
        for index, row in enumerate(rows)
    ]
    best = rows[0]
    return {
        "employee_id": best["employee_id"],
        "employee_name": best.get("employee_name"),
        "skill_maturity": best.get("skill_maturity"),
        "current_tickets": best.get("current_tickets"),
        "active_tasks": best.get("current_tickets"),
        "top_3": top_3,
        "selection_policy": (
            "Eligible software employees are ranked by SkillMaturity descending, "
            "then CurrentTickets ascending, then employee_id ascending."
        ),
    }


@mcp.tool()
def get_employee_groups(employee_id: int, include_completed: bool = False) -> list[dict[str, Any]]:
    """List latest groups assigned to an employee."""
    latest_cte = _latest_assignment_cte()
    query = latest_cte + sql.SQL(
        """
        SELECT
            la.group_id,
            la.status,
            la.datetime AS assigned_datetime,
            NULL::timestamp AS completed_datetime,
            COUNT(DISTINCT ti.incident_number) AS incident_count
        FROM latest_assignment la
        LEFT JOIN {incident_table} ti ON ti.group_id = la.group_id
        WHERE la.employee_id = %s
          AND (%s OR la.status IN ('Assigned', 'WIP', 'Pause'))
        GROUP BY la.group_id, la.status, la.datetime
        ORDER BY la.datetime DESC NULLS LAST, la.group_id
        """
    ).format(incident_table=_qualified(INCIDENT_TABLE))
    return _fetch_all(query, (int(employee_id), bool(include_completed)))


@mcp.tool()
def get_group_incidents(group_id: int) -> list[dict[str, Any]]:
    """Return every incident in a group, newest first."""
    query = sql.SQL(
        """
        SELECT
            incident_number,
            "timestamp",
            hw_system,
            sw_system,
            issue,
            source,
            severity,
            status,
            group_id
        FROM {incident_table}
        WHERE group_id = %s
        ORDER BY "timestamp" DESC NULLS LAST, incident_number
        """
    ).format(incident_table=_qualified(INCIDENT_TABLE))
    return _fetch_all(query, (int(group_id),))


@mcp.tool()
def get_group_summary(group_id: int) -> dict[str, Any]:
    """Return aggregated systems, issues, latest assignment, statuses, and time range for a group."""
    latest_cte = _latest_assignment_cte()
    query = latest_cte + sql.SQL(
        """
        SELECT
            ti.group_id,
            COUNT(*) AS total_incidents,
            ARRAY_AGG(DISTINCT ti.hw_system) FILTER (WHERE ti.hw_system IS NOT NULL) AS hw_systems,
            ARRAY_AGG(DISTINCT ti.sw_system) FILTER (WHERE ti.sw_system IS NOT NULL) AS sw_systems,
            ARRAY_AGG(DISTINCT ti.status) FILTER (WHERE ti.status IS NOT NULL) AS incident_statuses,
            ARRAY_AGG(DISTINCT ti.issue) FILTER (WHERE ti.issue IS NOT NULL) AS issues,
            MIN(ti."timestamp") AS first_seen,
            MAX(ti."timestamp") AS last_seen,
            la.employee_id AS employee_id,
            la.status AS assignment_status,
            la.datetime AS assignment_datetime
        FROM {incident_table} ti
        LEFT JOIN latest_assignment la ON la.group_id = ti.group_id
        WHERE ti.group_id = %s
        GROUP BY ti.group_id, la.employee_id, la.status, la.datetime
        """
    ).format(incident_table=_qualified(INCIDENT_TABLE))
    return _fetch_one(query, (int(group_id),)) or {}


@mcp.tool()
def get_group_assignment(group_id: int) -> dict[str, Any]:
    """Return the latest active assignment event for a group, if one exists."""
    latest_cte = _latest_assignment_cte()
    query = latest_cte + sql.SQL(
        """
        SELECT
            employee_id,
            group_id,
            datetime AS assigned_datetime,
            NULL::timestamp AS completed_datetime,
            status
        FROM latest_assignment
        WHERE group_id = %s
          AND status IN ('Assigned', 'WIP', 'Pause')
        LIMIT 1
        """
    )
    return _fetch_one(query, (int(group_id),)) or {}


@mcp.tool()
def assign_employee(employee_id: int, group_id: int, reason: str = "") -> dict[str, Any]:
    """Append an Assigned event for a group."""
    with db_connection() as conn:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                row = _insert_assignment_event(
                    cur,
                    employee_id=int(employee_id),
                    group_id=int(group_id),
                    status="Assigned",
                    comment=reason or "Assigned by intake workflow.",
                )
            conn.commit()
            return row
        except Exception:
            conn.rollback()
            logger.exception("Unable to assign employee %s to group %s", employee_id, group_id)
            raise


@mcp.tool()
def update_group_status(group_id: int, new_status: str, reason: str = "") -> dict[str, Any]:
    """Append a lifecycle event and update incident status for the whole group."""
    incident_status, assignment_status, _terminal = _normalize_status(new_status)
    group_id = int(group_id)
    reason = _clean_text(reason)

    latest_cte = _latest_assignment_cte()
    previous_query = latest_cte + sql.SQL(
        """
        SELECT ti.status AS incident_status, la.status AS assignment_status, la.employee_id
        FROM {incident_table} ti
        LEFT JOIN latest_assignment la ON la.group_id = ti.group_id
        WHERE ti.group_id = %s
        ORDER BY ti."timestamp" DESC NULLS LAST
        LIMIT 1
        """
    ).format(incident_table=_qualified(INCIDENT_TABLE))
    incident_query = sql.SQL(
        "UPDATE {table} SET status = %s WHERE group_id = %s"
    ).format(table=_qualified(INCIDENT_TABLE))

    with db_connection() as conn:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(previous_query, (group_id,))
                previous = cur.fetchone()
                if not previous:
                    raise ValueError(f"Group {group_id} does not exist.")
                cur.execute(incident_query, (incident_status, group_id))
                incidents_updated = cur.rowcount
                event = _insert_assignment_event(
                    cur,
                    employee_id=previous["employee_id"],
                    group_id=group_id,
                    status=assignment_status,
                    comment=reason,
                )
            conn.commit()
                        

            search_indexing = None

            if incident_status == "Closed":
                try:
                    search_indexing = _index_closed_group(
                        group_id
                    )

                except Exception as exc:
                    logger.exception(
                        "Group %s was closed successfully, "
                        "but Azure AI Search indexing failed.",
                        group_id,
                    )

                    search_indexing = {
                        "indexed": False,
                        "error": str(exc),
                    }

            return {
                "group_id": group_id,
                "previous_status": (
                    previous["assignment_status"]
                    or previous["incident_status"]
                ),
                "new_status": incident_status,
                "assignment_status": assignment_status,
                "employee_id": previous["employee_id"],
                "reason": reason,
                "incidents_updated": incidents_updated,
                "assignment_event": event,
                "search_indexing": search_indexing,
            }
        except Exception:
            conn.rollback()
            logger.exception("Unable to update status for group %s", group_id)
            raise
@mcp.tool()
def index_closed_group(
    group_id: int,
) -> dict[str, Any]:
    """
    Build and upsert one Closed group into the
    Azure AI Search resolution index.
    """

    return _index_closed_group(
        int(group_id)
    )

@mcp.tool()
def search_similar_closed_resolutions(
    group_id: int,
    question: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Find strongly similar previously Closed groups.

    Azure AI Search first retrieves candidate cases
    using hybrid text + vector search.

    Candidate cases are then validated using cosine
    similarity between the current incident context
    and the historical incident context.

    Weak historical matches are removed before being
    returned to the agent.
    """

    group_id = int(group_id)

    top_k = max(
        1,
        min(int(top_k), 10),
    )

    # ---------------------------------------------
    # CURRENT GROUP CONTEXT
    # ---------------------------------------------

    current_group = get_group_summary(
        group_id
    )

    if not current_group:
        raise ValueError(
            f"Group {group_id} does not exist."
        )

    hw_systems = (
        current_group.get("hw_systems")
        or []
    )

    sw_systems = (
        current_group.get("sw_systems")
        or []
    )

    issues = (
        current_group.get("issues")
        or []
    )

    query_case = {
        "hw_systems": hw_systems,
        "sw_systems": sw_systems,
        "issues": issues,
    }

    # ---------------------------------------------
    # BUILD ISSUE CONTEXT
    # ---------------------------------------------

    issue_context = _build_issue_embedding_text(
        query_case
    )

    # IMPORTANT:
    # Vector similarity should focus on the actual
    # incident problem, not the engineer's generic
    # question such as "How do I resolve this?"
    query_vector = _generate_embedding(
        issue_context
    )

    question = _clean_text(
        question
    )

    search_text = issue_context

    if question:
        search_text = (
            f"{issue_context}\n\n"
            f"Engineer question:\n"
            f"{question}"
        )

    # ---------------------------------------------
    # AZURE AI SEARCH CANDIDATE RETRIEVAL
    # ---------------------------------------------

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=50,
        fields="issue_vector",
    )

    # Retrieve more candidates than final top_k.
    # We will filter weak matches ourselves.
    candidate_count = min(
        max(top_k * 3, 10),
        30,
    )

    results = _get_search_client().search(
        search_text=search_text,
        vector_queries=[
            vector_query
        ],
        filter=(
            f"status eq 'Closed' "
            f"and group_id ne {group_id}"
        ),
        scoring_profile="recentClosedBoost",
        top=candidate_count,
        select=[
            "group_id",
            "hw_systems",
            "sw_systems",
            "issues",
            "combined_issue_text",
            "resolution_comment",
            "resolved_by",
            "resolved_at",
        ],
    )

    # ---------------------------------------------
    # VALIDATE CANDIDATE SIMILARITY
    # ---------------------------------------------

    MINIMUM_SIMILARITY = 0.60

    matches: list[dict[str, Any]] = []

    for result in results:
        historical_case = {
            "hw_systems": (
                result.get("hw_systems")
                or []
            ),
            "sw_systems": (
                result.get("sw_systems")
                or []
            ),
            "issues": (
                result.get("issues")
                or []
            ),
        }

        historical_context = (
            _build_issue_embedding_text(
                historical_case
            )
        )

        historical_vector = _generate_embedding(
            historical_context
        )

        similarity = _cosine_similarity(
            query_vector,
            historical_vector,
        )

        logger.info(
            "Current group %s vs historical group %s "
            "cosine similarity: %.4f",
            group_id,
            result["group_id"],
            similarity,
        )

        if similarity < MINIMUM_SIMILARITY:
            logger.info(
                "Historical group %s rejected as weak match.",
                result["group_id"],
            )
            continue

        matches.append(
            {
                "historical_group_id": int(
                    result["group_id"]
                ),
                "hw_systems": result.get(
                    "hw_systems",
                    [],
                ),
                "sw_systems": result.get(
                    "sw_systems",
                    [],
                ),
                "issues": result.get(
                    "issues",
                    [],
                ),
                "resolution_comment": result.get(
                    "resolution_comment",
                    "",
                ),
                "resolved_by": result.get(
                    "resolved_by"
                ),
                "resolved_at": _json_safe(
                    result.get("resolved_at")
                ),
                "semantic_similarity": round(
                    similarity,
                    4,
                ),
                "search_score": result.get(
                    "@search.score"
                ),
            }
        )

    # ---------------------------------------------
    # FINAL RANKING
    # ---------------------------------------------

    matches.sort(
        key=lambda match: match[
            "semantic_similarity"
        ],
        reverse=True,
    )

    matches = matches[:top_k]

    return {
        "current_group_id": group_id,
        "query_context": {
            "hw_systems": hw_systems,
            "sw_systems": sw_systems,
            "issues": issues,
        },
        "minimum_similarity": (
            MINIMUM_SIMILARITY
        ),
        "matches_found": len(
            matches
        ),
        "matches": matches,
    }

@mcp.tool()
def reassign_group(group_id: int, new_employee_id: int, reason: str = "") -> dict[str, Any]:
    """Append a Re-assigned event for the old employee, then Assigned for the new employee."""
    group_id = int(group_id)
    new_employee_id = int(new_employee_id)
    reason = _clean_text(reason)

    latest_cte = _latest_assignment_cte()
    current_query = latest_cte + sql.SQL(
        """
        SELECT employee_id, group_id, status
        FROM latest_assignment
        WHERE group_id = %s
          AND status IN ('Assigned', 'WIP', 'Pause')
        LIMIT 1
        """
    )

    with db_connection() as conn:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(current_query, (group_id,))
                current = cur.fetchone()
                if not current:
                    raise ValueError(f"Group {group_id} has no active assignment to reassign.")
                if current["employee_id"] == new_employee_id:
                    return {
                        "group_id": group_id,
                        "employee_id": current["employee_id"],
                        "new_employee_id": new_employee_id,
                        "changed": False,
                        "reason": reason,
                        "confirmation": "Group is already assigned to that employee.",
                    }
                reassign_event = _insert_assignment_event(
                    cur,
                    employee_id=current["employee_id"],
                    group_id=group_id,
                    status="Re-assigned",
                    comment=reason,
                )
                assign_event = _insert_assignment_event(
                    cur,
                    employee_id=new_employee_id,
                    group_id=group_id,
                    status="Assigned",
                    comment=reason or "Assigned after re-assignment.",
                )
            conn.commit()
            return {
                "group_id": group_id,
                "previous_employee_id": current["employee_id"],
                "employee_id": new_employee_id,
                "changed": True,
                "reason": reason,
                "reassign_event": reassign_event,
                "assignment_event": assign_event,
            }
        except Exception:
            conn.rollback()
            logger.exception("Unable to reassign group %s", group_id)
            raise


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "sse")
    logger.info("Starting MCP server on %s:%s using %s", MCP_HOST, MCP_PORT, transport)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
