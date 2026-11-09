import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor


load_dotenv()
SCHEMA = os.getenv("POSTGRES_SCHEMA", "ew-kc-schema")
INCIDENT_TABLE = os.getenv("INCIDENT_TABLE", "t_incidents")
ASSIGNMENT_TABLE = os.getenv("ASSIGNMENT_TABLE", "t_group_assignment")
ASSIGNMENT_VIEW = os.getenv("ASSIGNMENT_VIEW", "v_group_assignment")
EMPLOYEE_TABLE = os.getenv("EMPLOYEE_TABLE", "m_employee")

ACTIVE_ASSIGNMENT_STATUSES = ("Assigned", "WIP", "Pause")
TERMINAL_STATUSES = ("Closed", "Cancelled")
_pool: pool.ThreadedConnectionPool | None = None


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"')
    return default


def _get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(
            1,
            5,
            host=_env("POSTGRES_HOST", "DB_HOST"),
            dbname=_env("POSTGRES_DB", "DB_NAME"),
            user=_env("POSTGRES_USER", "DB_USER"),
            password=_env("POSTGRES_PASSWORD", "DB_PASS"),
            port=_env("POSTGRES_PORT", "DB_PORT", default="5432"),
            sslmode=_env("POSTGRES_SSLMODE", "DB_SSLMODE", default="require"),
            connect_timeout=10,
            application_name="incident-streamlit-ui",
        )
    return _pool


def _reset_pool() -> None:
    global _pool
    if _pool is None:
        return
    try:
        _pool.closeall()
    finally:
        _pool = None


def _is_transient_connection_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cursor already closed",
            "connection already closed",
            "connection not open",
            "server closed the connection",
            "terminating connection",
        )
    )


@contextmanager
def _connection():
    global _pool
    connection = _get_pool().getconn()
    close_connection = False
    try:
        yield connection
    except Exception:
        try:
            connection.rollback()
        except Exception:
            close_connection = True
        raise
    finally:
        try:
            _get_pool().putconn(connection, close=close_connection)
        except pool.PoolError:
            try:
                connection.close()
            finally:
                _reset_pool()


def _table(name: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(SCHEMA), sql.Identifier(name))


def _read(query: sql.Composable, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    for attempt in range(2):
        try:
            with _connection() as connection:
                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
            return pd.DataFrame(rows)
        except Exception as exc:
            if attempt == 0 and _is_transient_connection_error(exc):
                _reset_pool()
                continue
            raise
    return pd.DataFrame()


@lru_cache(maxsize=64)
def _columns(table_name: str) -> frozenset[str]:
    query = sql.SQL(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        """
    )
    rows = _read(query, (SCHEMA, table_name))
    if rows.empty:
        return frozenset()
    return frozenset(rows["column_name"].tolist())


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
    return not _read(query, (SCHEMA, name, SCHEMA, name)).empty


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


def _assignment_sort_sql(alias: str = "tga") -> sql.Composed:
    pieces = [
        sql.SQL("{}.{} DESC NULLS LAST").format(
            sql.Identifier(alias), sql.Identifier(_assignment_time_column())
        )
    ]
    if "id" in _columns(ASSIGNMENT_TABLE):
        pieces.append(sql.SQL("{}.{} DESC").format(sql.Identifier(alias), sql.Identifier("id")))
    return sql.SQL(", ").join(pieces)


def _latest_assignment_cte() -> sql.Composed:
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
        assignment_table=_table(ASSIGNMENT_TABLE),
        time_column=sql.Identifier(_assignment_time_column()),
        sort_sql=_assignment_sort_sql("tga"),
    )


def get_employees() -> pd.DataFrame:
    query = sql.SQL(
        "SELECT employee_id, employee_name FROM {employees} ORDER BY employee_name"
    ).format(employees=_table(EMPLOYEE_TABLE))
    return _read(query)


def get_assigned_groups(employee_id: int, include_completed: bool = False) -> pd.DataFrame:
    latest_cte = _latest_assignment_cte()
    query = latest_cte + sql.SQL(
        """
        SELECT
            la.group_id,
            la.status,
            la.datetime AS assigned_datetime,
            NULL::timestamp AS completed_datetime
        FROM latest_assignment la
        WHERE la.employee_id = %s
          AND (%s OR la.status IN ('Assigned', 'WIP', 'Pause'))
        ORDER BY la.datetime DESC NULLS LAST, la.group_id
        """
    )
    return _read(query, (int(employee_id), bool(include_completed)))


def get_group_incidents(group_id: int) -> pd.DataFrame:
    query = sql.SQL(
        """
        SELECT incident_number, group_id, "timestamp", hw_system, sw_system,
               issue, source, severity, status
        FROM {incidents}
        WHERE group_id = %s
        ORDER BY "timestamp" DESC NULLS LAST
        """
    ).format(incidents=_table(INCIDENT_TABLE))
    return _read(query, (int(group_id),))


def get_group_overview() -> pd.DataFrame:
    latest_cte = _latest_assignment_cte()
    query = latest_cte + sql.SQL(
        """
        , incident_counts AS (
            SELECT
                ti.group_id,
                COUNT(*) AS incident_number,
                MIN(ti."timestamp") AS first_timestamp,
                MAX(ti."timestamp") AS last_timestamp
            FROM {incidents} ti
            GROUP BY ti.group_id
        )
        SELECT
            la.group_id,
            COALESCE(me.employee_name, 'Unassigned') AS worker_name,
            la.status,
            COALESCE(ic.incident_number, 0) AS incident_number,
            COALESCE(ic.first_timestamp, la.datetime) AS first_timestamp,
            COALESCE(ic.last_timestamp, la.datetime) AS last_timestamp
        FROM latest_assignment la
        LEFT JOIN incident_counts ic ON ic.group_id = la.group_id
        LEFT JOIN {employees} me ON me.employee_id = la.employee_id
        ORDER BY la.group_id
        """
    ).format(incidents=_table(INCIDENT_TABLE), employees=_table(EMPLOYEE_TABLE))
    return _read(query)


def get_assignment_timeline(group_id: int) -> pd.DataFrame:
    time_column = _assignment_time_column()
    comment_column = _first_column(ASSIGNMENT_TABLE, ("comment", "reason"))
    comment_expr = (
        sql.SQL("{} AS comment").format(sql.Identifier(comment_column))
        if comment_column
        else sql.SQL("''::text AS comment")
    )
    query = sql.SQL(
        """
        SELECT
            tga.employee_id,
            me.employee_name,
            tga.group_id,
            tga.{time_column} AS datetime,
            {comment_expr},
            tga.status
        FROM {assignment_table} tga
        LEFT JOIN {employees} me ON me.employee_id = tga.employee_id
        WHERE tga.group_id = %s
        ORDER BY tga.{time_order} DESC NULLS LAST
        """
    ).format(
        assignment_table=_table(ASSIGNMENT_TABLE),
        employees=_table(EMPLOYEE_TABLE),
        time_column=sql.Identifier(time_column),
        time_order=sql.Identifier(time_column),
        comment_expr=comment_expr,
    )
    return _read(query, (int(group_id),))


def get_v_incident_details() -> pd.DataFrame:
    return get_group_overview()


def get_grp_incident_df(group_id: int) -> pd.DataFrame:
    incidents = get_group_incidents(group_id)
    columns = [column for column in ("incident_number", "timestamp") if column in incidents.columns]
    return incidents[columns] if columns else incidents


def getEmployee_data() -> pd.DataFrame:
    return get_employees()


def AssignedGroupIDs(employee_id: int) -> pd.DataFrame:
    return get_assigned_groups(employee_id)


def FetchAllIncidents(group_id: int) -> pd.DataFrame:
    return get_group_incidents(group_id)
