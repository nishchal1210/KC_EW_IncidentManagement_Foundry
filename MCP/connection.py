import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def get_connection():
    return psycopg2.connect(
        host=_env("POSTGRES_HOST", "DB_HOST"),
        database=_env("POSTGRES_DB", "DB_NAME"),
        user=_env("POSTGRES_USER", "DB_USER"),
        password=_env("POSTGRES_PASSWORD", "DB_PASS"),
        port=_env("POSTGRES_PORT", "DB_PORT", default="5432"),
        sslmode=_env("POSTGRES_SSLMODE", "DB_SSLMODE", default="require"),
        connect_timeout=int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10")),
        application_name=os.getenv("POSTGRES_APP_NAME", "incident-mcp"),
    )
