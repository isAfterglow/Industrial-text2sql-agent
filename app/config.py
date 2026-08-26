from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目统一配置。

    默认读取项目根目录下的 .env 文件。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    ENVIRONMENT: str = "development"
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
    PROMETHEUS_MULTIPROC_DIR: str = ""

    # ==================================================
    # LLM
    # ==================================================

    LLM_API_KEY: str
    LLM_MODEL: str
    LLM_BASE_URL: str
    # Each stage has its own budget.  The case-level evaluator remains the
    # final circuit breaker, while this prevents a local model stall from
    # consuming the entire repair budget.
    LLM_REQUEST_TIMEOUT_SECONDS: int = 15
    LLM_7B_API_KEY: str = ""
    LLM_7B_MODEL: str = "qwen2.5:7b"
    LLM_7B_BASE_URL: str = ""
    LLM_7B_REQUEST_TIMEOUT_SECONDS: int = 30
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = ""
    DEEPSEEK_BASE_URL: str = ""
    DEEPSEEK_REQUEST_TIMEOUT_SECONDS: int = 30
    # Optional OpenAI-compatible env file. This avoids duplicating API keys in
    # the project .env while keeping the router portable across deployments.
    DEEPSEEK_ENV_FILE: str = ""

    # ==================================================
    # Resin Database
    # ==================================================

    RESIN_DB_HOST: str
    RESIN_DB_PORT: int = 3306
    RESIN_DB_USER: str
    RESIN_DB_PASSWORD: str
    RESIN_DB_NAME: str

    # ==================================================
    # Table names
    # ==================================================

    RESIN_TABLE_STATIC: str
    RESIN_TABLE_MATERIAL_THERMAL_PROPERTY: str
    RESIN_TABLE_THERMAL_RESPONSE: str

    # ==================================================
    # Text2SQL
    # ==================================================

    # 单次查询最多返回多少行
    SQL_MAX_ROWS: int = 200

    # 数据库查询超时时间
    SQL_TIMEOUT_SECONDS: int = 10

    # SQL校验或执行失败后，最多允许LLM修复多少次
    SQL_MAX_REPAIR_ATTEMPTS: int = 3
    # Agent-loop circuit breakers. These are separate from LangGraph's
    # recursion limit so repeated plan/repair cycles become a typed failure.
    AGENT_MAX_GRAPH_STEPS: int = 32
    AGENT_MAX_SAME_ERROR_REPEATS: int = 2
    AGENT_MAX_TOTAL_MODEL_TIME_SECONDS: int = 90

    # ==================================================
    # Session memory (Redis preferred, SQLite fallback)
    # ==================================================
    SESSION_STORE_MODE: str = "auto"
    REDIS_URL: str = ""
    SESSION_MEMORY_TTL_SECONDS: int = 86400
    SESSION_MEMORY_DB_PATH: str = "data/session_memory.sqlite3"

    # off: no gate; risk: advanced/free-form plans; always: every valid SQL.
    APPROVAL_MODE: str = "risk"
    AGENT_TASK_DB_PATH: str = "data/agent_tasks.sqlite3"
    AGENT_MAX_CONCURRENT_TASKS: int = 4
    AGENT_AUTH_DB_PATH: str = "data/agent_auth.sqlite3"
    JWT_SECRET: str = "change-this-in-production"
    JWT_EXPIRE_MINUTES: int = 480
    AUTH_BOOTSTRAP_DEMO_USERS: bool = True
    AUTH_DEMO_PASSWORD: str = "agent-demo-password"
    TASK_QUEUE_MODE: str = "auto"  # auto, redis, local
    TASK_QUEUE_NAME: str = "text2sql-agent"
    TASK_JOB_TIMEOUT_SECONDS: int = 180
    USER_TASKS_PER_MINUTE: int = 20
    USER_MAX_ACTIVE_TASKS: int = 3
    MODEL_PRIMARY_3B_CONCURRENCY: int = 1
    MODEL_FALLBACK_7B_CONCURRENCY: int = 1
    MODEL_DEEPSEEK_CONCURRENCY: int = 3

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        if self.ENVIRONMENT.lower() in {"production", "prod"}:
            if self.JWT_SECRET == "change-this-in-production" or len(self.JWT_SECRET) < 32:
                raise ValueError("JWT_SECRET must be a non-default secret of at least 32 characters in production")
            if self.AUTH_BOOTSTRAP_DEMO_USERS:
                raise ValueError("AUTH_BOOTSTRAP_DEMO_USERS must be false in production")
            if not self.CORS_ORIGINS.strip():
                raise ValueError("CORS_ORIGINS must be configured in production")
        return self

    @property
    def allowed_tables(self) -> tuple[str, ...]:
        """Text2SQL允许访问的表白名单。"""

        return (
            self.RESIN_TABLE_STATIC,
            self.RESIN_TABLE_MATERIAL_THERMAL_PROPERTY,
            self.RESIN_TABLE_THERMAL_RESPONSE,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """缓存配置对象，避免每个节点重复读取.env。"""

    return Settings()
