from functools import lru_cache

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

    # ==================================================
    # LLM
    # ==================================================

    LLM_API_KEY: str
    LLM_MODEL: str
    LLM_BASE_URL: str

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
    SQL_MAX_REPAIR_ATTEMPTS: int = 1

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