from app.config import Settings
import pytest


def _values(**overrides):
    values = {
        "LLM_API_KEY": "x", "LLM_MODEL": "x", "LLM_BASE_URL": "http://localhost",
        "RESIN_DB_HOST": "localhost", "RESIN_DB_USER": "reader", "RESIN_DB_PASSWORD": "x", "RESIN_DB_NAME": "db",
        "RESIN_TABLE_STATIC": "material_static", "RESIN_TABLE_MATERIAL_THERMAL_PROPERTY": "material_thermal_property",
        "RESIN_TABLE_THERMAL_RESPONSE": "thermal_response",
    }
    values.update(overrides)
    return values


def test_production_rejects_default_jwt_secret():
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings(**_values(ENVIRONMENT="production", AUTH_BOOTSTRAP_DEMO_USERS=False))


def test_production_rejects_demo_bootstrap():
    with pytest.raises(ValueError, match="AUTH_BOOTSTRAP"):
        Settings(**_values(ENVIRONMENT="production", JWT_SECRET="a" * 40))

