import unittest

from app.schema import _load_profile, set_active_profile, get_schema_catalog, match_question_semantic_columns
from app.sql_guard import validate_and_normalize_sql


class ProfilePortabilityTests(unittest.TestCase):
    """The core graph consumes profile contracts, not domain-specific code."""

    def test_profiles_have_portable_contract(self):
        required = {"name", "tables", "relationships", "semantic_terms", "policy"}
        for name in ("resin", "steel_industry", "ecommerce"):
            profile = _load_profile(name)
            self.assertTrue(required.issubset(profile))
            self.assertTrue(profile["tables"])
            self.assertTrue(profile["relationships"])
            self.assertTrue(profile["policy"].get("allowed_tables"))

    def test_switching_profile_changes_catalog_without_graph_changes(self):
        set_active_profile("resin")

    def test_ecommerce_profile_supports_linking_join_and_policy(self):
        set_active_profile("ecommerce")
        try:
            catalog = get_schema_catalog()
            self.assertEqual(set(catalog["policy"]["allowed_tables"]), {"orders", "customers", "products"})
            matches = match_question_semantic_columns("查询订单金额最高的客户名称")
            self.assertEqual(set(matches), {"order_amount", "customer_name"})
            sql = "SELECT c.customer_name, SUM(o.order_amount) AS total_amount FROM orders o JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.customer_name ORDER BY total_amount DESC LIMIT 5"
            result = validate_and_normalize_sql(sql, set(catalog["policy"]["allowed_tables"]), 200, question="查询订单金额最高的前5个客户名称")
            self.assertTrue(result.valid, result.error)
            sensitive = validate_and_normalize_sql("SELECT email FROM customers", set(catalog["policy"]["allowed_tables"]), 200, question="查询客户邮箱")
            self.assertTrue(sensitive.valid)
            forbidden = validate_and_normalize_sql("SELECT * FROM audit_log", set(catalog["policy"]["allowed_tables"]), 200, question="查询审计表")
            self.assertFalse(forbidden.valid)
        finally:
            set_active_profile("resin")
        resin_tables = set(get_schema_catalog()["tables"])
        set_active_profile("steel_industry")
        steel_tables = set(get_schema_catalog()["tables"])
        self.assertNotEqual(resin_tables, steel_tables)
        self.assertTrue(steel_tables)
        set_active_profile("resin")


if __name__ == "__main__":
    unittest.main()
