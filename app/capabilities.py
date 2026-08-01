"""Profile-owned query capability registry.

The Agent asks this registry which constrained families a Profile supports;
domain compilers remain responsible for SQL semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileCapabilities:
    profile: str
    entity_key: str
    relation_families: frozenset[str]
    analytical_families: frozenset[str]


_REGISTRY = {
    "resin": ProfileCapabilities(
        profile="resin",
        entity_key="sample_id",
        relation_families=frozenset({"one_to_one_join", "per_sample_temporal_aggregate", "material_plan_temporal_aggregate"}),
        analytical_families=frozenset({"static_filter_temporal_aggregate"}),
    ),
    "steel_industry": ProfileCapabilities(
        profile="steel_industry",
        entity_key="reading_id",
        relation_families=frozenset({"profile_fact_query"}),
        analytical_families=frozenset({"group_topk", "period_change", "group_outlier", "cumulative_share", "conditional_comparison", "group_threshold", "correlation", "group_share", "rising_sequence"}),
    ),
}


def get_profile_capabilities(profile: str) -> ProfileCapabilities:
    return _REGISTRY.get(profile, _REGISTRY["resin"])


def capability_family(profile: str, query_type: str) -> str:
    capabilities = get_profile_capabilities(profile)
    return "relation_query_plan" if query_type in capabilities.relation_families else "query_spec"
