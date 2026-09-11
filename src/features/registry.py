"""Feature registry: a declarative record of every generated feature name,
its source attribute, window, and operation (docs/dataset_strategy.md
section 13). Written to data/audit/feature_registry/ on every gold-feature
build for reproducibility and auditability.
"""

from __future__ import annotations

from pydantic import BaseModel


class FeatureRegistryEntry(BaseModel):
    feature_name: str
    source_attribute: str
    window_days: int | None = None
    operation: str
    partition_by: str = "drive_id"
    order_by: str = "date"
    version: int = 1
    dtype: str = "float32"


def build_registry(
    attributes: list[str],
    windows_days: tuple[int, ...],
    spike_thresholds: dict[str, float],
) -> list[FeatureRegistryEntry]:
    entries: list[FeatureRegistryEntry] = []
    for attr in attributes:
        for window in windows_days:
            for op in ("mean", "median", "min", "max", "std", "range"):
                entries.append(
                    FeatureRegistryEntry(
                        feature_name=f"{attr}_{window}d_{op}",
                        source_attribute=attr,
                        window_days=window,
                        operation=op,
                    )
                )
            entries.append(
                FeatureRegistryEntry(
                    feature_name=f"{attr}_{window}d_delta",
                    source_attribute=attr,
                    window_days=window,
                    operation="delta",
                )
            )
            entries.append(
                FeatureRegistryEntry(
                    feature_name=f"{attr}_{window}d_slope",
                    source_attribute=attr,
                    window_days=window,
                    operation="linear_slope_approx",
                )
            )
            entries.append(
                FeatureRegistryEntry(
                    feature_name=f"{attr}_{window}d_positive_count",
                    source_attribute=attr,
                    window_days=window,
                    operation="positive_day_count",
                )
            )
        entries.append(
            FeatureRegistryEntry(
                feature_name=f"{attr}_acceleration_7d_vs_30d",
                source_attribute=attr,
                operation="slope_delta",
            )
        )
        entries.append(
            FeatureRegistryEntry(
                feature_name=f"{attr}_zero_to_nonzero",
                source_attribute=attr,
                operation="zero_to_nonzero_flag",
            )
        )
        if attr in spike_thresholds:
            for window in windows_days:
                entries.append(
                    FeatureRegistryEntry(
                        feature_name=f"{attr}_{window}d_spike_count",
                        source_attribute=attr,
                        window_days=window,
                        operation="spike_count",
                        dtype="int16",
                    )
                )

    entries.append(
        FeatureRegistryEntry(
            feature_name="drive_age_days",
            source_attribute="date",
            operation="lifecycle",
        )
    )
    entries.append(
        FeatureRegistryEntry(
            feature_name="feature_confidence",
            source_attribute="telemetry_coverage_30d",
            operation="feature_confidence",
        )
    )
    return entries
