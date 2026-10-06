"""Feature registry: a declarative record of every generated feature name,
its source attribute, window, and operation (docs/dataset_strategy.md
section 13). Written to data/audit/feature_registry/ on every gold-feature
build for reproducibility and auditability.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.features.cross_vendor import ATTRIBUTE_RATIOS


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
    *,
    secondary_attributes: list[str] | None = None,
    secondary_windows_days: tuple[int, ...] = (),
    gold_columns: list[str] | None = None,
) -> list[FeatureRegistryEntry]:
    """`secondary_attributes`/`secondary_windows_days` register the light
    family (value + deltas). `gold_columns` (the gold table's column names)
    registers the features whose presence depends on the data - active-defect
    velocity, days-since-last-increase, temperature spike, power-on days -
    only when they were actually produced."""
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

    for numerator, _denominator, name in ATTRIBUTE_RATIOS:
        if numerator in attributes:
            entries.append(
                FeatureRegistryEntry(
                    feature_name=name,
                    source_attribute=numerator,
                    operation="cross_vendor_ratio",
                )
            )
    if "reallocated_sector_count" in attributes:
        entries.append(
            FeatureRegistryEntry(
                feature_name="reallocated_per_capacity",
                source_attribute="reallocated_sector_count",
                operation="cross_vendor_ratio",
            )
        )
    for attr in attributes:
        for window in windows_days:
            entries.append(
                FeatureRegistryEntry(
                    feature_name=f"{attr}_{window}d_model_zscore",
                    source_attribute=attr,
                    window_days=window,
                    operation="model_family_zscore",
                )
            )

    for attr in secondary_attributes or []:
        for window in secondary_windows_days:
            entries.append(
                FeatureRegistryEntry(
                    feature_name=f"{attr}_{window}d_delta",
                    source_attribute=attr,
                    window_days=window,
                    operation="delta",
                )
            )

    for column in gold_columns or []:
        if column.startswith("active_defect_"):
            entries.append(
                FeatureRegistryEntry(
                    feature_name=column,
                    source_attribute="active_defect_total",
                    operation="defect_velocity",
                )
            )
        elif column.endswith("_days_since_last_increase"):
            entries.append(
                FeatureRegistryEntry(
                    feature_name=column,
                    source_attribute=column.removesuffix("_days_since_last_increase"),
                    operation="days_since_last_increase",
                    dtype="int16",
                )
            )
        elif column == "temperature_spike":
            entries.append(
                FeatureRegistryEntry(
                    feature_name=column,
                    source_attribute="temperature_celsius",
                    operation="short_max_minus_long_mean",
                )
            )
        elif column == "power_on_days":
            entries.append(
                FeatureRegistryEntry(
                    feature_name=column, source_attribute="power_on_hours", operation="lifecycle"
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
