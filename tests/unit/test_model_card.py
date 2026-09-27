from src.models.model_card import build_model_card, render_model_card_markdown


def _sample_card(**overrides):
    defaults = dict(
        horizon_days=14,
        model_params={"objective": "binary", "num_leaves": 31},
        model_version=1,
        feature_registry_version=1,
        dataset_version=None,
        feature_columns=["reallocated_sector_count_7d_mean", "drive_age_days"],
        threshold_result={"threshold": 0.8, "precision": 0.96, "recall": 0.4},
        validation_metrics={"auprc": 0.7, "precision": 0.96, "recall": 0.4},
        test_metrics={"auprc": 0.65, "precision": 0.94, "recall": 0.38},
        shap_top_features=[{"feature": "drive_age_days", "importance": 0.5}],
        train_row_count=1000,
    )
    defaults.update(overrides)
    return build_model_card(**defaults)


def test_build_model_card_reports_unversioned_dataset_when_absent():
    card = _sample_card(dataset_version=None)
    assert card["training_data"]["dataset_version"] == "unversioned"
    assert card["training_data"]["feature_count"] == 2
    assert card["model_details"]["primary_horizon_days"] == 14


def test_build_model_card_defaults_model_type_to_lightgbm():
    card = _sample_card()
    assert card["model_details"]["type"] == "lightgbm"


def test_build_model_card_reports_xgboost_model_type():
    card = _sample_card(model_type="xgboost")
    assert card["model_details"]["type"] == "xgboost"


def test_build_model_card_carries_dataset_version_when_present():
    card = _sample_card(dataset_version="2026-01-01_v3")
    assert card["training_data"]["dataset_version"] == "2026-01-01_v3"


def test_build_model_card_includes_evaluation_and_explainability():
    card = _sample_card()
    assert card["evaluation"]["test_metrics"]["auprc"] == 0.65
    assert card["explainability"]["method"] == "shap.TreeExplainer"
    assert card["explainability"]["top_features"][0]["feature"] == "drive_age_days"
    assert len(card["limitations_and_risks"]) >= 1
    assert card["evaluation"]["test_warning_lead_time"] is None


def test_build_model_card_includes_warning_lead_time_when_provided():
    card = _sample_card(test_warning_lead_time={"mean_lead_time_days": 4.5})
    assert card["evaluation"]["test_warning_lead_time"]["mean_lead_time_days"] == 4.5


def test_render_model_card_markdown_contains_key_sections():
    card = _sample_card()
    markdown = render_model_card_markdown(card)
    assert "# Model Card" in markdown
    assert "## Model Details" in markdown
    assert "## Training Data" in markdown
    assert "## Evaluation" in markdown
    assert "## Explainability" in markdown
    assert "## Limitations and Risks" in markdown
    assert "drive_age_days" in markdown
