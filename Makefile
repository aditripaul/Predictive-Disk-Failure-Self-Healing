.PHONY: install lint test data-check \
	ingest-backblaze ingest-smartz ingest-synthetic-stub build-silver build-features build-labels train \
	agent-demo dashboard api final-report

install:
	uv sync

lint:
	uv run ruff check .
	uv run mypy src data_contracts

test:
	uv run pytest

data-check:
	uv run pytest tests/golden

ingest-backblaze:
	uv run python pipelines/ingest_backblaze.py

ingest-smartz:
	uv run python pipelines/ingest_smartz.py

ingest-synthetic-stub:
	uv run python pipelines/ingest_synthetic_stub.py

build-silver:
	uv run python pipelines/build_silver.py

build-features:
	uv run python pipelines/build_gold_features.py

build-labels:
	uv run python pipelines/build_labels.py

train:
	uv run python pipelines/train_model.py

agent-demo:
	uv run python -m src.agent.demo

dashboard:
	uv run streamlit run src/dashboards/app.py

api:
	uv run uvicorn src.api.main:app --reload

final-report:
	uv run python pipelines/generate_final_report.py
