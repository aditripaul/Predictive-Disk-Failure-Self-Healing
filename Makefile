.PHONY: install lint test data-check \
	ingest-backblaze ingest-smartz ingest-synthetic-stub build-silver build-features build-labels train \
	agent-demo dashboard api final-report

install:
	uv sync

lint:
	ruff check .
	mypy src data_contracts

test:
	pytest

data-check:
	pytest tests/golden

ingest-backblaze:
	python pipelines/ingest_backblaze.py

ingest-smartz:
	python pipelines/ingest_smartz.py

ingest-synthetic-stub:
	python pipelines/ingest_synthetic_stub.py

build-silver:
	python pipelines/build_silver.py

build-features:
	python pipelines/build_gold_features.py

build-labels:
	python pipelines/build_labels.py

train:
	python pipelines/train_model.py

agent-demo:
	python -m src.agent.demo

dashboard:
	streamlit run src/dashboards/app.py

api:
	uvicorn src.api.main:app --reload

final-report:
	python pipelines/generate_final_report.py
