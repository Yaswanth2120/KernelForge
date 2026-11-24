test:
	python -m pytest -m "not gpu and not slow"
lint:
	ruff check .
gpu-test:
	bash scripts/gpu_session.sh test
bench-all:
	bash scripts/gpu_session.sh bench all
report:
	@echo "Report generation is not implemented yet"; exit 2
