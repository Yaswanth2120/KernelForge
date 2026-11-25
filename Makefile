test:
	python -m pytest -m "not gpu and not slow"
lint:
	ruff check .
gpu-test:
	bash scripts/gpu_session.sh test
bench-all:
	bash scripts/gpu_session.sh bench all
report:
	kernelforge report --raw reports/raw --out reports
