.PHONY: install lint format typecheck test cov bench check live build clean doctor

install:      ## dev install
	pip install -e ".[dev]"
lint:         ## ruff lint + format check
	ruff check . && ruff format --check .
format:       ## auto-fix style
	ruff check --fix . && ruff format .
typecheck:
	mypy
test:         ## fast unit tests
	pytest -q
cov:          ## tests + coverage gate (same as CI)
	pytest -q --cov=csva --cov-branch --cov-report=term-missing --cov-fail-under=90
bench:        ## rule-baseline regression gate (same as CI)
	python -m csva benchmark --repeats 5 --min-success 1.0 --min-accuracy 1.0
check: lint typecheck cov bench   ## everything CI runs, locally
doctor:       ## diagnose Gemini key / model / quota
	python -m csva doctor
live:         ## real Gemini smoke tests (needs GEMINI_API_KEY)
	pytest -m live -o addopts='' -q
build:
	python -m build && twine check --strict dist/*
clean:
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml junit.xml *.egg-info src/*.egg-info calls.db
