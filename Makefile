PYTHON ?= .venv/bin/python

.PHONY: test
test:
	$(PYTHON) -m pytest -q
