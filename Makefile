PYTHON ?= python

.PHONY: reproduce check

reproduce:
	$(PYTHON) -m analysis.reproduce

check:
	$(PYTHON) tools/check_package.py
