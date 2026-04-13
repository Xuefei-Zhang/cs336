PYTHON ?= python

.PHONY: setup lint test smoke e2e obs-up obs-down serve loadtest

setup:
	$(PYTHON) -m pip install -e .

lint:
	ruff check .

test:
	pytest -q

smoke:
	@printf 'Smoke target placeholder\n'

e2e:
	@printf 'E2E target placeholder\n'

obs-up:
	@printf 'Observability up placeholder\n'

obs-down:
	@printf 'Observability down placeholder\n'

serve:
	@printf 'Serve target placeholder\n'

loadtest:
	@printf 'Loadtest target placeholder\n'
