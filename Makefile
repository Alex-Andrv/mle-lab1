VENV := .venv

.venv:
	poetry install --no-root
	poetry check

setup: .venv

split:
	poetry run python src/split.py

test:
	poetry run coverage run -m pytest tests -W ignore::DeprecationWarning
	poetry run coverage report