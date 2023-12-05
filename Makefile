dev-setup:
	poetry install --with dev
	
format: dev-setup
	poetry run black .

test-setup:
	poetry install --with dev,integrations

test: test-setup
	poetry run pytest tests/

install:
	poetry install

build:
	poetry build

update-readme:
	poetry run jupyter nbconvert --to markdown --output README.md main.ipynb 

