build:
	python -m build

install:
	pip install -e .

lint:
	ruff check src --ignore E731,E741,F405
	ruff format src --diff
	mypy src

lint-fix:
	ruff check src --ignore E731,E741,F405 --fix
	ruff format src

clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf src/GQEMTS/__pycache__
	rm -rf build
	rm -rf dist
	rm -rf GQEMTS.egg-info
	rm -rf src/GQEMTS.egg-info
