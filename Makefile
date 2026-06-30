.PHONY: install test lint evaluate dashboard clean

install:
	pip install -e .[dev,ner]

test:
	pytest --cov=demne --tb=short -q

lint:
	ruff check .
	black --check .

evaluate:
	python main.py metrics
	python main.py tree --no-visualize

dashboard:
	streamlit run dashboard/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
